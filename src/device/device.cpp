#include "device/device.h"

#include <cerrno>
#include <cstring>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "common/clock.h"
#include "common/cpu.h"

namespace softgpu::device {

namespace {

bool is_copy_op(uint32_t op) {
    return op == SG_OP_COPY_H2D || op == SG_OP_COPY_D2H || op == SG_OP_COPY_D2D;
}
bool is_compute_op(uint32_t op) {
    return op == SG_OP_FILL || op == SG_OP_VADD_F32 || op == SG_OP_GEMM_F32;
}

} // namespace

Device::Device(uint64_t vram_bytes, uint32_t copy_engines, uint32_t channels)
    : num_engines_(1 + copy_engines), num_channels_(channels), vram_(new uint8_t[vram_bytes]),
      vram_size_(vram_bytes) {
    // Touch every page so first-use page faults do not show up as device
    // "busy" time in the first benchmark that runs.
    std::memset(vram_.get(), 0, vram_bytes);
}

Device::~Device() { power_off(); }

void Device::power_on(int cpu) {
    if (running_.exchange(true)) return;
    for (uint32_t e = 0; e < num_engines_; ++e) engines_.emplace_back(&Device::run, this, e);
#ifdef __linux__
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(engines_[0].native_handle(), sizeof set, &set);
    }
#else
    (void)cpu;
#endif
}

void Device::power_off() {
    if (!running_.exchange(false)) return;
    for (auto& t : engines_)
        if (t.joinable()) t.join();
    engines_.clear();
}

void Device::reset_stats() {
    for (uint32_t e = 0; e < num_engines_; ++e) {
        auto& st = stats_[e];
        st.busy_cycles.store(0, std::memory_order_relaxed);
        st.wait_cycles.store(0, std::memory_order_relaxed);
        st.idle_cycles.store(0, std::memory_order_relaxed);
        st.cmds_executed.store(0, std::memory_order_relaxed);
        st.batches.store(0, std::memory_order_relaxed);
        st.stats_gen.fetch_add(1, std::memory_order_release);
    }
}

bool Device::vram_range_ok(uint64_t off, uint64_t len) const {
    return off <= vram_size_ && len <= vram_size_ - off;
}

// One engine's loop: a runlist over its channels. Each round visits every
// channel with published work and runs it until it empties or its head is a
// WAIT_FENCE that is not yet satisfied — then moves on rather than blocking,
// so one stream's semaphore never stalls another stream's commands. Still
// always-on and spinning when there is nothing to do (stage 7 changes that).
//
// Time accounting: executing = busy; a round that made no progress is
// charged to `wait` if some channel had pending work (all heads blocked) or
// to `idle` if none did.
void Device::run(uint32_t engine) {
    EngineStats& st = stats_[engine];
    struct Cursor { const sg_cmd* ring; uint64_t mask; uint64_t get; };
    Cursor cur[SG_MAX_CHANNELS];
    for (uint32_t c = 0; c < num_channels_; ++c) {
        Channel& ch = ch_[engine][c];
        cur[c] = {reinterpret_cast<const sg_cmd*>(ch.ring_base), ch.ring_mask,
                  ch.get.load(std::memory_order_relaxed)};
    }
    uint64_t mark = now_cycles();
    uint32_t gen = st.stats_gen.load(std::memory_order_relaxed);
    bool was_active = false;

    while (running_.load(std::memory_order_relaxed)) {
        bool progress = false, pending = false;
        for (uint32_t c = 0; c < num_channels_; ++c) {
            Channel& ch = ch_[engine][c];
            Cursor& k = cur[c];
            // Acquire pairs with the driver's release store of `put` and
            // makes every slot below it visible to this thread.
            uint64_t put = ch.put.load(std::memory_order_acquire);
            if (put == k.get) continue;
            pending = true;
            while (k.get != put) {
                const sg_cmd& cmd = k.ring[k.get & k.mask];
                int rc = 0;
                if (cmd.opcode == SG_OP_WAIT_FENCE) {
                    // Semaphore acquire. Cannot deadlock: the driver only
                    // publishes waits on fences whose commands were enqueued
                    // earlier (ADR 003).
                    if (cmd.arg0 >= num_engines_ || cmd.arg1 >= num_channels_) {
                        rc = -EINVAL;
                    } else if (ch_[cmd.arg0][cmd.arg1].get.load(std::memory_order_acquire) < cmd.src1) {
                        break; // blocked: leave this channel for now
                    }
                } else {
                    const uint64_t t0 = now_cycles();
                    rc = execute(engine, cmd);
                    st.busy_cycles.fetch_add(now_cycles() - t0, std::memory_order_relaxed);
                }
                if (rc != 0) {
                    int expected = 0;
                    ch.sticky_error.compare_exchange_strong(expected, rc, std::memory_order_relaxed);
                }
                ++k.get;
                progress = true;
                st.cmds_executed.fetch_add(1, std::memory_order_relaxed);
                // Retire each command individually (release publishes its
                // results) so a host or another channel waiting on an early
                // fence is not held up by the rest. Also frees the slot.
                ch.get.store(k.get, std::memory_order_release);
                if (k.get == put) put = ch.put.load(std::memory_order_acquire);
            }
        }

        const uint64_t now = now_cycles();
        if (progress) {
            if (!was_active) st.batches.fetch_add(1, std::memory_order_relaxed);
            was_active = true;
        } else {
            // A stats reset while idle restarts the timer, so time from
            // before the reset is not charged to the new window.
            const uint32_t g = st.stats_gen.load(std::memory_order_acquire);
            if (g != gen) {
                gen = g;
                mark = now;
            }
            (pending ? st.wait_cycles : st.idle_cycles).fetch_add(now - mark, std::memory_order_relaxed);
            was_active = false;
            cpu_relax();
        }
        mark = now;
    }
}

int Device::execute(uint32_t engine, const sg_cmd& c) {
    uint8_t* vram = vram_.get();

    // Engine class check: a real copy engine has no ALUs and a compute engine
    // no DMA. The driver validates this too; the device is the last line.
    if (engine == SG_ENGINE_COMPUTE ? is_copy_op(c.opcode) : is_compute_op(c.opcode))
        return -EINVAL;

    switch (c.opcode) {
    case SG_OP_NOP:
        return 0;

    case SG_OP_FILL:
        if (!vram_range_ok(c.dst, c.size)) return -EFAULT;
        std::memset(vram + c.dst, static_cast<int>(c.value & 0xff), c.size);
        return 0;

    // Host addresses in COPY_H2D/COPY_D2H are trusted, exactly as a DMA
    // engine trusts the bus addresses its driver programs. The driver is
    // responsible for only ever handing it its staging buffers or pinned
    // user memory.
    case SG_OP_COPY_H2D:
        if (!vram_range_ok(c.dst, c.size)) return -EFAULT;
        std::memcpy(vram + c.dst, reinterpret_cast<const void*>(c.src0), c.size);
        return 0;

    case SG_OP_COPY_D2H:
        if (!vram_range_ok(c.src0, c.size)) return -EFAULT;
        std::memcpy(reinterpret_cast<void*>(c.dst), vram + c.src0, c.size);
        return 0;

    case SG_OP_COPY_D2D:
        if (!vram_range_ok(c.dst, c.size) || !vram_range_ok(c.src0, c.size)) return -EFAULT;
        std::memmove(vram + c.dst, vram + c.src0, c.size);
        return 0;

    case SG_OP_VADD_F32: {
        const uint64_t bytes = uint64_t{c.arg0} * sizeof(float);
        if (!vram_range_ok(c.dst, bytes) || !vram_range_ok(c.src0, bytes) ||
            !vram_range_ok(c.src1, bytes))
            return -EFAULT;
        const float* a = reinterpret_cast<const float*>(vram + c.src0);
        const float* b = reinterpret_cast<const float*>(vram + c.src1);
        float* out = reinterpret_cast<float*>(vram + c.dst);
        for (uint32_t i = 0; i < c.arg0; ++i) out[i] = a[i] + b[i];
        return 0;
    }

    case SG_OP_GEMM_F32: {
        const uint64_t m = c.arg0, n = c.arg1, k = c.arg2;
        if (!vram_range_ok(c.src0, m * k * sizeof(float)) ||
            !vram_range_ok(c.src1, k * n * sizeof(float)) ||
            !vram_range_ok(c.dst, m * n * sizeof(float)))
            return -EFAULT;
        const float* a = reinterpret_cast<const float*>(vram + c.src0);
        const float* b = reinterpret_cast<const float*>(vram + c.src1);
        float* out = reinterpret_cast<float*>(vram + c.dst);
        // i-k-j ordering keeps the inner loop streaming over contiguous rows
        // of B and C. Deliberately no blocking; this is the reference engine.
        for (uint64_t i = 0; i < m; ++i) {
            float* crow = out + i * n;
            for (uint64_t j = 0; j < n; ++j) crow[j] = 0.0f;
            for (uint64_t p = 0; p < k; ++p) {
                const float aip = a[i * k + p];
                const float* brow = b + p * n;
                for (uint64_t j = 0; j < n; ++j) crow[j] += aip * brow[j];
            }
        }
        return 0;
    }

    default:
        return -EINVAL;
    }
}

} // namespace softgpu::device
