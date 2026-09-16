#include "device/device.h"

#include <cerrno>
#include <cstring>

#include "common/clock.h"
#include "common/cpu.h"

namespace softgpu::device {

Device::Device(uint64_t vram_bytes)
    : vram_(new uint8_t[vram_bytes]), vram_size_(vram_bytes) {
    // Touch every page so first-use page faults do not show up as device
    // "busy" time in the first benchmark that runs.
    std::memset(vram_.get(), 0, vram_bytes);
}

Device::~Device() { power_off(); }

void Device::power_on() {
    if (running_.exchange(true)) return;
    engine_ = std::thread(&Device::run, this);
}

void Device::power_off() {
    if (!running_.exchange(false)) return;
    if (engine_.joinable()) engine_.join();
}

void Device::reset_stats() {
    regs_.busy_cycles.store(0, std::memory_order_relaxed);
    regs_.idle_cycles.store(0, std::memory_order_relaxed);
    regs_.cmds_executed.store(0, std::memory_order_relaxed);
}

bool Device::vram_range_ok(uint64_t off, uint64_t len) const {
    return off <= vram_size_ && len <= vram_size_ - off;
}

// The engine loop. BASELINE: the engine is always on and always spinning on
// the doorbell — there is no idle state, no clock gating, nothing. Idle
// cycles are therefore 100% "wasted power", which is the point of measuring
// them from day one.
void Device::run() {
    uint32_t seen = regs_.completed.load(std::memory_order_relaxed);
    uint64_t idle_start = now_cycles();

    while (running_.load(std::memory_order_relaxed)) {
        // Acquire pairs with the driver's release on the doorbell store and
        // makes the mailbox contents visible to this thread. Without it this
        // is a real bug on ARM, not a theoretical one.
        if (regs_.doorbell.load(std::memory_order_acquire) == seen) {
            cpu_relax();
            continue;
        }

        const uint64_t exec_start = now_cycles();
        regs_.idle_cycles.fetch_add(exec_start - idle_start, std::memory_order_relaxed);

        const int rc = execute(regs_.mailbox);

        const uint64_t exec_end = now_cycles();
        regs_.busy_cycles.fetch_add(exec_end - exec_start, std::memory_order_relaxed);
        regs_.cmds_executed.fetch_add(1, std::memory_order_relaxed);
        regs_.last_error.store(rc, std::memory_order_relaxed);

        // Release publishes the results (VRAM writes, host DMA writes,
        // last_error) before the driver can observe the ticket.
        ++seen;
        regs_.completed.store(seen, std::memory_order_release);
        idle_start = exec_end;
    }
}

int Device::execute(const sg_cmd& c) {
    uint8_t* vram = vram_.get();

    switch (c.opcode) {
    case SG_OP_NOP:
        return 0;

    case SG_OP_FILL:
        if (!vram_range_ok(c.dst, c.size)) return -EFAULT;
        std::memset(vram + c.dst, static_cast<int>(c.value & 0xff), c.size);
        return 0;

    // Host addresses in COPY_H2D/COPY_D2H are trusted, exactly as a DMA
    // engine trusts the bus addresses its driver programs. The driver is
    // responsible for only ever handing it its own staging buffer.
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
