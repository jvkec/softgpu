// STAGE 3b driver: engines, channels, cross-channel fences.
//
// What changed from stage 3a:
//   * the device has a compute engine and one or more copy engines, each
//     serving several channels (rings) with their own fence counters; SUBMIT
//     names the (engine, channel);
//   * SG_OP_WAIT_FENCE lets a channel block on another channel's fence, which
//     is how the runtime keeps a stream in order across engines;
//   * every fence the driver stores — staging slot reuse, deferred frees and
//     unpins — is now (engine, channel, value), or a snapshot of every
//     channel's PUT; SG_IOC_WAIT with SG_WAIT_ALL waits on such a snapshot.
//
// Still deliberately unchanged: the big driver lock, spinning as the wait
// mechanism, the staging pool geometry, the VRAM allocator.
//
// Deadlock freedom: a WAIT_FENCE is accepted only if its target value is
// <= that engine's PUT at submission time, i.e. the command it waits for was
// enqueued before the WAIT itself. By induction on enqueue order the
// dependency graph is a DAG, so no engine can wait on something behind it.

#include "driver/sg_driver.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "common/cpu.h"
#include "device/device.h"
#include "driver/vram_alloc.h"
#include "softgpu/sg_ioctl.h"

namespace softgpu::driver {
namespace {

constexpr int kFd = 3; // the one and only descriptor we hand out
constexpr uint32_t kDefaultRingDepth = 1024; // chosen from the depth sweep in ADR 001

// Tuning knobs for the decision records, not user-facing settings.
uint64_t env_pow2(const char* name, uint64_t def, uint64_t lo, uint64_t hi) {
    const char* e = std::getenv(name);
    if (!e) return def;
    unsigned long long v = std::strtoull(e, nullptr, 10);
    if (v < lo || v > hi || (v & (v - 1)) != 0) return def;
    return v;
}
uint64_t env_int(const char* name, uint64_t def, uint64_t lo, uint64_t hi) {
    const char* e = std::getenv(name);
    if (!e) return def;
    unsigned long long v = std::strtoull(e, nullptr, 10);
    return (v < lo || v > hi) ? def : v;
}
uint32_t ring_depth_from_env() { return uint32_t(env_pow2("SG_RING_DEPTH", kDefaultRingDepth, 2, 65536)); }
uint32_t staging_slots_from_env() { return uint32_t(env_pow2("SG_STAGING_SLOTS", SG_STAGING_SLOTS, 1, 64)); }
uint64_t staging_chunk_from_env() { return env_pow2("SG_STAGING_CHUNK", SG_STAGING_CHUNK, 4096, 64ull << 20); }
uint32_t copy_engines_from_env() { return uint32_t(env_int("SG_COPY_ENGINES", SG_COPY_ENGINES, 1, SG_MAX_ENGINES - 1)); }
uint32_t channels_from_env() { return uint32_t(env_int("SG_CHANNELS", SG_CHANNELS, 1, SG_MAX_CHANNELS)); }
// SG_DEVICE_CPU=<n> pins the compute engine's thread; -1 (default) leaves it to the OS.
int device_cpu_from_env() {
    const char* e = std::getenv("SG_DEVICE_CPU");
    return e ? std::atoi(e) : -1;
}

struct FreeDeleter { void operator()(void* p) const { std::free(p); } };

struct Fence {
    uint32_t engine = 0;
    uint32_t channel = 0;
    uint64_t value = 0;
};

// A snapshot of every channel's PUT: "everything submitted up to now".
using PutSnapshot = uint64_t[SG_MAX_ENGINES][SG_MAX_CHANNELS];

// Something the user has released that in-flight commands may still touch.
// Safe to recycle once every channel has retired the PUT snapshot.
struct Pending {
    enum Kind { kVram, kPin } kind;
    uint64_t addr;
    uint64_t size;
    PutSnapshot fence;
};

struct Driver {
    std::mutex lock; // still the big driver lock; stage 4 breaks it up
    uint32_t num_engines = 0;
    uint32_t num_channels = 0;
    device::Device dev{SG_VRAM_SIZE, copy_engines_from_env(), channels_from_env()};
    VramAllocator vram{SG_VRAM_SIZE, SG_ALLOC_ALIGN};

    std::unique_ptr<sg_cmd, FreeDeleter> ring[SG_MAX_ENGINES][SG_MAX_CHANNELS];
    uint32_t depth = 0;
    PutSnapshot put = {}; // driver-side mirrors of the PUT registers

    // Staging pool for pageable copies: `slots` chunks of `chunk` bytes, each
    // reusable once the DMA that last used it (slot_fence) has retired.
    std::vector<uint8_t> staging;
    std::vector<Fence> slot_fence;
    uint32_t slots = 0;
    uint64_t chunk = 0;
    uint32_t next_slot = 0; // rotates across submissions, not just within one

    std::map<uint64_t, uint64_t> pins; // pinned host ranges: addr -> size
    std::vector<Pending> pending;      // deferred frees / unpins

    // `waits` is bumped by SG_IOC_WAIT, which runs without the driver lock
    // by design (a thread synchronizing must not block submitters), so it is
    // the one counter that must be atomic. TSan found this in stage 3a.
    std::atomic<uint64_t> waits{0};
    uint64_t submits = 0, stalls = 0, staging_waits = 0;
    uint64_t bytes_h2d = 0, bytes_d2h = 0, bytes_direct = 0, bytes_staged = 0;

    Driver() {
        num_engines = dev.num_engines();
        num_channels = dev.num_channels();
        depth = ring_depth_from_env();
        for (uint32_t e = 0; e < num_engines; ++e)
            for (uint32_t c = 0; c < num_channels; ++c) {
                void* mem = nullptr;
                if (posix_memalign(&mem, 64, size_t{depth} * sizeof(sg_cmd)) != 0) std::abort();
                std::memset(mem, 0, size_t{depth} * sizeof(sg_cmd));
                ring[e][c].reset(static_cast<sg_cmd*>(mem));
                dev.channel(e, c).ring_base = reinterpret_cast<uint64_t>(ring[e][c].get());
                dev.channel(e, c).ring_mask = depth - 1;
            }
        slots = staging_slots_from_env();
        chunk = staging_chunk_from_env();
        staging.assign(size_t{slots} * chunk, 0);
        slot_fence.assign(slots, Fence{});
    }

    uint8_t* slot(uint32_t i) { return staging.data() + size_t{i} * chunk; }
};

std::mutex g_open_lock;
std::unique_ptr<Driver> g_drv;
int g_refs = 0;

inline int sticky(Driver& d) {
    for (uint32_t e = 0; e < d.num_engines; ++e)
        for (uint32_t c = 0; c < d.num_channels; ++c)
            if (int rc = d.dev.channel(e, c).sticky_error.load(std::memory_order_relaxed)) return rc;
    return 0;
}
inline uint64_t retired(Driver& d, uint32_t e, uint32_t c) {
    return d.dev.channel(e, c).get.load(std::memory_order_acquire);
}

// Block until (engine, channel) has retired everything up to `value`.
int wait_fence(Driver& d, Fence f, uint64_t* counter = nullptr) {
    if (retired(d, f.engine, f.channel) < f.value) {
        if (counter) ++*counter;
        else d.waits.fetch_add(1, std::memory_order_relaxed);
        while (retired(d, f.engine, f.channel) < f.value) cpu_relax();
    }
    return sticky(d) ? -EIO : 0;
}

void snapshot_puts(Driver& d, PutSnapshot out) {
    std::memcpy(out, d.put, sizeof(PutSnapshot));
}
bool all_retired(Driver& d, const PutSnapshot fence) {
    for (uint32_t e = 0; e < d.num_engines; ++e)
        for (uint32_t c = 0; c < d.num_channels; ++c)
            if (retired(d, e, c) < fence[e][c]) return false;
    return true;
}

// Block until every channel has retired a PUT snapshot. Lock-free; the
// snapshot itself must have been taken under the lock.
int wait_snapshot(Driver& d, const PutSnapshot fence) {
    if (!all_retired(d, fence)) {
        d.waits.fetch_add(1, std::memory_order_relaxed);
        while (!all_retired(d, fence)) cpu_relax();
    }
    return sticky(d) ? -EIO : 0;
}

// Wait for everything submitted so far. Caller holds d.lock.
int drain(Driver& d) {
    PutSnapshot snap;
    snapshot_puts(d, snap);
    return wait_snapshot(d, snap);
}

// Append one command to a channel's ring and ring its doorbell. Returns the
// fence. Caller holds d.lock, which makes the driver each ring's single
// producer.
uint64_t enqueue(Driver& d, uint32_t engine, uint32_t channel, const sg_cmd& cmd) {
    auto& c = d.dev.channel(engine, channel);
    uint64_t& put = d.put[engine][channel];
    if (put - retired(d, engine, channel) >= d.depth) {
        ++d.stalls;
        while (put - retired(d, engine, channel) >= d.depth) cpu_relax();
    }
    d.ring[engine][channel].get()[put & (d.depth - 1)] = cmd;
    ++put;
    c.put.store(put, std::memory_order_release); // release: slot before doorbell
    return put;
}

// Recycle everything the device has finished with.
void reclaim(Driver& d) {
    if (d.pending.empty()) return;
    auto keep = d.pending.begin();
    for (auto& p : d.pending) {
        if (all_retired(d, p.fence)) {
            if (p.kind == Pending::kVram) d.vram.release(p.addr, p.size);
            // kPin: nothing to give back in the model; a real driver would
            // drop its page references / IOMMU mappings here.
        } else {
            *keep++ = p;
        }
    }
    d.pending.erase(keep, d.pending.end());
}

// Is [addr, addr+size) entirely inside one pinned range?
bool pinned(const Driver& d, uint64_t addr, uint64_t size) {
    auto it = d.pins.upper_bound(addr);
    if (it == d.pins.begin()) return false;
    --it;
    return addr - it->first <= it->second && size <= it->second - (addr - it->first);
}

bool is_copy_op(uint32_t op) {
    return op == SG_OP_COPY_H2D || op == SG_OP_COPY_D2H || op == SG_OP_COPY_D2D;
}
bool is_compute_op(uint32_t op) {
    return op == SG_OP_FILL || op == SG_OP_VADD_F32 || op == SG_OP_GEMM_F32;
}

int do_submit(Driver& d, sg_submit_args& a) {
    sg_cmd cmd = a.cmd;
    const uint32_t eng = a.engine, chn = a.channel;
    if (cmd.opcode >= SG_OP_COUNT || cmd.flags != 0 || cmd.reserved != 0) return -EINVAL;
    if (eng >= d.num_engines || chn >= d.num_channels) return -EINVAL;
    // Engine classes: copies only on copy engines, compute only on engine 0.
    if (eng == SG_ENGINE_COMPUTE ? is_copy_op(cmd.opcode) : is_compute_op(cmd.opcode)) return -EINVAL;
    if (sticky(d)) return -EIO;
    ++d.submits;
    a.out_flags = 0;

    switch (cmd.opcode) {
    case SG_OP_NOP:
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_WAIT_FENCE:
        // Only fences already published may be waited on (see header note).
        if (cmd.arg0 >= d.num_engines || cmd.arg1 >= d.num_channels ||
            cmd.src1 > d.put[cmd.arg0][cmd.arg1])
            return -EINVAL;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_FILL:
        if (!d.vram.owns_range(cmd.dst, cmd.size)) return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_COPY_D2D:
        if (!d.vram.owns_range(cmd.dst, cmd.size) || !d.vram.owns_range(cmd.src0, cmd.size))
            return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;

    case SG_OP_COPY_H2D: {
        if (a.host_ptr == 0 || !d.vram.owns_range(cmd.dst, cmd.size)) return -EFAULT;
        const uint64_t dst = cmd.dst, total = cmd.size;
        d.bytes_h2d += total;

        if (pinned(d, a.host_ptr, total)) {
            // Direct DMA from the user's buffer. Asynchronous: the caller owns
            // the "don't touch it until the fence retires" contract.
            cmd.src0 = a.host_ptr;
            a.fence = enqueue(d, eng, chn, cmd);
            a.out_flags |= SG_SUBMIT_DIRECT;
            d.bytes_direct += total;
            return 0;
        }

        // Pageable: pipeline through the staging pool. memcpy of chunk i+1
        // proceeds while the DMA of chunk i is in flight; we only block when
        // wrapping onto a slot whose DMA has not retired yet. The slot cursor
        // persists across calls so that a sequence of small copies pipelines
        // too, instead of all queuing on slot 0.
        d.bytes_staged += total;
        const auto* src = reinterpret_cast<const uint8_t*>(a.host_ptr);
        for (uint64_t off = 0; off < total; off += d.chunk) {
            const uint32_t s = d.next_slot;
            d.next_slot = (s + 1) % d.slots;
            const uint64_t n = std::min<uint64_t>(d.chunk, total - off);
            if (int rc = wait_fence(d, d.slot_fence[s], &d.staging_waits)) return rc;
            std::memcpy(d.slot(s), src + off, n);
            cmd.src0 = reinterpret_cast<uint64_t>(d.slot(s));
            cmd.dst = dst + off;
            cmd.size = n;
            a.fence = enqueue(d, eng, chn, cmd);
            d.slot_fence[s] = {eng, chn, a.fence};
        }
        return 0;
    }

    case SG_OP_COPY_D2H: {
        if (a.host_ptr == 0 || !d.vram.owns_range(cmd.src0, cmd.size)) return -EFAULT;
        const uint64_t src = cmd.src0, total = cmd.size;
        d.bytes_d2h += total;

        if (pinned(d, a.host_ptr, total)) {
            cmd.dst = a.host_ptr;
            a.fence = enqueue(d, eng, chn, cmd);
            a.out_flags |= SG_SUBMIT_DIRECT;
            d.bytes_direct += total;
            return 0;
        }

        // Pageable: keep up to `slots` DMAs ahead of the host copy-out. A
        // slot's previous user may have been a DMA on another engine, so
        // each issue waits on that slot's own fence before reusing it.
        d.bytes_staged += total;
        auto* dst = reinterpret_cast<uint8_t*>(a.host_ptr);
        const uint64_t nchunks = (total + d.chunk - 1) / d.chunk;
        const uint32_t base = d.next_slot;
        auto slot_of = [&](uint64_t i) { return uint32_t((base + i) % d.slots); };
        auto issue = [&](uint64_t i) -> int {
            const uint32_t s = slot_of(i);
            if (int rc = wait_fence(d, d.slot_fence[s], &d.staging_waits)) return rc;
            cmd.src0 = src + i * d.chunk;
            cmd.dst = reinterpret_cast<uint64_t>(d.slot(s));
            cmd.size = std::min<uint64_t>(d.chunk, total - i * d.chunk);
            d.slot_fence[s] = {eng, chn, enqueue(d, eng, chn, cmd)};
            return 0;
        };
        for (uint64_t i = 0; i < std::min<uint64_t>(d.slots, nchunks); ++i)
            if (int rc = issue(i)) return rc;
        for (uint64_t i = 0; i < nchunks; ++i) {
            if (int rc = wait_fence(d, d.slot_fence[slot_of(i)])) return rc;
            const uint64_t n = std::min<uint64_t>(d.chunk, total - i * d.chunk);
            std::memcpy(dst + i * d.chunk, d.slot(slot_of(i)), n);
            if (i + d.slots < nchunks)
                if (int rc = issue(i + d.slots)) return rc;
        }
        d.next_slot = slot_of(nchunks);
        a.fence = d.put[eng][chn]; // everything issued here has retired
        return 0;
    }

    case SG_OP_VADD_F32: {
        const uint64_t bytes = uint64_t{cmd.arg0} * sizeof(float);
        if (!d.vram.owns_range(cmd.dst, bytes) || !d.vram.owns_range(cmd.src0, bytes) ||
            !d.vram.owns_range(cmd.src1, bytes))
            return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;
    }

    case SG_OP_GEMM_F32: {
        const uint64_t m = cmd.arg0, n = cmd.arg1, k = cmd.arg2;
        if (!d.vram.owns_range(cmd.src0, m * k * sizeof(float)) ||
            !d.vram.owns_range(cmd.src1, k * n * sizeof(float)) ||
            !d.vram.owns_range(cmd.dst, m * n * sizeof(float)))
            return -EFAULT;
        a.fence = enqueue(d, eng, chn, cmd);
        return 0;
    }

    default:
        return -EINVAL;
    }
}

} // namespace
} // namespace softgpu::driver

using namespace softgpu::driver;

extern "C" int sg_drv_open(void) {
    std::lock_guard<std::mutex> g(g_open_lock);
    if (!g_drv) {
        g_drv = std::make_unique<Driver>();
        g_drv->dev.power_on(device_cpu_from_env());
    }
    ++g_refs;
    return kFd;
}

extern "C" int sg_drv_close(int fd) {
    std::lock_guard<std::mutex> g(g_open_lock);
    if (fd != kFd || g_refs == 0) return -EBADF;
    if (--g_refs == 0) {
        {
            std::lock_guard<std::mutex> dl(g_drv->lock);
            drain(*g_drv); // drain before pulling the plug
        }
        reclaim(*g_drv);
        g_drv->dev.power_off();
        g_drv.reset();
    }
    return 0;
}

extern "C" int sg_drv_ioctl(int fd, unsigned int req, void* arg) {
    if (fd != kFd) return -EBADF;
    Driver* dp;
    {
        std::lock_guard<std::mutex> g(g_open_lock);
        if (!g_drv) return -EBADF;
        dp = g_drv.get();
    }
    Driver& d = *dp;

    // WAIT only reads fence registers, so it does not hold the driver lock
    // while waiting: one thread synchronizing must not block submitters.
    // SG_WAIT_ALL takes the lock just long enough to snapshot the PUTs.
    if (req == SG_IOC_WAIT) {
        if (!arg) return -EINVAL;
        auto* w = static_cast<sg_wait_args*>(arg);
        if (w->engine == SG_WAIT_ALL) {
            PutSnapshot snap;
            {
                std::lock_guard<std::mutex> g(d.lock);
                snapshot_puts(d, snap);
            }
            return wait_snapshot(d, snap);
        }
        if (w->engine >= d.num_engines || w->channel >= d.num_channels) return -EINVAL;
        return wait_fence(d, {w->engine, w->channel, w->fence});
    }

    std::lock_guard<std::mutex> g(d.lock);
    switch (req) {
    case SG_IOC_QUERY: {
        if (!arg) return -EINVAL;
        auto* q = static_cast<sg_query_args*>(arg);
        q->abi_version = SG_ABI_VERSION;
        q->staging_slots = d.slots;
        q->vram_size = SG_VRAM_SIZE;
        q->staging_chunk = d.chunk;
        q->num_engines = d.num_engines;
        q->num_channels = d.num_channels;
        return 0;
    }
    case SG_IOC_ALLOC: {
        if (!arg) return -EINVAL;
        auto* a = static_cast<sg_alloc_args*>(arg);
        if (a->size == 0) return -EINVAL;
        reclaim(d);
        auto addr = d.vram.alloc(a->size);
        if (!addr) {
            // Maybe everything we need is sitting in the deferred list.
            if (int rc = drain(d)) return rc;
            reclaim(d);
            addr = d.vram.alloc(a->size);
            if (!addr) return -ENOMEM;
        }
        a->addr = *addr;
        return 0;
    }
    case SG_IOC_FREE: {
        if (!arg) return -EINVAL;
        reclaim(d);
        const uint64_t addr = static_cast<sg_free_args*>(arg)->addr;
        uint64_t size = 0;
        if (!d.vram.detach(addr, &size)) return -EINVAL;
        // Nothing may be handed this memory until every command issued so
        // far, on any engine, has retired; recycle it then rather than
        // draining now.
        Pending p{Pending::kVram, addr, size, {}};
        snapshot_puts(d, p.fence);
        if (all_retired(d, p.fence)) d.vram.release(addr, size);
        else d.pending.push_back(p);
        return 0;
    }
    case SG_IOC_PIN: {
        if (!arg) return -EINVAL;
        auto* p = static_cast<sg_pin_args*>(arg);
        if (p->addr == 0 || p->size == 0 || p->addr + p->size < p->addr) return -EINVAL;
        reclaim(d);
        // Reject overlap with any existing pin.
        auto next = d.pins.lower_bound(p->addr);
        if (next != d.pins.end() && next->first < p->addr + p->size) return -EEXIST;
        if (next != d.pins.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second > p->addr) return -EEXIST;
        }
        d.pins.emplace(p->addr, p->size);
        return 0;
    }
    case SG_IOC_UNPIN: {
        if (!arg) return -EINVAL;
        auto* p = static_cast<sg_pin_args*>(arg);
        auto it = d.pins.find(p->addr);
        if (it == d.pins.end()) return -EINVAL;
        // Stop treating the range as pinned immediately (new copies go via
        // staging); tell the caller when every engine is done with it.
        Pending pend{Pending::kPin, it->first, it->second, {}};
        snapshot_puts(d, pend.fence);
        d.pending.push_back(pend);
        d.pins.erase(it);
        return 0;
    }
    case SG_IOC_SUBMIT:
        if (!arg) return -EINVAL;
        return do_submit(d, *static_cast<sg_submit_args*>(arg));
    case SG_IOC_STATS: {
        if (!arg) return -EINVAL;
        auto* s = static_cast<sg_stats_args*>(arg);
        std::memset(s, 0, sizeof *s);
        s->num_engines = d.num_engines;
        for (uint32_t e = 0; e < d.num_engines; ++e) {
            auto& st = d.dev.stats(e);
            s->busy_cycles[e] = st.busy_cycles.load(std::memory_order_relaxed);
            s->wait_cycles[e] = st.wait_cycles.load(std::memory_order_relaxed);
            s->idle_cycles[e] = st.idle_cycles.load(std::memory_order_relaxed);
            s->cmds_executed[e] = st.cmds_executed.load(std::memory_order_relaxed);
            s->batches[e] = st.batches.load(std::memory_order_relaxed);
        }
        s->submits = d.submits;
        s->waits = d.waits.load(std::memory_order_relaxed);
        s->stalls = d.stalls;
        s->staging_waits = d.staging_waits;
        s->bytes_h2d = d.bytes_h2d;
        s->bytes_d2h = d.bytes_d2h;
        s->bytes_direct = d.bytes_direct;
        s->bytes_staged = d.bytes_staged;
        return 0;
    }
    case SG_IOC_RESET_STATS:
        d.dev.reset_stats();
        d.submits = d.stalls = d.staging_waits = 0;
        d.waits.store(0, std::memory_order_relaxed);
        d.bytes_h2d = d.bytes_d2h = d.bytes_direct = d.bytes_staged = 0;
        return 0;
    default:
        return -ENOTTY;
    }
}
