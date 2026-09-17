// STAGE 3a driver: pinned host memory, pipelined staging, deferred frees.
//
// What changed from stage 1:
//   * host memory can be pinned (SG_IOC_PIN); copies to/from a pinned range
//     are one DMA command straight to the user buffer — no host copy;
//   * pageable copies bounce through a pool of staging slots instead of one
//     buffer, so the host memcpy of chunk i+1 overlaps the DMA of chunk i;
//   * freeing VRAM or unpinning memory that in-flight commands may still
//     reference is deferred until the fence at free time retires, instead of
//     draining the whole queue.
//
// Still deliberately unchanged: one ring, the big driver lock, spinning as
// the wait mechanism, and the device model itself.

#include "driver/sg_driver.h"

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
uint32_t ring_depth_from_env() { return uint32_t(env_pow2("SG_RING_DEPTH", kDefaultRingDepth, 2, 65536)); }
uint32_t staging_slots_from_env() { return uint32_t(env_pow2("SG_STAGING_SLOTS", SG_STAGING_SLOTS, 1, 64)); }
uint64_t staging_chunk_from_env() { return env_pow2("SG_STAGING_CHUNK", SG_STAGING_CHUNK, 4096, 64ull << 20); }
// SG_DEVICE_CPU=<n> pins the engine thread; -1 (default) leaves it to the OS.
int device_cpu_from_env() {
    const char* e = std::getenv("SG_DEVICE_CPU");
    return e ? std::atoi(e) : -1;
}

struct FreeDeleter { void operator()(void* p) const { std::free(p); } };

// Something the user has released that in-flight commands may still touch.
struct Pending {
    enum Kind { kVram, kPin } kind;
    uint64_t addr;
    uint64_t size;
    uint64_t fence; // safe to recycle once the device has retired this
};

struct Driver {
    std::mutex lock; // still the big driver lock; stage 4 breaks it up
    device::Device dev{SG_VRAM_SIZE};
    VramAllocator vram{SG_VRAM_SIZE, SG_ALLOC_ALIGN};

    std::unique_ptr<sg_cmd, FreeDeleter> ring;
    uint32_t depth = 0;
    uint64_t put = 0; // driver-side mirror of the PUT register

    // Staging pool for pageable copies: `slots` chunks of `chunk` bytes, each
    // reusable once the DMA that last used it (slot_fence) has retired.
    std::vector<uint8_t> staging;
    std::vector<uint64_t> slot_fence;
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
        depth = ring_depth_from_env();
        void* mem = nullptr;
        if (posix_memalign(&mem, 64, size_t{depth} * sizeof(sg_cmd)) != 0) std::abort();
        std::memset(mem, 0, size_t{depth} * sizeof(sg_cmd));
        ring.reset(static_cast<sg_cmd*>(mem));
        dev.regs().ring_base = reinterpret_cast<uint64_t>(ring.get());
        dev.regs().ring_mask = depth - 1;

        slots = staging_slots_from_env();
        chunk = staging_chunk_from_env();
        staging.assign(size_t{slots} * chunk, 0);
        slot_fence.assign(slots, 0);
    }

    uint8_t* slot(uint32_t i) { return staging.data() + size_t{i} * chunk; }
};

std::mutex g_open_lock;
std::unique_ptr<Driver> g_drv;
int g_refs = 0;

inline int sticky(Driver& d) {
    return d.dev.regs().sticky_error.load(std::memory_order_relaxed);
}
inline uint64_t retired(Driver& d) { return d.dev.regs().get.load(std::memory_order_acquire); }

// Block until the device has retired everything up to `fence`.
int wait_fence(Driver& d, uint64_t fence, uint64_t* counter = nullptr) {
    if (retired(d) < fence) {
        if (counter) ++*counter;
        else d.waits.fetch_add(1, std::memory_order_relaxed);
        while (retired(d) < fence) cpu_relax();
    }
    return sticky(d) ? -EIO : 0;
}

// Append one command and ring the doorbell. Returns its fence. Caller holds
// d.lock, which is what makes the driver the ring's single producer.
uint64_t enqueue(Driver& d, const sg_cmd& cmd) {
    auto& r = d.dev.regs();
    if (d.put - retired(d) >= d.depth) {
        ++d.stalls;
        while (d.put - retired(d) >= d.depth) cpu_relax();
    }
    d.ring.get()[d.put & (d.depth - 1)] = cmd;
    ++d.put;
    r.put.store(d.put, std::memory_order_release); // release: slot before doorbell
    return d.put;
}

// Recycle everything the device has finished with.
void reclaim(Driver& d) {
    if (d.pending.empty()) return;
    const uint64_t done = retired(d);
    auto keep = d.pending.begin();
    for (auto& p : d.pending) {
        if (p.fence <= done) {
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

int do_submit(Driver& d, sg_submit_args& a) {
    sg_cmd cmd = a.cmd;
    if (cmd.opcode >= SG_OP_COUNT || cmd.flags != 0 || cmd.reserved != 0) return -EINVAL;
    if (sticky(d)) return -EIO;
    ++d.submits;
    a.out_flags = 0;

    switch (cmd.opcode) {
    case SG_OP_NOP:
        a.fence = enqueue(d, cmd);
        return 0;

    case SG_OP_FILL:
        if (!d.vram.owns_range(cmd.dst, cmd.size)) return -EFAULT;
        a.fence = enqueue(d, cmd);
        return 0;

    case SG_OP_COPY_D2D:
        if (!d.vram.owns_range(cmd.dst, cmd.size) || !d.vram.owns_range(cmd.src0, cmd.size))
            return -EFAULT;
        a.fence = enqueue(d, cmd);
        return 0;

    case SG_OP_COPY_H2D: {
        if (a.host_ptr == 0 || !d.vram.owns_range(cmd.dst, cmd.size)) return -EFAULT;
        const uint64_t dst = cmd.dst, total = cmd.size;
        d.bytes_h2d += total;

        if (pinned(d, a.host_ptr, total)) {
            // Direct DMA from the user's buffer. Asynchronous: the caller owns
            // the "don't touch it until the fence retires" contract.
            cmd.src0 = a.host_ptr;
            a.fence = enqueue(d, cmd);
            a.out_flags |= SG_SUBMIT_DIRECT;
            d.bytes_direct += total;
            return 0;
        }

        // Pageable: pipeline through the staging pool. memcpy of chunk i+1
        // proceeds while the DMA of chunk i is in flight; we only block when
        // wrapping onto a slot whose DMA has not retired yet.
        // The slot cursor persists across calls so that a sequence of small
        // copies (each one chunk) also pipelines instead of all queuing on
        // slot 0 — the first measurement of this stage showed exactly that.
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
            d.slot_fence[s] = a.fence = enqueue(d, cmd);
        }
        return 0;
    }

    case SG_OP_COPY_D2H: {
        if (a.host_ptr == 0 || !d.vram.owns_range(cmd.src0, cmd.size)) return -EFAULT;
        const uint64_t src = cmd.src0, total = cmd.size;
        d.bytes_d2h += total;

        if (pinned(d, a.host_ptr, total)) {
            cmd.dst = a.host_ptr;
            a.fence = enqueue(d, cmd);
            a.out_flags |= SG_SUBMIT_DIRECT;
            d.bytes_direct += total;
            return 0;
        }

        // Pageable: keep up to `slots` DMAs ahead of the host copy-out. The
        // device executes in order, so a D2H DMA into a slot can never
        // overtake an earlier H2D DMA reading from it.
        d.bytes_staged += total;
        auto* dst = reinterpret_cast<uint8_t*>(a.host_ptr);
        const uint64_t nchunks = (total + d.chunk - 1) / d.chunk;
        const uint32_t base = d.next_slot;
        auto slot_of = [&](uint64_t i) { return uint32_t((base + i) % d.slots); };
        auto issue = [&](uint64_t i) {
            cmd.src0 = src + i * d.chunk;
            cmd.dst = reinterpret_cast<uint64_t>(d.slot(slot_of(i)));
            cmd.size = std::min<uint64_t>(d.chunk, total - i * d.chunk);
            d.slot_fence[slot_of(i)] = enqueue(d, cmd);
        };
        for (uint64_t i = 0; i < std::min<uint64_t>(d.slots, nchunks); ++i) issue(i);
        for (uint64_t i = 0; i < nchunks; ++i) {
            if (int rc = wait_fence(d, d.slot_fence[slot_of(i)])) return rc;
            const uint64_t n = std::min<uint64_t>(d.chunk, total - i * d.chunk);
            std::memcpy(dst + i * d.chunk, d.slot(slot_of(i)), n);
            if (i + d.slots < nchunks) issue(i + d.slots);
        }
        d.next_slot = slot_of(nchunks);
        a.fence = d.put; // everything issued here has retired
        return 0;
    }

    case SG_OP_VADD_F32: {
        const uint64_t bytes = uint64_t{cmd.arg0} * sizeof(float);
        if (!d.vram.owns_range(cmd.dst, bytes) || !d.vram.owns_range(cmd.src0, bytes) ||
            !d.vram.owns_range(cmd.src1, bytes))
            return -EFAULT;
        a.fence = enqueue(d, cmd);
        return 0;
    }

    case SG_OP_GEMM_F32: {
        const uint64_t m = cmd.arg0, n = cmd.arg1, k = cmd.arg2;
        if (!d.vram.owns_range(cmd.src0, m * k * sizeof(float)) ||
            !d.vram.owns_range(cmd.src1, k * n * sizeof(float)) ||
            !d.vram.owns_range(cmd.dst, m * n * sizeof(float)))
            return -EFAULT;
        a.fence = enqueue(d, cmd);
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
        wait_fence(*g_drv, g_drv->put); // drain before pulling the plug
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

    // WAIT only reads the fence register, so it does not take the driver
    // lock: one thread synchronizing must not block others from submitting.
    if (req == SG_IOC_WAIT) {
        if (!arg) return -EINVAL;
        return wait_fence(d, static_cast<sg_wait_args*>(arg)->fence);
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
            if (int rc = wait_fence(d, d.put)) return rc;
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
        // far has retired; recycle it then rather than draining now.
        if (retired(d) >= d.put) d.vram.release(addr, size);
        else d.pending.push_back({Pending::kVram, addr, size, d.put});
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
        // staging); tell the caller when the device is done with it.
        d.pending.push_back({Pending::kPin, it->first, it->second, d.put});
        d.pins.erase(it);
        p->fence = d.put;
        return 0;
    }
    case SG_IOC_SUBMIT:
        if (!arg) return -EINVAL;
        return do_submit(d, *static_cast<sg_submit_args*>(arg));
    case SG_IOC_STATS: {
        if (!arg) return -EINVAL;
        auto* s = static_cast<sg_stats_args*>(arg);
        auto& r = d.dev.regs();
        s->busy_cycles = r.busy_cycles.load(std::memory_order_relaxed);
        s->idle_cycles = r.idle_cycles.load(std::memory_order_relaxed);
        s->cmds_executed = r.cmds_executed.load(std::memory_order_relaxed);
        s->batches = r.batches.load(std::memory_order_relaxed);
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
