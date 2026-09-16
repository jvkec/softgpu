// STAGE 1 driver: command ring with fences.
//
// What changed from the baseline:
//   * commands go into a ring in system memory instead of a one-slot
//     mailbox; the doorbell is the ring's PUT pointer;
//   * submission is asynchronous — SG_IOC_SUBMIT returns a fence, and the
//     host only blocks when it must: on SG_IOC_WAIT, before reading a D2H
//     copy out of staging, or before reusing the staging buffer;
//   * a full ring applies backpressure (the submitter spins for a slot).
//
// What deliberately did *not* change, so the ring's effect is isolated:
//   * the big driver lock;
//   * one 4 MiB staging buffer for pageable copies;
//   * spinning on the fence register instead of interrupts.
//
// Validation is unchanged: every address is checked against live
// allocations before the device sees it, and the device checks again.
// Because commands now execute after SUBMIT returns, a device-side failure
// is reported as a sticky error on the next WAIT/SUBMIT, as CUDA does.

#include "driver/sg_driver.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
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

// SG_RING_DEPTH=<power of two in [2, 65536]> overrides the ring depth. It is
// a tuning knob for the decision record, not a user-facing setting.
uint32_t ring_depth_from_env() {
    const char* e = std::getenv("SG_RING_DEPTH");
    if (!e) return kDefaultRingDepth;
    unsigned long v = std::strtoul(e, nullptr, 10);
    if (v < 2 || v > 65536 || (v & (v - 1)) != 0) return kDefaultRingDepth;
    return static_cast<uint32_t>(v);
}

struct FreeDeleter { void operator()(void* p) const { std::free(p); } };

struct Driver {
    std::mutex lock; // still the big driver lock; stage 4 breaks it up
    device::Device dev{SG_VRAM_SIZE};
    VramAllocator vram{SG_VRAM_SIZE, SG_ALLOC_ALIGN};
    std::vector<uint8_t> staging = std::vector<uint8_t>(SG_STAGING_SIZE);

    std::unique_ptr<sg_cmd, FreeDeleter> ring;
    uint32_t depth = 0;
    uint64_t put = 0;           // driver-side mirror of the PUT register
    uint64_t staging_fence = 0; // staging may be overwritten once this retires

    uint64_t submits = 0;
    uint64_t waits = 0;
    uint64_t stalls = 0;
    uint64_t bytes_h2d = 0;
    uint64_t bytes_d2h = 0;

    Driver() {
        depth = ring_depth_from_env();
        void* mem = nullptr;
        if (posix_memalign(&mem, 64, size_t{depth} * sizeof(sg_cmd)) != 0) std::abort();
        std::memset(mem, 0, size_t{depth} * sizeof(sg_cmd));
        ring.reset(static_cast<sg_cmd*>(mem));
        // Program the ring into the device before powering it on.
        dev.regs().ring_base = reinterpret_cast<uint64_t>(ring.get());
        dev.regs().ring_mask = depth - 1;
    }
};

std::mutex g_open_lock;
std::unique_ptr<Driver> g_drv;
int g_refs = 0;

// SG_DEVICE_CPU=<n> pins the engine thread; -1 (default) leaves it to the OS.
int device_cpu_from_env() {
    const char* e = std::getenv("SG_DEVICE_CPU");
    return e ? std::atoi(e) : -1;
}

inline int sticky(Driver& d) {
    return d.dev.regs().sticky_error.load(std::memory_order_relaxed);
}

// Block until the device has retired everything up to `fence`.
int wait_fence(Driver& d, uint64_t fence) {
    auto& r = d.dev.regs();
    if (r.get.load(std::memory_order_acquire) < fence) {
        ++d.waits;
        while (r.get.load(std::memory_order_acquire) < fence) cpu_relax();
    }
    return sticky(d) ? -EIO : 0;
}

// Append one command and ring the doorbell. Returns its fence. Caller holds
// d.lock, which is what makes the driver the ring's single producer.
uint64_t enqueue(Driver& d, const sg_cmd& cmd) {
    auto& r = d.dev.regs();
    // Backpressure: the slot we are about to write must have been retired.
    // Acquire on `get` orders the device's read of that slot before our write.
    if (d.put - r.get.load(std::memory_order_acquire) >= d.depth) {
        ++d.stalls;
        while (d.put - r.get.load(std::memory_order_acquire) >= d.depth) cpu_relax();
    }
    d.ring.get()[d.put & (d.depth - 1)] = cmd;
    ++d.put;
    // Release orders the slot write before the doorbell.
    r.put.store(d.put, std::memory_order_release);
    return d.put;
}

int do_submit(Driver& d, sg_submit_args& a) {
    sg_cmd cmd = a.cmd;
    if (cmd.opcode >= SG_OP_COUNT || cmd.flags != 0 || cmd.reserved != 0) return -EINVAL;
    if (sticky(d)) return -EIO;
    ++d.submits;

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
        const auto* src = reinterpret_cast<const uint8_t*>(a.host_ptr);
        const uint64_t dst = cmd.dst, total = cmd.size;
        d.bytes_h2d += total;
        // Pageable copy: memcpy into staging, DMA out of it. The DMA is now
        // asynchronous, but with one staging buffer the *next* chunk must
        // wait for it — so multi-chunk copies are still serialized. That is
        // the measured motivation for stage 3.
        for (uint64_t off = 0; off < total; off += SG_STAGING_SIZE) {
            const uint64_t n = std::min<uint64_t>(SG_STAGING_SIZE, total - off);
            if (int rc = wait_fence(d, d.staging_fence)) return rc;
            std::memcpy(d.staging.data(), src + off, n);
            cmd.src0 = reinterpret_cast<uint64_t>(d.staging.data());
            cmd.dst = dst + off;
            cmd.size = n;
            d.staging_fence = a.fence = enqueue(d, cmd);
        }
        return 0;
    }

    case SG_OP_COPY_D2H: {
        if (a.host_ptr == 0 || !d.vram.owns_range(cmd.src0, cmd.size)) return -EFAULT;
        auto* dst = reinterpret_cast<uint8_t*>(a.host_ptr);
        const uint64_t src = cmd.src0, total = cmd.size;
        d.bytes_d2h += total;
        // The device executes in order, so this DMA cannot overwrite staging
        // before an earlier H2D DMA has finished reading it. The host must
        // wait before copying out, which makes D2H fully synchronous.
        for (uint64_t off = 0; off < total; off += SG_STAGING_SIZE) {
            const uint64_t n = std::min<uint64_t>(SG_STAGING_SIZE, total - off);
            cmd.src0 = src + off;
            cmd.dst = reinterpret_cast<uint64_t>(d.staging.data());
            cmd.size = n;
            d.staging_fence = a.fence = enqueue(d, cmd);
            if (int rc = wait_fence(d, a.fence)) return rc;
            std::memcpy(dst + off, d.staging.data(), n);
        }
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
        // Drain outstanding work before pulling the plug.
        wait_fence(*g_drv, g_drv->put);
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
        q->reserved = 0;
        q->vram_size = SG_VRAM_SIZE;
        q->staging_size = SG_STAGING_SIZE;
        return 0;
    }
    case SG_IOC_ALLOC: {
        if (!arg) return -EINVAL;
        auto* a = static_cast<sg_alloc_args*>(arg);
        if (a->size == 0) return -EINVAL;
        auto addr = d.vram.alloc(a->size);
        if (!addr) return -ENOMEM;
        a->addr = *addr;
        return 0;
    }
    case SG_IOC_FREE: {
        if (!arg) return -EINVAL;
        // Freeing memory that in-flight commands still reference would be a
        // use-after-free on the device. Draining first is the simple, safe
        // answer; deferred frees tied to fences come with stage 3.
        if (int rc = wait_fence(d, d.put)) return rc;
        return d.vram.free(static_cast<sg_free_args*>(arg)->addr) ? 0 : -EINVAL;
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
        s->waits = d.waits;
        s->stalls = d.stalls;
        s->bytes_h2d = d.bytes_h2d;
        s->bytes_d2h = d.bytes_d2h;
        return 0;
    }
    case SG_IOC_RESET_STATS:
        d.dev.reset_stats();
        d.submits = d.waits = d.stalls = d.bytes_h2d = d.bytes_d2h = 0;
        return 0;
    default:
        return -ENOTTY;
    }
}
