// BASELINE driver. Deliberately naive in ways that real drivers are not, so
// that each later stage has a concrete, measured reason to exist:
//
//   * one global mutex around every operation (allocation and submission);
//   * one command in flight: write mailbox, ring doorbell, spin until done;
//   * pageable host memory: every H2D/D2H copy bounces through a single
//     4 MiB staging buffer, one device command per chunk;
//   * completion by busy-polling the `completed` register — no interrupts.
//
// What it does *not* skimp on: validation. Every address the user hands us
// is checked against live allocations before the device sees it, and the
// device checks again. That layering is the same in the kernel module.

#include "driver/sg_driver.h"

#include <cerrno>
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

struct Driver {
    std::mutex lock; // BASELINE: the big driver lock
    device::Device dev{SG_VRAM_SIZE};
    VramAllocator vram{SG_VRAM_SIZE, SG_ALLOC_ALIGN};
    std::vector<uint8_t> staging = std::vector<uint8_t>(SG_STAGING_SIZE);
    uint32_t ticket = 0;

    // driver-side counters (device-side ones live in its registers)
    uint64_t submits = 0;
    uint64_t bytes_h2d = 0;
    uint64_t bytes_d2h = 0;
};

std::mutex g_open_lock;
std::unique_ptr<Driver> g_drv;
int g_refs = 0;

// Program one command and wait for it to retire. Caller holds d.lock.
int kick(Driver& d, const sg_cmd& cmd) {
    auto& r = d.dev.regs();
    r.mailbox = cmd;
    // Release orders the mailbox write before the doorbell; the device's
    // acquire load on the doorbell picks both up.
    const uint32_t ticket = r.doorbell.fetch_add(1, std::memory_order_release) + 1;
    d.ticket = ticket;
    while (r.completed.load(std::memory_order_acquire) != ticket) cpu_relax();
    return r.last_error.load(std::memory_order_relaxed);
}

int do_submit(Driver& d, sg_submit_args& a) {
    sg_cmd cmd = a.cmd;
    if (cmd.opcode >= SG_OP_COUNT || cmd.flags != 0 || cmd.reserved != 0) return -EINVAL;
    ++d.submits;

    switch (cmd.opcode) {
    case SG_OP_NOP:
        return kick(d, cmd);

    case SG_OP_FILL:
        if (!d.vram.owns_range(cmd.dst, cmd.size)) return -EFAULT;
        return kick(d, cmd);

    case SG_OP_COPY_D2D:
        if (!d.vram.owns_range(cmd.dst, cmd.size) || !d.vram.owns_range(cmd.src0, cmd.size))
            return -EFAULT;
        return kick(d, cmd);

    case SG_OP_COPY_H2D: {
        if (a.host_ptr == 0 || !d.vram.owns_range(cmd.dst, cmd.size)) return -EFAULT;
        const auto* src = reinterpret_cast<const uint8_t*>(a.host_ptr);
        const uint64_t dst = cmd.dst, total = cmd.size;
        d.bytes_h2d += total;
        // BASELINE: pageable copy = memcpy into staging, then DMA from staging,
        // one chunk at a time, fully serialized. Two passes over every byte.
        for (uint64_t off = 0; off < total; off += SG_STAGING_SIZE) {
            const uint64_t n = std::min<uint64_t>(SG_STAGING_SIZE, total - off);
            std::memcpy(d.staging.data(), src + off, n);
            cmd.src0 = reinterpret_cast<uint64_t>(d.staging.data());
            cmd.dst = dst + off;
            cmd.size = n;
            if (int rc = kick(d, cmd)) return rc;
        }
        return 0;
    }

    case SG_OP_COPY_D2H: {
        if (a.host_ptr == 0 || !d.vram.owns_range(cmd.src0, cmd.size)) return -EFAULT;
        auto* dst = reinterpret_cast<uint8_t*>(a.host_ptr);
        const uint64_t src = cmd.src0, total = cmd.size;
        d.bytes_d2h += total;
        for (uint64_t off = 0; off < total; off += SG_STAGING_SIZE) {
            const uint64_t n = std::min<uint64_t>(SG_STAGING_SIZE, total - off);
            cmd.src0 = src + off;
            cmd.dst = reinterpret_cast<uint64_t>(d.staging.data());
            cmd.size = n;
            if (int rc = kick(d, cmd)) return rc;
            std::memcpy(dst + off, d.staging.data(), n);
        }
        return 0;
    }

    case SG_OP_VADD_F32: {
        const uint64_t bytes = uint64_t{cmd.arg0} * sizeof(float);
        if (!d.vram.owns_range(cmd.dst, bytes) || !d.vram.owns_range(cmd.src0, bytes) ||
            !d.vram.owns_range(cmd.src1, bytes))
            return -EFAULT;
        return kick(d, cmd);
    }

    case SG_OP_GEMM_F32: {
        const uint64_t m = cmd.arg0, n = cmd.arg1, k = cmd.arg2;
        if (!d.vram.owns_range(cmd.src0, m * k * sizeof(float)) ||
            !d.vram.owns_range(cmd.src1, k * n * sizeof(float)) ||
            !d.vram.owns_range(cmd.dst, m * n * sizeof(float)))
            return -EFAULT;
        return kick(d, cmd);
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
        g_drv->dev.power_on();
    }
    ++g_refs;
    return kFd;
}

extern "C" int sg_drv_close(int fd) {
    std::lock_guard<std::mutex> g(g_open_lock);
    if (fd != kFd || g_refs == 0) return -EBADF;
    if (--g_refs == 0) {
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
        s->submits = d.submits;
        s->bytes_h2d = d.bytes_h2d;
        s->bytes_d2h = d.bytes_d2h;
        return 0;
    }
    case SG_IOC_RESET_STATS:
        d.dev.reset_stats();
        d.submits = d.bytes_h2d = d.bytes_d2h = 0;
        return 0;
    default:
        return -ENOTTY;
    }
}
