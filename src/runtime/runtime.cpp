// User-mode runtime: implements sg_runtime.h on top of the driver's
// ioctl-shaped interface. It owns no device state — only a descriptor — and
// never touches VRAM or registers. Its jobs are argument marshalling and
// mapping -errno to sgError_t.

#include "softgpu/sg_runtime.h"

#include <atomic>
#include <cerrno>
#include <cstring>

#include "driver/sg_driver.h"
#include "softgpu/sg_ioctl.h"

namespace {

int g_fd = -1;
// Highest fence handed out so far. Submitters race to publish theirs after
// the ioctl returns, so keep the max rather than the last writer's value;
// sgDeviceSynchronize() then means "everything any thread has submitted".
std::atomic<uint64_t> g_last_fence{0};

void publish_fence(uint64_t f) {
    uint64_t cur = g_last_fence.load(std::memory_order_relaxed);
    while (cur < f && !g_last_fence.compare_exchange_weak(cur, f, std::memory_order_relaxed)) {}
}

sgError_t from_errno(int rc) {
    switch (rc) {
    case 0:       return SG_OK;
    case -EINVAL: return SG_ERR_INVALID_VALUE;
    case -ENOMEM: return SG_ERR_OUT_OF_MEMORY;
    case -EFAULT: return SG_ERR_INVALID_ADDRESS;
    case -EBADF:  return SG_ERR_NOT_INITIALIZED;
    case -EIO:    return SG_ERR_DEVICE;
    default:      return SG_ERR_UNKNOWN;
    }
}

sgError_t submit(sg_cmd cmd, uint64_t host_ptr = 0) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    sg_submit_args a{};
    a.cmd = cmd;
    a.host_ptr = host_ptr;
    int rc = sg_drv_ioctl(g_fd, SG_IOC_SUBMIT, &a);
    if (rc == 0) publish_fence(a.fence);
    return from_errno(rc);
}

sg_cmd make_cmd(sg_opcode op) {
    sg_cmd c{};
    c.opcode = op;
    return c;
}

} // namespace

extern "C" {

sgError_t sgInit(void) {
    if (g_fd >= 0) return SG_ERR_ALREADY_INITIALIZED;
    int fd = sg_drv_open();
    if (fd < 0) return from_errno(fd);
    sg_query_args q{};
    if (int rc = sg_drv_ioctl(fd, SG_IOC_QUERY, &q); rc != 0 || q.abi_version != SG_ABI_VERSION) {
        sg_drv_close(fd);
        return rc ? from_errno(rc) : SG_ERR_DEVICE;
    }
    g_fd = fd;
    g_last_fence.store(0, std::memory_order_relaxed);
    return SG_OK;
}

sgError_t sgShutdown(void) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    int rc = sg_drv_close(g_fd);
    g_fd = -1;
    return from_errno(rc);
}

sgError_t sgMalloc(sgDevPtr* out, size_t bytes) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!out || bytes == 0) return SG_ERR_INVALID_VALUE;
    sg_alloc_args a{};
    a.size = bytes;
    int rc = sg_drv_ioctl(g_fd, SG_IOC_ALLOC, &a);
    if (rc == 0) *out = a.addr;
    return from_errno(rc);
}

sgError_t sgFree(sgDevPtr ptr) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    sg_free_args a{ptr};
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_FREE, &a));
}

sgError_t sgMemset(sgDevPtr dst, int value, size_t bytes) {
    sg_cmd c = make_cmd(SG_OP_FILL);
    c.dst = dst;
    c.size = bytes;
    c.value = static_cast<uint32_t>(value) & 0xffu;
    return submit(c);
}

sgError_t sgMemcpyH2D(sgDevPtr dst, const void* src, size_t bytes) {
    if (!src) return SG_ERR_INVALID_VALUE;
    sg_cmd c = make_cmd(SG_OP_COPY_H2D);
    c.dst = dst;
    c.size = bytes;
    return submit(c, reinterpret_cast<uint64_t>(src));
}

sgError_t sgMemcpyD2H(void* dst, sgDevPtr src, size_t bytes) {
    if (!dst) return SG_ERR_INVALID_VALUE;
    sg_cmd c = make_cmd(SG_OP_COPY_D2H);
    c.src0 = src;
    c.size = bytes;
    return submit(c, reinterpret_cast<uint64_t>(dst));
}

sgError_t sgMemcpyD2D(sgDevPtr dst, sgDevPtr src, size_t bytes) {
    sg_cmd c = make_cmd(SG_OP_COPY_D2D);
    c.dst = dst;
    c.src0 = src;
    c.size = bytes;
    return submit(c);
}

sgError_t sgVaddF32(sgDevPtr c_, sgDevPtr a, sgDevPtr b, uint32_t n) {
    sg_cmd c = make_cmd(SG_OP_VADD_F32);
    c.dst = c_;
    c.src0 = a;
    c.src1 = b;
    c.arg0 = n;
    return submit(c);
}

sgError_t sgGemmF32(sgDevPtr c_, sgDevPtr a, sgDevPtr b, uint32_t m, uint32_t n, uint32_t k) {
    sg_cmd c = make_cmd(SG_OP_GEMM_F32);
    c.dst = c_;
    c.src0 = a;
    c.src1 = b;
    c.arg0 = m;
    c.arg1 = n;
    c.arg2 = k;
    return submit(c);
}

sgError_t sgDeviceSynchronize(void) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    sg_wait_args w{g_last_fence.load(std::memory_order_relaxed)};
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_WAIT, &w));
}

sgError_t sgGetStats(sgStats_t* out) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    if (!out) return SG_ERR_INVALID_VALUE;
    sg_stats_args s{};
    int rc = sg_drv_ioctl(g_fd, SG_IOC_STATS, &s);
    if (rc == 0) {
        out->device_busy_cycles = s.busy_cycles;
        out->device_idle_cycles = s.idle_cycles;
        out->device_cmds_executed = s.cmds_executed;
        out->device_batches = s.batches;
        out->driver_submits = s.submits;
        out->driver_waits = s.waits;
        out->driver_stalls = s.stalls;
        out->bytes_h2d = s.bytes_h2d;
        out->bytes_d2h = s.bytes_d2h;
    }
    return from_errno(rc);
}

sgError_t sgResetStats(void) {
    if (g_fd < 0) return SG_ERR_NOT_INITIALIZED;
    return from_errno(sg_drv_ioctl(g_fd, SG_IOC_RESET_STATS, nullptr));
}

const char* sgErrorString(sgError_t err) {
    switch (err) {
    case SG_OK:                      return "no error";
    case SG_ERR_NOT_INITIALIZED:     return "runtime not initialized";
    case SG_ERR_ALREADY_INITIALIZED: return "runtime already initialized";
    case SG_ERR_INVALID_VALUE:       return "invalid value";
    case SG_ERR_OUT_OF_MEMORY:       return "out of device memory";
    case SG_ERR_INVALID_ADDRESS:     return "invalid device address";
    case SG_ERR_DEVICE:              return "device error";
    default:                         return "unknown error";
    }
}

} // extern "C"
