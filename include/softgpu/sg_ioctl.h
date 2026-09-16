#ifndef SOFTGPU_SG_IOCTL_H
#define SOFTGPU_SG_IOCTL_H
/*
 * softgpu driver ABI.
 *
 * This is the contract between the user-mode runtime (src/runtime) and the
 * kernel-mode driver (src/driver). Today the driver is a userspace stand-in;
 * stage 8 replaces its implementation with a Linux char device that speaks
 * this exact ABI, so this header must stay plain C: fixed-width types,
 * explicit layout, no C++.
 *
 * Every stage of the project changes *implementations* behind this ABI,
 * not the ABI itself, unless a design decision explicitly calls for it.
 */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SG_ABI_VERSION   1u
#define SG_VRAM_SIZE     (256ull << 20) /* 256 MiB of modeled device memory      */
#define SG_STAGING_SIZE  (4ull << 20)   /* BASELINE: one 4 MiB host bounce buffer */
#define SG_ALLOC_ALIGN   256u           /* VRAM allocation granularity            */

enum sg_opcode {
    SG_OP_NOP      = 0,
    SG_OP_FILL     = 1, /* memset(VRAM[dst], value & 0xff, size)                 */
    SG_OP_COPY_H2D = 2, /* VRAM[dst]  <- host[src0], size bytes                  */
    SG_OP_COPY_D2H = 3, /* host[dst]  <- VRAM[src0], size bytes                  */
    SG_OP_COPY_D2D = 4, /* VRAM[dst]  <- VRAM[src0], size bytes                  */
    SG_OP_VADD_F32 = 5, /* f32 dst[i] = src0[i] + src1[i], for i < arg0          */
    SG_OP_GEMM_F32 = 6, /* f32 row-major dst[m x n] = src0[m x k] * src1[k x n]  */
    SG_OP_COUNT
};

/*
 * One command exactly as the device consumes it. Sized to one 64-byte cache
 * line on purpose: a command is written by one core and read by another, and
 * we do not want it straddling lines.
 */
struct sg_cmd {
    uint32_t opcode;   /* enum sg_opcode                                     */
    uint32_t flags;    /* reserved, must be 0                                */
    uint64_t dst;      /* VRAM offset, or host address for COPY_D2H          */
    uint64_t src0;     /* VRAM offset, or host address for COPY_H2D          */
    uint64_t src1;     /* VRAM offset (VADD/GEMM second operand)             */
    uint64_t size;     /* byte count for FILL / COPY_*                       */
    uint32_t arg0;     /* VADD: element count.  GEMM: m                      */
    uint32_t arg1;     /* GEMM: n                                            */
    uint32_t arg2;     /* GEMM: k                                            */
    uint32_t value;    /* FILL: byte pattern                                 */
    uint64_t reserved; /* must be 0                                          */
};

struct sg_query_args {
    uint32_t abi_version;
    uint32_t reserved;
    uint64_t vram_size;
    uint64_t staging_size;
};

struct sg_alloc_args {
    uint64_t size; /* in  */
    uint64_t addr; /* out: VRAM offset, SG_ALLOC_ALIGN aligned */
};

struct sg_free_args {
    uint64_t addr;
};

struct sg_submit_args {
    struct sg_cmd cmd;
    uint64_t host_ptr; /* user buffer for COPY_H2D/COPY_D2H, 0 otherwise */
};

struct sg_stats_args {
    uint64_t busy_cycles;   /* device: cycles spent executing commands  */
    uint64_t idle_cycles;   /* device: cycles spent waiting for work    */
    uint64_t cmds_executed; /* device: commands retired                 */
    uint64_t submits;       /* driver: SG_IOC_SUBMIT calls              */
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
};

/*
 * Request codes. Plain integers for now; the kernel module will wrap them in
 * _IOWR() with the same ordinals so the runtime does not change.
 */
enum sg_ioc {
    SG_IOC_QUERY       = 0x5301, /* sg_query_args  */
    SG_IOC_ALLOC       = 0x5302, /* sg_alloc_args  */
    SG_IOC_FREE        = 0x5303, /* sg_free_args   */
    SG_IOC_SUBMIT      = 0x5304, /* sg_submit_args */
    SG_IOC_STATS       = 0x5305, /* sg_stats_args  */
    SG_IOC_RESET_STATS = 0x5306  /* no argument    */
};

#ifdef __cplusplus
} /* extern "C" */
static_assert(sizeof(sg_cmd) == 64, "sg_cmd must be exactly one cache line");
#endif

#endif /* SOFTGPU_SG_IOCTL_H */
