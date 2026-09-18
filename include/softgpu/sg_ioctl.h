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

#define SG_ABI_VERSION   4u
#define SG_VRAM_SIZE     (256ull << 20) /* 256 MiB of modeled device memory      */
#define SG_STAGING_SLOTS 8u             /* default pageable-copy staging pool ...    */
#define SG_STAGING_CHUNK (256ull << 10) /* ... 8 x 256 KiB, from the sweep in ADR 002 */
#define SG_ALLOC_ALIGN   256u           /* VRAM allocation granularity            */

/*
 * Engines and channels. Engine 0 is the compute engine; engines
 * 1..num_engines-1 are copy engines (DMA). Every engine serves several
 * channels — independent command rings, each with its own fence counter and
 * executed in order — and round-robins between them, skipping a channel
 * whose head is a SG_OP_WAIT_FENCE that is not yet satisfied. That is what
 * lets two streams' work interleave on one engine instead of one stream's
 * semaphore wait blocking everyone behind it in a shared ring.
 */
#define SG_MAX_ENGINES    4u
#define SG_MAX_CHANNELS   8u
#define SG_ENGINE_COMPUTE 0u
#define SG_COPY_ENGINES   1u            /* default number of copy engines (SG_COPY_ENGINES env)   */
#define SG_CHANNELS       8u            /* default channels per engine (SG_CHANNELS env)          */
#define SG_WAIT_ALL       0xffffffffu   /* sg_wait_args.engine: wait for everything submitted so far */

enum sg_opcode {
    SG_OP_NOP        = 0,
    SG_OP_FILL       = 1, /* memset(VRAM[dst], value & 0xff, size)                 */
    SG_OP_COPY_H2D   = 2, /* VRAM[dst]  <- host[src0], size bytes                  */
    SG_OP_COPY_D2H   = 3, /* host[dst]  <- VRAM[src0], size bytes                  */
    SG_OP_COPY_D2D   = 4, /* VRAM[dst]  <- VRAM[src0], size bytes                  */
    SG_OP_VADD_F32   = 5, /* f32 dst[i] = src0[i] + src1[i], for i < arg0          */
    SG_OP_GEMM_F32   = 6, /* f32 row-major dst[m x n] = src0[m x k] * src1[k x n]  */
    SG_OP_WAIT_FENCE = 7, /* block this channel until (engine arg0, channel arg1)'s fence >= src1 */
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
    uint32_t arg0;     /* VADD: element count.  GEMM: m.  WAIT: engine      */
    uint32_t arg1;     /* GEMM: n.  WAIT: channel                            */
    uint32_t arg2;     /* GEMM: k                                            */
    uint32_t value;    /* FILL: byte pattern                                 */
    uint64_t reserved; /* must be 0                                          */
};

struct sg_query_args {
    uint32_t abi_version;
    uint32_t staging_slots;
    uint64_t vram_size;
    uint64_t staging_chunk;
    uint32_t num_engines;  /* 1 compute + copy engines */
    uint32_t num_channels; /* rings per engine         */
};

struct sg_alloc_args {
    uint64_t size; /* in  */
    uint64_t addr; /* out: VRAM offset, SG_ALLOC_ALIGN aligned */
};

struct sg_free_args {
    uint64_t addr;
};

#define SG_SUBMIT_DIRECT 1u /* out_flags: DMA went straight to/from the user buffer */

struct sg_submit_args {
    struct sg_cmd cmd;
    uint64_t host_ptr;  /* in:  user buffer for COPY_H2D/COPY_D2H, 0 otherwise  */
    uint64_t fence;     /* out: fence value on (engine, channel) that retires with this */
    uint32_t out_flags; /* out: SG_SUBMIT_*                                     */
    uint32_t engine;    /* in:  engine to submit to; opcode must match its class */
    uint32_t channel;   /* in:  channel (ring) on that engine                   */
    uint32_t reserved2;
};

/*
 * Pin a host range so the device may DMA to/from it directly. Ranges may not
 * overlap; UNPIN takes the exact addr given to PIN. Unpinning a range that an
 * in-flight command still references is deferred until that command retires.
 */
struct sg_pin_args {
    uint64_t addr;
    uint64_t size;
};

/*
 * Block until (engine, channel) has retired every command up to `fence`, or,
 * with engine == SG_WAIT_ALL, until everything submitted before the call on
 * every channel has retired (channel and fence are ignored).
 */
struct sg_wait_args {
    uint32_t engine;
    uint32_t channel;
    uint64_t fence;
};

struct sg_stats_args {
    /* per engine, indexed 0..num_engines-1 */
    uint32_t num_engines;
    uint32_t reserved;
    uint64_t busy_cycles[SG_MAX_ENGINES];   /* executing commands                   */
    uint64_t wait_cycles[SG_MAX_ENGINES];   /* blocked in SG_OP_WAIT_FENCE          */
    uint64_t idle_cycles[SG_MAX_ENGINES];   /* no commands                          */
    uint64_t cmds_executed[SG_MAX_ENGINES];
    uint64_t batches[SG_MAX_ENGINES];       /* idle->busy transitions               */
    /* driver-side */
    uint64_t submits;       /* driver: SG_IOC_SUBMIT calls                      */
    uint64_t waits;         /* driver: times the host blocked on a fence        */
    uint64_t stalls;        /* driver: times submission blocked on a full ring  */
    uint64_t staging_waits; /* driver: times the host blocked for a staging slot */
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
    uint64_t bytes_direct;  /* of the above, DMA'd to/from pinned memory        */
    uint64_t bytes_staged;  /* of the above, bounced through the staging pool   */
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
    SG_IOC_RESET_STATS = 0x5306, /* no argument    */
    SG_IOC_WAIT        = 0x5307, /* sg_wait_args   */
    SG_IOC_PIN         = 0x5308, /* sg_pin_args    */
    SG_IOC_UNPIN       = 0x5309  /* sg_pin_args (size ignored) */
};

#ifdef __cplusplus
} /* extern "C" */
static_assert(sizeof(sg_cmd) == 64, "sg_cmd must be exactly one cache line");
#endif

#endif /* SOFTGPU_SG_IOCTL_H */
