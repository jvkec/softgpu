#ifndef SOFTGPU_SG_RUNTIME_H
#define SOFTGPU_SG_RUNTIME_H
/*
 * softgpu runtime API — the user-facing library, shaped like the CUDA runtime.
 *
 * C ABI so it can be consumed from any language and so the boundary between
 * "application" and "driver stack" is explicit.
 */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t sgDevPtr; /* opaque device address (VRAM offset) */

/*
 * Streams order work. Only the default stream (NULL) exists yet; the type is
 * declared now so the *Async signatures do not change when real streams land.
 */
typedef struct sgStream* sgStream_t;

typedef enum sgError {
    SG_OK = 0,
    SG_ERR_NOT_INITIALIZED,
    SG_ERR_ALREADY_INITIALIZED,
    SG_ERR_INVALID_VALUE,
    SG_ERR_OUT_OF_MEMORY,
    SG_ERR_INVALID_ADDRESS,
    SG_ERR_DEVICE,
    SG_ERR_UNKNOWN
} sgError_t;

typedef struct sgStats {
    uint64_t device_busy_cycles;
    uint64_t device_idle_cycles;
    uint64_t device_cmds_executed;
    uint64_t device_batches;   /* times the device woke up and found work   */
    uint64_t driver_submits;
    uint64_t driver_waits;     /* times the host blocked waiting on the device */
    uint64_t driver_stalls;    /* times submission blocked on a full queue     */
    uint64_t staging_waits;    /* times the host blocked for a staging slot    */
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
    uint64_t bytes_direct;     /* copied straight to/from pinned host memory   */
    uint64_t bytes_staged;     /* bounced through the staging pool             */
} sgStats_t;

/* Lifecycle. Not thread-safe with respect to each other. */
sgError_t sgInit(void);
sgError_t sgShutdown(void);

/* Device memory. sgFree is safe while work referencing the memory is in
 * flight: the memory is recycled only after that work retires. */
sgError_t sgMalloc(sgDevPtr* out, size_t bytes);
sgError_t sgFree(sgDevPtr ptr);

/*
 * Pinned host memory. Copies to/from pinned memory are DMA'd directly (no
 * staging copy) and may be asynchronous. sgMallocHost allocates and pins;
 * sgHostRegister pins memory you already own (page-aligned ranges recommended).
 */
sgError_t sgMallocHost(void** out, size_t bytes);
sgError_t sgFreeHost(void* ptr);
sgError_t sgHostRegister(void* ptr, size_t bytes);
sgError_t sgHostUnregister(void* ptr);

/*
 * Copies. The synchronous forms return when the host buffer may be reused
 * (H2D) or the data has arrived (D2H). The *Async forms return as soon as
 * the copy is queued when the host buffer is pinned — do not touch it until
 * sgDeviceSynchronize() — and behave like the synchronous forms for pageable
 * memory. `stream` must be NULL for now.
 */
sgError_t sgMemset(sgDevPtr dst, int value, size_t bytes);
sgError_t sgMemcpyH2D(sgDevPtr dst, const void* src, size_t bytes);
sgError_t sgMemcpyD2H(void* dst, sgDevPtr src, size_t bytes);
sgError_t sgMemcpyD2D(sgDevPtr dst, sgDevPtr src, size_t bytes);
sgError_t sgMemcpyH2DAsync(sgDevPtr dst, const void* src, size_t bytes, sgStream_t stream);
sgError_t sgMemcpyD2HAsync(void* dst, sgDevPtr src, size_t bytes, sgStream_t stream);

/* Compute. All operands are f32 in device memory. */
sgError_t sgVaddF32(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t n);
sgError_t sgGemmF32(sgDevPtr c, sgDevPtr a, sgDevPtr b, uint32_t m, uint32_t n, uint32_t k);

/*
 * Wait for all submitted work to finish.
 * BASELINE: every call above is synchronous, so this is a no-op. It exists
 * so applications written today keep working once submission goes async.
 */
sgError_t sgDeviceSynchronize(void);

/* Telemetry. */
sgError_t sgGetStats(sgStats_t* out);
sgError_t sgResetStats(void);
const char* sgErrorString(sgError_t err);

#ifdef __cplusplus
}
#endif
#endif /* SOFTGPU_SG_RUNTIME_H */
