# 002 — Pinned host memory, pipelined staging, deferred frees

## Context

Stage 1 ([001](001-command-ring.md)) moved the bottleneck off the round trip
and onto the data path:

* `memcpy` plateaued at ~25 GB/s with `dev_util ≈ 0.5`: every byte was copied
  twice — host `memcpy` into one 4 MiB staging buffer, then device DMA out of
  it — and the two copies were strictly serialized. Above 4 MiB it dropped to
  ~18 GB/s as chunks serialized further.
* `vadd` was 80–90% copies at every size. The ring removed one wait per
  iteration, but the second H2D still blocked on the first's DMA to free the
  single staging buffer, so wall time barely moved.
* `sgFree` drained the entire queue (`wait_fence(put)`): safe, but a full
  stall for any application that frees while work is in flight.

## Problem

Pageable copies pay for two passes over the data with no overlap between
them, and there is no way for an application to opt out of the staging copy
at all. Separately, resource release is coupled to queue drain.

## Options considered

1. **Bigger staging buffer.** Fewer chunks, same serialization; measured
   `dev_util` of 0.5 says the problem is overlap, not chunk count.
2. **Pinned host memory + pipelined staging pool** (this record). Pinned
   memory (`cudaMallocHost`/`cudaHostRegister`) lets the device DMA straight
   to and from the user's buffer — one pass, asynchronous. For pageable
   memory, a pool of staging slots lets the host copy chunk *i+1* while the
   device DMAs chunk *i*, which is exactly what CUDA's pageable `cudaMemcpy`
   does under the hood.
3. **Make all host memory pinned by default.** Simplest for users, but
   pinning is a real cost on real systems (unswappable pages, IOMMU
   mappings) and the pageable path is the one applications actually hit
   most; both need to exist and be measured.
4. **A second (copy) engine so copies overlap compute.** Necessary, and
   coming — but it changes the device model. This stage deliberately changes
   only the driver and runtime so the data-path effect is attributable.

Option 2, plus fence-deferred frees so `sgFree`/unpin no longer drain.

## Decision

**ABI** (`include/softgpu/sg_ioctl.h`, version 2 → 3):
`SG_IOC_PIN`/`SG_IOC_UNPIN` with `sg_pin_args {addr, size, fence(out)}`;
`sg_submit_args.out_flags` with `SG_SUBMIT_DIRECT`; stats gain
`staging_waits`, `bytes_direct`, `bytes_staged`; `sg_query_args` reports the
staging pool geometry.

**Driver** (`src/driver/driver.cpp`):

* *Pin registry* — a map of non-overlapping host ranges. A copy whose host
  buffer lies entirely inside one range is a single DMA command with the
  user's address, flagged `SG_SUBMIT_DIRECT`, returned asynchronously. The
  device model was already DMA-capable to any host address the driver
  authorises, so it did not change.
* *Staging pool* — `SG_STAGING_SLOTS × SG_STAGING_CHUNK`, each slot with
  its own fence and a slot cursor that persists across calls. H2D: wait for
  the slot's previous DMA only if it has not retired, `memcpy`, enqueue.
  D2H: keep up to *slots* DMAs issued ahead of the host copy-out. The
  device executing in order is what makes reusing a slot for D2H safe while
  an earlier H2D DMA from it may still be queued. The first default was
  4 × 1 MiB (the old footprint); the sweep below changed it to 8 × 256 KiB.
* *Deferred release* — `sgFree` detaches the range from the live set at once
  (validation stops accepting it) but returns it to the free list only when
  the fence captured at free time retires; the pending list is reclaimed on
  every allocation-side ioctl and at close. Unpin works the same way and
  reports the fence so `sgFreeHost` can wait for exactly the DMAs that
  reference the memory before `free()`ing it.

**Runtime** (`include/softgpu/sg_runtime.h`, `src/runtime/runtime.cpp`):
`sgMallocHost`/`sgFreeHost`/`sgHostRegister`/`sgHostUnregister`;
`sgMemcpyH2DAsync`/`sgMemcpyD2HAsync(…, sgStream_t)` with the stream type
declared now (only `NULL` accepted) so stage 3b does not change signatures.
The synchronous copies keep their contract: when the driver reports a direct
DMA they wait on its fence before returning.

**Not changed, on purpose:** the device model, one ring, the big lock,
spinning waits. Every effect below is attributable to the three changes.

**Found along the way:** ThreadSanitizer flagged a data race on the driver's
`waits` counter — `SG_IOC_WAIT` intentionally runs without the driver lock
(a thread synchronizing must not block submitters), and stage 1 incremented
a plain integer there. Stage 1's tests never had two threads waiting at
once; `sgFreeHost` does. The counter is now atomic. A stage-1 latent bug
caught by a stage-3a test, which is the reason the sanitizer builds are part
of every stage's verification.

## Hypotheses (written before measuring)

* **H1** Pinned copies ≥ 1 MiB reach roughly 2× pageable stage 1 — a single
  copy's speed, ~45–50 GB/s — with `dev_util → ~1.0`.
* **H2** Pipelined pageable copies reach ~1.6–1.8× stage 1 at ≥ 4 MiB;
  4 KiB copies are unchanged (round-trip bound, one chunk, nothing to
  overlap).
* **H3** `vadd` with pinned buffers and async copies: waits per iteration
  2 → 1; n = 1 M from ~795 µs to ~400 µs; the kernel's share of time becomes
  visible again because the copies stop dominating.
* **H4** Staging sweep: 4 × 1 MiB ≈ optimal. 64 KiB chunks pay per-command
  overhead (~72 ns each plus the slot wait); 1 × 4 MiB regresses to stage 1;
  more than 4 slots buys little because the host `memcpy` and the device DMA
  run at similar speeds, so two in flight already keep both busy.
* **H5** `sgMalloc`+`sgFree` with a 512³ GEMM (~14 ms) in flight: from a
  ~14 ms drain to sub-microsecond.

## Results

Stage 1 (`results/ring.json`) vs stage 3a with the final default pool
(`results/pinned.json`, 8 × 256 KiB). Same-cluster mode, best of 15 passes
across 5 processes per benchmark; every configuration passed the placement
canary before and after.

### Copies (`memcpy`, GB/s)

| size | stage 1 pageable | 3a pageable | 3a pinned | pageable `dev_util` (1 → 3a) |
|---:|---:|---:|---:|---:|
| 4 KiB H2D / D2H | 6.5 / 6.5 | 14.0 / 8.1 | 16.0 / 16.1 | 0.30 → 0.68 / 0.58 → 0.48 |
| 64 KiB | 23.1 / 24.9 | 36.4 / 25.0 | **66.4 / 69.6** | 0.43 → 0.99 / 0.54 → 0.36 |
| 1 MiB | 26.2 / 26.3 | 43.2 / 36.1 | **65.6 / 56.9** | 0.58 → 0.99 / 0.41 → 0.60 |
| 4 MiB | 24.7 / 24.7 | **44.1 / 40.7** | 59.6 / 57.0 | 0.55 → 1.00 / 0.45 → 0.89 |
| 16 MiB | 19.8 / 19.1 | 29.4 / 30.3 | 32.0 / 33.8 | 0.52 → 0.99 / 0.54 → 0.99 |
| 64 MiB | 18.1 / 18.0 | 27.8 / 28.0 | 32.4 / 31.5 | 0.40 → 0.64 / 0.60 → 1.00 |

**H1 confirmed and exceeded.** Pinned copies between 64 KiB and 4 MiB run
at 57–70 GB/s — 2.3–2.8× stage 1 — because the only copy left is the
device's. Above ~8 MiB every path converges on ~30 GB/s: that is one core
streaming through DRAM once the data no longer fits in the cluster's L2, and
it bounds pinned and pageable alike. The device engine is now the limit.

**H2 confirmed.** Pipelined pageable copies reach 1.5–1.8× stage 1 at
≥ 1 MiB and, for H2D, drive the device to ≥ 99% utilization: the host
`memcpy` is fully hidden behind the DMA. D2H gains less at small sizes
because the host copy-out cannot start until the DMA has landed — inherent
to the direction, not to the design.

Not predicted: **4 KiB H2D more than doubled** (6.5 → 14.0 GB/s) without
any per-copy work changing. Consecutive small copies now land in different
staging slots, so the host no longer waits for copy *i*'s DMA before
staging copy *i+1*. The first measurement of this stage caught the bug —
the slot cursor reset to 0 on every call, and `staging_waits_per_op` read
exactly 1.0 for every small-copy row.

### Staging pool sweep (`results/pinned-s{1,2,8}.json`, `pinned-c{64k,256k,4m}.json`, `pinned-s4-c1m.json`)

| pool | 1 MiB H2D | 4 MiB H2D | 64 MiB H2D | note |
|---|---:|---:|---:|---|
| 1 × 1 MiB | 26.2 | 24.8 | 18.1 | = stage 1: one slot cannot pipeline |
| 2 × 1 MiB | 24.7 | 44.5 | 29.6 | two slots already capture the whole gain |
| 4 × 1 MiB (first default) | 24.4 | 44.7 | 29.4 | 1 MiB copies are a single chunk: no pipeline |
| 8 × 1 MiB | 24.3 | 45.0 | 29.7 | no further gain |
| 4 × 64 KiB | 39.6 | 40.6 | 21.7 | pipelines small copies; ~72 ns/command shows at 64 MiB |
| 4 × 256 KiB | 43.4 | 43.4 | 27.4 | |
| 4 × 4 MiB | 26.0 | 24.9 | 29.6 | nothing under 4 MiB pipelines |
| **8 × 256 KiB (default)** | **43.2** | **44.1** | 27.8 | half the memory of the first default |

**H4 refuted in the details.** Slot count matters only between 1 and 2 —
host `memcpy` and device DMA run at similar speed, so double-buffering is
enough. Chunk size is the real knob: it sets the smallest copy that can be
pipelined, at a per-command cost that only shows for very large copies
(64 KiB chunks lose 25% at 64 MiB; 256 KiB loses ~5%). **Decision: 8 × 256
KiB** — 2 MiB total, pipelines everything from 512 KiB up, and costs ~5% on
copies larger than the pool relative to 1 MiB chunks.

### Workload (`vadd`: H2D, H2D, VADD, D2H per iteration)

| n | stage 1 | 3a pageable | 3a pinned + async | waits/iter (pinned) |
|---:|---:|---:|---:|---:|
| 1,024 | 2.58 µs | 2.29 | **1.32** | 1 |
| 65,536 | 42.1 | 38.5 | **23.5** | 1 |
| 1,048,576 | 795 | 540 | **581** | 1 |
| 16,777,216 | 14,198 | 10,195 | **10,944** | 1 |

**H3 partly confirmed.** Pinned buffers with asynchronous copies halve the
small cases (1.32 µs is one round trip plus four ~80 ns submits) and take
waits per iteration to exactly one. The large cases stop at −27…−32% rather
than the predicted −50%, and pageable now *ties or beats* pinned at ≥ 1 M
elements. Both facts have the same cause: the device is one in-order engine,
so it copies a, copies b, adds, copies c — 16 MiB of traffic per 1 M-element
iteration at ~30 GB/s is ~540 µs before the kernel runs at all. Pinned
memory removed the host's copy; nothing yet overlaps the device's. That is
stage 3b.

### Resource release (`alloc`: one `sgMalloc`+`sgFree` pair)

| | p50 | p99 |
|---|---:|---:|
| nothing in flight | 209 ns | 292 ns |
| 512³ GEMM (13.7 ms) in flight | **667 ns** | 1.8 µs |

**H5 confirmed.** Stage 1's `sgFree` waited for the queue to drain — the
whole 13.7 ms GEMM (`gemm` row, unchanged). Deferred release makes freeing
under load a list append; the memory is recycled when the fence retires.

### Unchanged, as expected

`submit` (84 ns async, 208 ns round trip), `gemm` (10.2 / 191 / 13,676 µs)
and `mt` throughput (−1…−6%) are within noise of stage 1: the changes were
confined to the copy path and the allocator. (`mt` 8-thread p99 doubled to
28 µs; with 8 submitters on 6 vCPUs that metric is scheduler noise and
moves between any two runs.)

## Consequences

* Two copy paths now exist, chosen by the driver per call from the pin
  registry. Applications that care can pin; the pageable path is 1.5–2×
  better than it was without them changing anything. The `SG_SUBMIT_DIRECT`
  contract — the runtime waits for direct DMAs on synchronous calls — is
  what keeps `sgMemcpyH2D` correct on pinned memory; `*Async` hands that
  responsibility to the caller, as CUDA does.
* `sgFree` and `sgHostUnregister` no longer drain. `sgFreeHost` waits only
  for the DMAs that reference the buffer (the fence UNPIN returns). The
  pending list is bounded by how far the device lags, and `sgMalloc`
  falls back to a drain-then-reclaim only when the free list alone cannot
  satisfy a request.
* The staging pool footprint halved (4 → 2 MiB) while getting faster; its
  geometry is a documented knob (`SG_STAGING_SLOTS`, `SG_STAGING_CHUNK`).
* The bottleneck has moved again, this time onto the device: one in-order
  engine serializes copies with compute, and one core caps large copies at
  ~30 GB/s. Stage 3b adds a copy engine with its own ring, cross-ring
  semaphores, and streams/events in the runtime so copies for chunk *i+1*
  overlap compute on chunk *i*.
* Methodology: every benchmark now runs in its own process and every
  configuration is bracketed by a placement canary; two stage-1-era
  artefacts (a latent data race and an order-dependent result) fell out of
  that discipline rather than out of luck.
