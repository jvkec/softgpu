# 003 — Copy engine, channels, and streams

## Context

Stage 3a ([002](002-pinned-memory-and-pipelined-staging.md)) removed the
host's copy from the data path and moved the bottleneck onto the device:

* one in-order engine executes H2D a → H2D b → VADD → D2H c, so pinned
  `vadd n=1M` stopped at 581 µs — ~16 MiB of device traffic at ~30 GB/s
  before the kernel even runs — and pageable tied pinned at large n;
* every copy path converged on ~30 GB/s above L2 size: one core's DRAM
  stream, and the same core that has to run the kernel.

Real GPUs answer this with **separate copy engines** fed by their own
command queues, **semaphores** to order work across engines, and
**streams** in the runtime so applications can say which work is
independent. This is the first stage that changes the device model since
stage 0.

## Problem

Copies and compute cannot overlap because one engine does both, and even
with two engines nothing in the API lets an application express that chunk
*i+1*'s copy is independent of chunk *i*'s kernel.

## Options considered

1. **Second engine, one ring per engine, streams in the runtime.** The
   obvious first cut, and the one we built first — see *What we measured
   first* below. It serializes at a shared ring.
2. **Engines × channels with a runlist** (this record). Each engine serves
   several rings; a channel whose head is an unsatisfied semaphore is
   skipped, not waited on. This is the GPU host-scheduler model (NVIDIA
   channels/runlists, AMD HQDs).
3. **Host-side scheduling**: the runtime holds back commands until their
   dependencies retire and submits them later. Keeps the device trivial but
   makes the host the scheduler — latency, CPU, and the exact thing GPUs
   moved into hardware.

## Decision

**Device** (`src/device`): a compute engine plus `SG_COPY_ENGINES` copy
engines (default 1), each serving `SG_CHANNELS` channels (default 8). A
channel is what a single ring was in stage 1 — `ring_base`/`ring_mask`,
`put`, `get`, sticky error — and `get` is its fence. New command
`SG_OP_WAIT_FENCE(engine, channel, value)`: a semaphore acquire. The engine
loop is a **runlist**: each round visits every channel with published work
and runs it until it empties or its head is a WAIT that is not yet
satisfied, then moves on. Time is accounted per engine as *busy*
(executing), *wait* (work pending but every head blocked) or *idle*.

**Deadlock freedom.** The driver accepts a WAIT only if its target value is
≤ that channel's PUT at submission time — the command being waited for was
enqueued strictly before the WAIT. By induction on enqueue order the "waits
on" graph is a DAG, so some channel head is always runnable until every
ring is drained. The runlist also means a WAIT can never block a command on
a *different* channel of the same engine.

**ABI** (`include/softgpu/sg_ioctl.h`, version 3 → 4): `SUBMIT` names
`(engine, channel)`; `WAIT` takes `(engine, channel, fence)` or
`SG_WAIT_ALL` ("everything submitted before this call"); `QUERY` reports
`num_engines`/`num_channels`; stats are per engine with the new
`wait_cycles`. Fences inside the driver — staging slot reuse, deferred
frees and unpins — become `(engine, channel, value)` or a snapshot of every
channel's PUT.

**Runtime** (`include/softgpu/sg_runtime.h`, `src/runtime`): `sgStream_t`
becomes real, `sgEvent_t` is added. A stream owns a channel index and a
copy engine (`id mod N`), and keeps its commands in order *across* engines
by submitting `WAIT_FENCE` on the immediate predecessor whenever the engine
changes — sufficient by transitivity, because rings are in order and the
predecessor already waited on its own predecessor. `sgEventRecord` is a NOP
in the stream (so it flushes pending waits) whose fence the event captures;
`sgStreamWaitEvent` queues that fence as an extra WAIT ahead of the
stream's next command. The default stream (`NULL`) is an ordinary stream
that does **not** synchronize with others — CUDA's per-thread-default-stream
semantics; the legacy blocking default stream is a footgun we would have to
pay for in every benchmark. `sgDeviceSynchronize` and `sgFreeHost` use
`SG_WAIT_ALL`. New API: `sgStreamCreate/Destroy/Synchronize`,
`sgEventCreate/Destroy/Record/Synchronize`, `sgStreamWaitEvent`,
`sgMemsetAsync`, `sgMemcpyD2DAsync`, `sgVaddF32Async`, `sgGemmF32Async`.

**Not changed, on purpose:** ring depth, staging pool, big driver lock,
spinning waits, VRAM allocator.

## What we measured first, and why the design changed

The first implementation had exactly one ring per engine, shared by all
streams. It passed every test — ordering was correct — and the new
`pipeline` benchmark showed **no overlap at all**: `concurrency` (engines
busy on average) of 0.99–1.00 and speedups of 1.00–1.09 in every row
(`results/streams-v1-shared-ring.json`). The reason is head-of-line
blocking. With two streams the copy engine's ring holds, in submission
order, `a0 b0 WAIT(vadd0) c0 a1 b1 …`: the engine reaches `WAIT(vadd0)` and
must stall there, so stream 1's `a1 b1` — which could have run during
`vadd0` — sit behind it. Σ busy = wall exactly because one engine is always
waiting on the other. Semaphores in a shared in-order queue serialize the
queue; independence needs separate queues and a scheduler that skips a
blocked one. That is what channels are for, and `SG_CHANNELS=1` reproduces
the first result with the final code (`results/streams-ch1.json`).

## Hypotheses (written before measuring the channel design)

* **H1** `pipeline R=1` (copy-heavy): S=1 `concurrency ≈ 1.0`, no speedup
  — an in-order stream cannot overlap with itself even across engines;
  S=2 ≈ 1.6–1.8× and `concurrency ≈ 1.7`; S=4 adds little (two engines, one
  runnable stream each is enough).
* **H2** `pipeline R=4` (compute-heavy): copies hide behind compute at S≥2
  → speedup ≈ 1 + copy/compute ≈ 1.2×, compute-engine utilization → ~0.9+.
* **H3** Two copy engines help only the copy-heavy case (H2D and D2H of
  different chunks overlap): R=1, S≥2 gains ~1.3× more; R=4 unchanged. Both
  CEs stream from the same DRAM, so < 2×.
* **H4** Existing benchmarks unchanged within noise (`submit`, `batch`,
  `mt`, `gemm`, pinned `memcpy` ±5%); default-stream pinned `vadd` pays one
  extra WAIT per engine change (~0.1–0.3 µs at small n).
* **H5** In `R=4, S=1` the copy engine's *wait* time dominates its non-idle
  time (≥ 90%): the signature of a serialized stream.

## Results

Stage 3a (`results/pinned.json`) vs 3b (`results/streams.json`, one copy
engine, eight channels), same-cluster mode, best of 15 passes across 5
processes per benchmark, canary-gated before and after each configuration.

### Copy–compute overlap (`pipeline`: 16 chunks × 1 M floats, H2D a, H2D b, R× VADD, D2H c)

| | µs/chunk | speedup vs S=1 | engines busy (avg) | compute util | copy-engine util | CE time blocked on semaphore |
|---|---:|---:|---:|---:|---:|---:|
| R=1, S=1 | 645 | 1.00 | 0.99 | 0.32 | 0.67 | 33% |
| R=1, S=2 | **449** | **1.47** | 1.49 | 0.51 | **0.98** | 2% |
| R=1, S=4 | 448 | 1.48 | 1.51 | 0.52 | 1.00 | 0% |
| R=4, S=1 | 1,405 | 1.00 | 1.00 | 0.70 | 0.30 | 70% |
| R=4, S=2 | **1,197** | **1.40** | 1.45 | **0.97** | 0.48 | 52% |
| R=4, S=4 | 1,103 | 1.27 | 1.42 | 0.97 | 0.45 | 55% |

**H1 confirmed in shape, short in magnitude.** One stream cannot overlap
with itself (exactly one engine busy at a time), two streams give 1.47×
with 1.5 engines busy — and then the copy engine is *saturated* (98–100%
utilization), so a third and fourth stream add nothing. The bottleneck has
moved onto the copy engine, which is what a second CE is for (below).

**H2 confirmed and exceeded.** With compute dominating, two streams take the
compute engine to 97% utilization and the copies hide almost entirely:
1.40× against a predicted ~1.2×. The copy engine then spends half its
non-idle time blocked on semaphores, which is fine — it has nothing better
to do.

**H5 confirmed.** In `R=4, S=1` the copy engine is blocked 70% of its
non-idle time: the signature of a serialized stream.

### Single ring vs channels (`results/streams-v1-shared-ring.json`, `streams-ch1.json`, `streams.json`)

| | shared ring (first design) | `SG_CHANNELS=1` (final code) | 8 channels |
|---|---:|---:|---:|
| R=1, S=2 speedup / engines busy | 1.01 / 0.99 | 1.01 / 0.99 | **1.47 / 1.49** |
| R=4, S=2 speedup / engines busy | 1.09 / 1.00 | 1.04 / 1.00 | **1.40 / 1.45** |
| R=1, S=2 CE blocked on semaphore | 34% | 33% | 2% |

The final code with one channel reproduces the first design's numbers
exactly: the overlap comes from the channels and the runlist, not from
having two engines.

### Two copy engines (`results/streams-ce2.json`, `SG_COPY_ENGINES=2`)

| | 1 CE | 2 CE | engines busy (2 CE) |
|---|---:|---:|---:|
| R=1, S=2 | 449 µs | 440 | 1.91 |
| R=1, S=4 | 448 | **392** (1.69×) | 2.48 |
| R=4, S=1 | 1,405 | 1,746 (+24%) | 1.00 |
| R=4, S=2 | 1,197 | 1,278 (+7%) | 1.41 |
| R=4, S=4 | 1,103 | 1,307 (+19%) | 1.40 |

**H3 confirmed, with a cost the hypothesis missed.** A second CE pays only
when copies dominate *and* there are at least four streams to feed both
engines (1.48× → 1.69×, two and a half engines busy). Everywhere else it
makes things 7–24% *worse*: a third always-on engine thread competes with
the compute engine's memory-bound kernel for DRAM bandwidth and for the
machine's six vCPUs. On a real GPU copy engines are separate silicon with
their own paths to memory; on a shared-memory model they are not free.
**Default: one copy engine**, `SG_COPY_ENGINES` stays as the knob.

### What got slower (`submit`, `mt`, `vadd`, `alloc`)

| | 3a | 3b | why |
|---|---:|---:|---|
| `submit fill_64B` p50 (async) | 83 ns | 125 ns | stream lock + channel bookkeeping in the runtime |
| `submit fill_64B_sync` p50 (round trip) | 208 ns | 250 ns | runlist scans 8 channels per round; the canary moved 208/583 → 250/666 in both placement modes |
| `mt` 8 threads, default stream | 3.81 M/s | 3.35 M/s | every thread also serializes on the default stream's lock |
| `mt` 8 threads, one stream each (`results/streams-mt.json`) | — | **4.09 M/s** | only the driver lock is shared — the reason per-thread streams exist |
| `vadd n=1024` pinned+async | 1.32 µs | 3.13 µs | two cross-engine semaphore handoffs per iteration, each a cross-core cache-line transfer (~0.2–0.6 µs) plus a WAIT command |
| `vadd n=1M` pinned+async | 581 µs | 512 µs | large copies and the kernel now overlap slightly even on one stream (the kernel's D2H can start while the next iteration's H2D queues) |
| `alloc` in flight | 667 ns | 833 ns | `SG_WAIT_ALL` snapshots and checks every channel |
| `memcpy` pinned ≥ 4 MiB, `gemm` | — | within 0–5% | unchanged paths |

**H4 partly refuted.** The overhead is not noise; it is the price of
dispatching to specialised engines: a tiny dependent chain that hops
copy → compute → copy pays two cross-core handoffs it did not pay when one
engine did everything. Real copy engines carry the same latency tax (µs-scale
launch on GPUs), and the standard advice — batch small transfers, keep
tiny dependent work on one engine — follows from the same numbers. A
heuristic that runs small copies on the compute engine when the stream is
already there would remove it; it is noted as future work rather than done
here, to keep the engine classes clean while the model is young.

## Consequences

* Applications can now express independence, and the device rewards it:
  1.4–1.5× on the chunked pipeline with two streams, up to 1.7× with two
  copy engines and four streams. One stream stays strictly in order across
  engines; the cost of that ordering is visible as `wait_cycles`.
* The engine/channel/runlist model is the real GPU host-scheduler shape,
  and it is the substrate for stage 6 (scheduling policy, time slicing,
  priorities across channels) and for stage 4 (per-channel submission can
  drop the big lock, since channels are already independent producers).
* Fixed per-command costs went up ~40 ns and tiny cross-engine chains got
  slower; the same thread-placement bimodality that governs host round
  trips now governs engine-to-engine semaphores. Stage 2's interrupt/hybrid
  wait and stage 4's lock removal are where these fixed costs get attacked.
* The first implementation was wrong in a way every test missed and one
  benchmark caught instantly: correctness tests check ordering, not
  concurrency. `concurrency` and `ce_wait_frac` are now part of every row.
