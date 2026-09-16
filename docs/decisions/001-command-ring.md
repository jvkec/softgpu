# 001 — Command ring with fences

## Context

The baseline ([000](000-baseline-results.md)) submits one command at a time
through a single-slot mailbox and spins until it retires. Its numbers said:

* every API call pays a full round trip (~250 ns), so `vadd n=1024` is four
  round trips and 91% copy/submit overhead;
* the device is idle 80%+ of the time in every benchmark except GEMM;
* eight submitters get *less* throughput than one (−56%) and 400× the tail
  latency, because they serialize on one lock around a blocking round trip.

## Problem

There is nowhere for more than one command to live. Every improvement we
want next — asynchronous submission, fences, multiple queues, DMA that
overlaps compute, interrupts instead of polling — needs a queue between the
driver and the device. The mailbox has to go first.

## Options considered

1. **Deeper mailbox (N slots, still synchronous).** Trivial, but every
   command still waits for completion; nothing becomes asynchronous.
2. **Ring buffer in system memory with PUT/GET pointers.** The driver
   appends commands and advances `put`; the device consumes and advances
   `get`. `get` doubles as a fence: command *n* has retired once
   `get > n`. This is the GPFIFO scheme every NVIDIA GPU since NV4 uses,
   and what Vulkan/CUDA queues sit on.
3. **Linked command lists (per-submission buffers chained by pointers).**
   More flexible (variable-length submissions, indirect execution), and how
   AMD/Intel front-ends work, but strictly more complexity than we need
   before we have measured what a ring gets us.

Option 2. It isolates one change — *a queue exists* — and gives us the fence
primitive every later stage builds on.

## Decision

**Device** (`src/device`): the register file loses its mailbox and gains
`ring_base`/`ring_mask` (programmed by the driver before power-on), a
`put` register (driver → device) and a `get` register (device → driver).
The engine drains every published command before going idle and retires
each one individually with a release store to `get`, so a host waiting on
an early fence is not held up by the rest of the batch. Errors become a
sticky `-errno` register, reported on the next submit or wait — CUDA's
sticky-error model.

**Driver** (`src/driver`): submission appends to the ring under the (still
global) lock, rings the doorbell with a release store, and returns
immediately with a fence. The host now blocks only where it must:

| operation | blocks on |
|---|---|
| FILL, D2D, VADD, GEMM | nothing — asynchronous |
| H2D (pageable, via staging) | the previous DMA out of the staging buffer, before overwriting it |
| D2H | its own fence, before copying out of staging |
| `sgDeviceSynchronize` | the highest fence any thread submitted |
| `sgFree` | full drain — in-flight commands may still reference the memory |

A full ring applies backpressure: the submitter spins for a free slot and
the stall is counted. `SG_RING_DEPTH` overrides the depth for experiments.

**ABI** (`include/softgpu/sg_ioctl.h`, version 1 → 2): `sg_submit_args`
gains an out-parameter `fence`; new `SG_IOC_WAIT`; `sg_stats_args` gains
`batches`, `waits`, `stalls`. The public runtime API is unchanged;
`sgDeviceSynchronize()` now does something.

**Not changed, on purpose:** the big lock, the single 4 MiB staging buffer,
and spinning as the wait mechanism. Each is a later stage, and leaving them
alone lets the ring's effect be read off cleanly.

## Measurement notes

Two things about the environment shaped how these numbers were collected;
both are worth knowing about for any cross-core benchmark on Apple silicon
under virtualization.

**Placement is bimodal.** The M1 Max has two 4-core performance clusters
and one efficiency cluster. When the submitter and the device thread land on
the same cluster, a doorbell round trip costs ~208 ns; across clusters,
~583 ns — 2.8× — because a cache-line handoff has to cross the fabric. The
guest cannot see or control host placement, and it changes over time (the
same guest-pinned vCPU pair reads 208 ns one minute and 584 ns the next).
Placement is also sticky per process and *workload-dependent*: after a
64 MiB streaming copy or a long GEMM the host tends to migrate the two
threads apart and leave them there, so a benchmark that runs after `memcpy`
in the same process inherits slow placement. Medians over a bimodal
distribution are meaningless, so `sgbench --repeat` keeps, per row, every
metric from the fastest pass; `scripts/vm.sh` runs every benchmark in its
own process, five times, and merges the same way. Runs are gated by a
canary (one round trip) before and after; a run that flips mid-way is
discarded. Results files record host load, run count and date in `meta`.

**Host load leaks in.** Spotlight indexing the new build trees tripled every
latency while host CPU was 70% idle. `build/` is now marked non-indexable and
the canary catches it.

## Results

Baseline vs ring (1024 slots), same-cluster mode, best of 15 passes across
5 processes per benchmark.
Full data: [`results/baseline.json`](../../results/baseline.json),
[`results/ring.json`](../../results/ring.json).

### Submission cost (`submit`)

| variant | baseline p50 | ring p50 | baseline ops/s | ring ops/s | host CPU ns/op |
|---|---:|---:|---:|---:|---:|
| fill_64B (async) | 250 | **84** | 3.69 M | **8.20 M** | 271 → 122 |
| fill_4KiB (async) | 333 | **83** | 2.64 M | **8.85 M** | 379 → 113 |
| fill_64B_sync (round trip) | 250 | 208 | 3.46 M | 4.16 M | 289 → 240 |

An asynchronous submit costs 84 ns — twice the API floor of 42 ns — and the
p99 is 125 ns. The round trip itself shed ~40 ns because the protocol got
simpler (no ticket handshake, one release store each way).

### Amortization (`batch`: B fills then one sync)

| B | baseline ns/op | ring ns/op | waits/op | avg cmds per device wake | device util |
|---:|---:|---:|---:|---:|---:|
| 1 | 223 | 166 | 1.000 | 1.00 | 0.17 → 0.36 |
| 4 | 212 | 89 | 0.250 | 2.00 | 0.17 → 0.56 |
| 16 | 208 | 77 | 0.062 | 2.74 | 0.18 → 0.74 |
| 64 | 209 | 74 | 0.016 | 2.80 | 0.18 → 0.78 |
| 256 | 208 | 73 | 0.004 | 2.81 | 0.18 → 0.80 |
| 1024 | 208 | 72 | 0.001 | 2.87 | 0.18 → 0.80 |

Cost per command more than halves as soon as the host stops waiting
(B ≥ 4), then flattens at ~72 ns. That floor is the *submitter's*
per-command work — lock, validation, slot write, doorbell — not the
device's: with ~2.8 commands per wake the device is keeping up comfortably
and still sits idle 20% of the time. **The bottleneck has moved from the
round trip to the submission path itself.**

### Concurrent submitters (`mt`, 4 KiB memset per op)

| threads | baseline ops/s | ring ops/s | baseline p99 | ring p99 | host CPU ns/op |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.03 M | **7.65 M** | 292 ns | 125 ns | 320 → 124 |
| 2 | 2.42 M | **6.38 M** | 1.13 µs | 0.75 µs | 482 → 308 |
| 4 | 1.33 M | **4.28 M** | 71.9 µs | 1.46 µs | 1,261 → 483 |
| 8 | 1.05 M | **3.83 M** | 223 µs | 14.0 µs | 2,154 → 805 |

Throughput 2.5–3.6×, tail latency down 94–98% at 4+ threads, CPU per op
down ~60%. But look at the shape: the ring still *loses* half its
throughput from 1 to 8 threads. The lock convoy is gone (threads no longer
hold the lock across a round trip) but the lock itself still serializes
every submit, and at 8 threads on 6 vCPUs the scheduler joins in. That is
stage 4.

### Real workload (`vadd`: H2D, H2D, VADD, D2H)

| n | baseline µs/iter | ring µs/iter | waits/iter | kernel fraction |
|---:|---:|---:|---:|---:|
| 1,024 | 3.23 | **2.58** | 4 → 2 | 0.12 → 0.04 |
| 65,536 | 44.2 | 42.1 | 4 → 2 | 0.22 → 0.002 |
| 1,048,576 | 779 | 795 | 4 → 2 | 0.21 → 0.000 |
| 16,777,216 | 14,159 | 14,198 | 49 → 47 | 0.20 → 0.000 |

Waits per iteration halved exactly as predicted (VADD and the first H2D no
longer block), the kernel's *visible* cost vanished because it overlaps
with the host's D2H submission, and the smallest case gained 20%. Above
that, wall time did not move: the copies are still serialized through one
staging buffer and still dominate. Stage 3.

### Unchanged, as expected

`memcpy` bandwidth within +0–8% at ≥ 1 MiB (the staging path is the same;
small copies gain a little from the cheaper doorbell); `gemm` within ±1%
(compute-bound work never cared about submission cost).

## Ring depth

`SG_RING_DEPTH` swept over 16 … 4096 slots (64 B each), `batch`/`mt`/`vadd`
only. Files: `results/ring-d{16,64,256,4096}.json` and `results/ring.json`
(1024).

| depth | batch B≥64 ns/op | avg cmds/wake | mt 8-thread ops/s | mt 8-thread p99 | stalls/op (8 thr) |
|---:|---:|---:|---:|---:|---:|
| 16 | 94 | 1.02 | 2.72 M | 66 µs | 0.005 |
| 64 | 94 | 1.02 | 2.88 M | 59 µs | 0.004 |
| 256 | 94 | 1.03 | 3.27 M | 33 µs | 0.002 |
| **1024** | **74** | **2.8** | **3.83 M** | **14 µs** | 0.000 |
| 4096 | 66 | 6.4 | 3.62 M | 25 µs | 0.000 |

Up to 256 slots the device retires each command before the next arrives
(one command per wake) and contended submitters occasionally hit a full
ring. At 1024 the device finally gets to fall behind and batch (2.8
commands per wake): 21% cheaper per command, 17% more aggregate throughput
and half the tail latency versus 256. 4096 batches deeper still and shaves
another 10% per command, but its multithreaded throughput and p99 are no
better, it costs 4× the memory (256 KiB vs 64 KiB), and it quadruples the
worst-case drain behind a `sgDeviceSynchronize`.

**Default depth: 1024.** Round trips (`fill_64B_sync`, 208 ns) and async
submit cost (84 ns) are identical at every depth, so latency-sensitive
callers lose nothing.

## Anomalies found and resolved

Two effects survived the first measurement clean-up and were recorded
rather than smoothed over; both turned out to be the same thing.

1. **`vadd n=1024` depended on what ran before it in the same process:**
   ~2.5 µs after `batch`, ~6.6 µs after `memcpy`, `gemm`, or nothing. The
   round-trip canary measured *after* each benchmark tracked it exactly
   (209 ns after `batch`, 540–625 ns after everything else). Workloads that
   stream memory or run long get the two threads migrated onto different
   host clusters, and they stay there. It was placement, not caching — the
   staging-buffer hypothesis this record briefly held was wrong.
2. **`batch B=1` read 2× slower than the equivalent `submit fill_64B_sync`**
   in some runs, for the same reason: it inherited the previous benchmark's
   placement.

Running every benchmark in its own process removed both. With that in
place, `vadd n=1024` is 3.23 → 2.58 µs and `batch B=1` is 166 ns against a
208 ns synchronous round trip.

## Consequences

* Everything above the driver is now asynchronous; a device error surfaces
  at the next wait, not the offending call. Applications that relied on
  baseline synchrony implicitly must call `sgDeviceSynchronize()` (the
  tests already did).
* `sgFree` drains the queue. Cheap today; wrong once there are multiple
  contexts. Deferred frees keyed on fences come with stage 3.
* The new floor is ~72 ns per command of submitter work under a global lock,
  and the device keeps up with it while idle 20% of the time. The next measurable wins are, in order:
  overlap the copies (stage 3, `vadd` and `memcpy` are 2× on the table),
  stop spinning (stage 2, `cpu_ns_per_op ≈ wall/op` everywhere), and remove
  the lock (stage 4, flat `mt` scaling).
