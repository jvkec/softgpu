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
Medians over a bimodal distribution are meaningless, so `sgbench --repeat`
keeps, per row, every metric from the fastest pass, and `scripts/vm.sh`
merges several separate processes the same way. Runs are gated by a canary
(one round trip) before and after; a run that flips mid-way is discarded.
Results files record host load and run count in `meta`.

**Host load leaks in.** Spotlight indexing the new build trees tripled every
latency while host CPU was 70% idle. `build/` is now marked non-indexable and
the canary catches it.

## Results

Baseline vs ring, same-cluster mode, best of 15 passes across 5 processes.
Full data: [`results/baseline.json`](../../results/baseline.json),
[`results/ring.json`](../../results/ring.json).

### Submission cost (`submit`)

| variant | baseline p50 | ring p50 | baseline ops/s | ring ops/s | host CPU ns/op |
|---|---:|---:|---:|---:|---:|
| fill_64B (async) | 250 | **84** | 3.49 M | **7.82 M** | 286 → 128 |
| fill_4KiB (async) | 292 | **83** | 2.77 M | **8.54 M** | 362 → 117 |
| fill_64B_sync (round trip) | 250 | 208 | 3.50 M | 4.02 M | 286 → 249 |

An asynchronous submit costs 84 ns — twice the API floor of 42 ns — and the
p99 is 125 ns. The round trip itself shed ~40 ns because the protocol got
simpler (no ticket handshake, one release store each way).

### Amortization (`batch`: B fills then one sync)

| B | baseline ns/op | ring ns/op | waits/op | avg cmds per device wake | device util |
|---:|---:|---:|---:|---:|---:|
| 1 | 224 | 183 | 1.000 | 1.00 | 0.17 → 0.33 |
| 4 | 209 | 104 | 0.250 | 1.38 | 0.18 → 0.54 |
| 16 | 205 | 97 | 0.062 | 1.10 | 0.18 → 0.59 |
| 64 | 205 | 95 | 0.016 | 1.04 | 0.18 → 0.60 |
| 256 | 204 | 94 | 0.004 | 1.02 | 0.18 → 0.61 |
| 1024 | 204 | 94 | 0.001 | 1.02 | 0.18 → 0.62 |

Cost per command halves as soon as the host stops waiting (B ≥ 4), then
flattens at ~94 ns. That floor is the *submitter's* per-command work — lock,
validation, slot write, doorbell — not the device's. `avg_batch ≈ 1` says
the device drains each 64-byte fill before the next one arrives: it is
faster than the host can feed it. **The bottleneck has moved from the round
trip to the submission path itself.**

### Concurrent submitters (`mt`, 4 KiB memset per op)

| threads | baseline ops/s | ring ops/s | baseline p99 | ring p99 | host CPU ns/op |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.66 M | **5.60 M** | 584 ns | 334 ns | 364 → 164 |
| 2 | 1.44 M | **3.49 M** | 11.3 µs | 1.46 µs | 787 → 466 |
| 4 | 1.17 M | **2.94 M** | 71.3 µs | 1.88 µs | 1,349 → 621 |
| 8 | 0.98 M | **2.92 M** | 234 µs | 34.8 µs | 2,215 → 1,006 |

Throughput 2–3×, tail latency down 85–97%, half the CPU per op. But look at
the shape: the ring is flat from 2 to 8 threads. The lock convoy is gone
(threads no longer hold the lock across a round trip) but the lock itself
still serializes every submit. That is stage 4.

### Real workload (`vadd`: H2D, H2D, VADD, D2H)

| n | baseline µs/iter | ring µs/iter | waits/iter | kernel fraction |
|---:|---:|---:|---:|---:|
| 1,024 | 7.00 | 6.34 (2.4–2.8 in isolation, see below) | 4 → 2 | 0.086 → 0.035 |
| 65,536 | 45.3 | 45.4 | 4 → 2 | 0.20 → 0.005 |
| 1,048,576 | 804 | 796 | 4 → 2 | 0.21 → 0.000 |
| 16,777,216 | 14,337 | 14,275 | 49 → 47 | 0.20 → 0.000 |

Waits per iteration halved exactly as predicted (VADD and the first H2D no
longer block), and the kernel's *visible* cost vanished because it overlaps
with the host's D2H submission. But wall time barely moved: the copies are
still serialized through one staging buffer and still dominate. Stage 3.

### Unchanged, as expected

`memcpy` bandwidth within ±3% at every size (the staging path is the same);
`gemm` within ±1% (compute-bound work never cared about submission cost).

## Ring depth

`SG_RING_DEPTH` swept over 16 … 4096 slots (64 B each), `batch`/`mt`/`vadd`
only. Files: `results/ring-d{16,64,256,4096}.json` and `results/ring.json`
(1024).

| depth | batch B≥64 ns/op | avg cmds/wake | mt 8-thread ops/s | mt 8-thread p99 | stalls/op (8 thr) |
|---:|---:|---:|---:|---:|---:|
| 16 | 95 | 1.02 | 2.58 M | 68 µs | 0.031 |
| 64 | 96 | 1.03 | 2.64 M | 61 µs | 0.007 |
| 256 | 95 | 1.04 | 2.92 M | 35 µs | 0.001 |
| **1024** | **76** | **2.4** | **3.30 M** | **23–31 µs** | 0.000 |
| 4096 | 80 | 2.2 | 3.63 M | 32 µs | 0.000 |

Below 256 slots, contended submitters hit a full ring and stall. At 1024 the
device finally gets to fall behind and batch (2.4 commands per wake), which
is worth ~20% per command and ~13% aggregate throughput over 256. 4096 buys
nothing further outside the noise, costs 4× the memory (256 KiB vs 64 KiB)
and 4× the worst-case drain time behind a `sgDeviceSynchronize`.

**Default depth: 1024.** Round trips are unaffected by depth, so latency-
sensitive callers lose nothing.

## Open questions

Two anomalies survived the measurement clean-up and are recorded rather than
hidden:

1. **`vadd n=1024` depends on what ran before it.** In a process that runs
   only `vadd` (or `batch` then `vadd`) it costs 2.4–2.8 µs; after `memcpy`
   or `gemm` in the same process it costs ~6 µs, reproducibly, at every ring
   depth. The `frac_*` breakdown says the extra time is in the two H2D
   calls. Leading hypothesis: the 64 MiB `memcpy` pass leaves the staging
   buffer's lines owned by the device's cluster, and cross-cluster
   write-invalidations on the following 4 KiB `memcpy`s into staging are
   what we are paying for. Pinned host memory (stage 3) removes the staging
   write entirely, which will settle it.
2. **`batch B=1` vs `submit fill_64B_sync`** are the same operation, yet
   `batch B=1` occasionally reads 2× slower (460–480 ns) in runs whose
   `submit` rows are in fast mode. Placement flipping during a pass is the
   suspect; the post-run canary should catch it going forward.

## Consequences

* Everything above the driver is now asynchronous; a device error surfaces
  at the next wait, not the offending call. Applications that relied on
  baseline synchrony implicitly must call `sgDeviceSynchronize()` (the
  tests already did).
* `sgFree` drains the queue. Cheap today; wrong once there are multiple
  contexts. Deferred frees keyed on fences come with stage 3.
* The new floor is ~90 ns per command of submitter work under a global lock,
  and the device outruns it. The next measurable wins are, in order:
  overlap the copies (stage 3, `vadd` and `memcpy` are 2× on the table),
  stop spinning (stage 2, `cpu_ns_per_op ≈ wall/op` everywhere), and remove
  the lock (stage 4, flat `mt` scaling).
