# 000 — Baseline results

Environment: UTM/QEMU VM on Apple M-series, 6 vCPUs, 15 GiB, Ubuntu 24.04.5,
Linux 6.8.0-139-generic aarch64, GCC 13.3 `-O3`. Data: [`results/baseline.json`](../../results/baseline.json),
produced by `scripts/vm.sh bench baseline all --threads 8`. Tests pass under
TSan and ASan/UBSan (clang 18) on the same VM.

Metric reminders: `cpu_ns_per_op` is *submitter-thread* CPU per API call
(host power proxy); `dev_util` is device busy / (busy + idle) over the run;
`dev_cmds_per_op` is how many device commands one API call became.

## Round-trip cost (`submit`)

| variant | p50_ns | p99_ns | ops_per_s | cpu_ns_per_op | dev_util |
|---|---:|---:|---:|---:|---:|
| nop (API floor) | 41 | 42 | 19,361,500 | 52 | 0.000 |
| fill_64B | 209 | 250 | 3,705,180 | 270 | 0.151 |
| fill_4KiB | 333 | 375 | 2,604,220 | 384 | 0.388 |

A device round trip costs **~170 ns above the API floor**: mutex, mailbox
write, doorbell release, cross-core wake of the engine, execute, `completed`
release, cross-core observe. p99 is within 20% of p50 — spinning on both
sides buys excellent tail latency, at the price of burning a full core on
each side. `cpu_ns_per_op ≈ wall/op` confirms the driver never sleeps.

**Hypothesis 1 confirmed** (sub-µs round trip, 100% CPU while waiting).

## Data path (`memcpy`)

| variant | GBps | ns_per_op | dev_cmds_per_op | dev_util |
|---|---:|---:|---:|---:|
| h2d_4KiB | 5.6 | 731 | 1 | 0.24 |
| h2d_64KiB | 14.2 | 4,620 | 1 | 0.41 |
| h2d_1024KiB | 24.5 | 42,800 | 1 | 0.54 |
| h2d_4096KiB | 24.7 | 170,000 | 1 | 0.55 |
| h2d_16384KiB | 18.3 | 918,000 | 4 | 0.41 |
| h2d_65536KiB | 17.9 | 3,750,000 | 16 | 0.40 |

(D2H is within 10% of H2D at every size.)

Three regimes:

* **Small (≤ 64 KiB):** fixed per-command cost dominates; 4 KiB copies run at
  5.6 GB/s because ~500 ns of the 731 ns is round trip, not bytes.
* **Medium (1–4 MiB):** plateau at ~24 GB/s. `dev_util ≈ 0.5` is the
  signature of the staging design: host `memcpy` into staging, *then* device
  `memcpy` out of it, strictly serialized, each moving every byte once. The
  machine can clearly do ~50 GB/s per copy; we get half.
* **Large (≥ 16 MiB):** drops to ~18 GB/s once the copy exceeds the 4 MiB
  staging buffer and becomes 4–16 serialized chunks, each with its own
  round trip and no overlap between chunk *n*'s DMA and chunk *n+1*'s memcpy.

**Hypothesis 2 confirmed.** Two independent fixes are visible in the data:
overlap the two copies (pipeline the staging buffer, ~2×), and remove one of
them entirely (pinned/zero-copy host memory, up to ~2× again). Stage 3.

## Small real workload (`vadd`: H2D a, H2D b, VADD, D2H c)

| n | us_per_iter | frac_h2d | frac_kernel | frac_d2h | dev_cmds_per_op | dev_util |
|---|---:|---:|---:|---:|---:|---:|
| 1,024 | 7.3 | 0.61 | 0.09 | 0.30 | 4 | 0.37 |
| 65,536 | 46.6 | 0.54 | 0.19 | 0.27 | 4 | 0.55 |
| 1,048,576 | 801 | 0.53 | 0.21 | 0.27 | 4 | 0.58 |
| 16,777,216 | 14,500 | 0.53 | 0.20 | 0.27 | 49 | 0.42 |

The kernel is **9–21% of wall time**; copies are the rest at every size.
At n=1024 the whole iteration is 7.3 µs for four API calls — 1.8 µs each,
which is round trip plus tiny copies through staging. There is no size at
which this workload is compute-bound.

**Hypothesis 3 confirmed**, and sharper than expected: even at 64 MiB of
data the kernel is a fifth of the time.

## Compute-bound work (`gemm`)

| dims | GFLOPS | us_per_op | dev_util |
|---|---:|---:|---:|
| 32³ | 6.0 | 10.9 | 0.96 |
| 128³ | 21.7 | 194 | 0.995 |
| 512³ | 19.6 | 13,700 | 1.00 |

When the device has real work, the driver disappears: utilization is ~100%
and the round trip is noise. The 32³ case is the crossover — 10.9 µs of
compute vs ~0.3 µs of submit. `GFLOPS` is the naive engine's single-core
scalar rate and is not interesting in itself; it is the yardstick for
"device time" in later stages.

**Hypothesis 5 confirmed.** It also gives us a rule of thumb: anything under
~10 µs of device work is submission-bound in this design.

## Concurrent submitters (`mt`, 4 KiB `memset` per op)

| threads | ops_per_s | p50_ns | p99_ns | cpu_ns_per_op | dev_util |
|---|---:|---:|---:|---:|---:|
| 1 | 2,236,380 | 292 | 584 | 436 | 0.255 |
| 2 | 1,194,470 | 583 | 26,499 | 1,012 | 0.194 |
| 4 | 1,023,250 | 583 | 104,786 | 1,638 | 0.175 |
| 8 | 974,831 | 583 | 243,820 | 2,292 | 0.153 |

Adding a second thread **halves** throughput and multiplies p99 by 45×.
By eight threads: −56% throughput, p99 = 244 µs (418×), and 5.3× the CPU per
operation — while the device sits idle 85% of the time. This is a lock
convoy: every submitter serializes on one `std::mutex`, the futex
sleep/wake on handoff costs more than the work it protects, and the device
is starved while threads are being rescheduled. With 8 threads on 6 vCPUs
(one of them the engine) oversubscription adds scheduler latency on top.

**Hypothesis 4 confirmed**, strongly: the baseline scales *negatively*.

## Where this leaves us

Ranked by measured impact, the baseline's problems are:

1. **The big lock** (mt): negative scaling, 400× tail latency. Fix: per-thread
   or per-context submission queues with a lock-free or finely-locked ring
   (stage 1 ring buffer lays the groundwork; stage 4 finishes it).
2. **Serialized staging copies** (memcpy, vadd): 2–4× bandwidth on the table.
   Fix: pinned host memory + DMA engine + overlap (stage 3).
3. **One command in flight** (submit, vadd small n): every API call pays a
   full round trip. Fix: ring buffer with batched doorbell (stage 1).
4. **Spinning on both sides** (all): 100% CPU per op on the host, 100%
   "power" on the device even when idle. Fix: interrupts/fences and a hybrid
   wait policy (stage 2), device idle gating (stage 7). Note this will
   *worsen* p50/p99 for tiny ops — that tradeoff is the point of stage 2.

Stage 1 is the ring buffer. It attacks (3) directly and is the prerequisite
for (1), (2) and (4), because fences, multiple queues, and asynchronous DMA
all need somewhere for more than one command to live.
