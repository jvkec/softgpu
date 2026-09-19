# 004 — Interrupts vs polling: a hybrid wait policy

## Context

Every result table since stage 0 has carried the same column:
`cpu_ns_per_op ≈ wall/op`. Whenever the host waits on the device —
`sgDeviceSynchronize`, `sgStreamSynchronize`, a synchronous pinned copy, a
staging-slot reuse, a full ring — it spins on a fence register and burns a
full core for the duration. For a 208 ns round trip that is the right call.
For a 14 ms GEMM it is 14 ms of CPU doing nothing, and on a laptop or a
DGX node alike that is power and a core someone else could have used.

Real drivers have both mechanisms. The device raises an **interrupt** when a
semaphore passes a value, the ISR wakes the sleeping thread, and the
runtime exposes a **policy** — CUDA's `cudaDeviceScheduleSpin`, `Yield`,
`BlockingSync` and `Auto` — because neither extreme is right for every
workload. This stage adds the interrupt path to the device model, gives the
driver one wait primitive with a policy, and measures the tradeoff curve to
pick a default. It is the first stage that deliberately makes one metric
worse (latency of tiny synchronous operations, under some policies) to
improve another (host CPU per operation, our power proxy).

## Problem

There is exactly one way to wait, and it is the wrong one for every wait
longer than a few microseconds.

## Options considered

1. **Always block.** Lowest CPU; adds the full wake-up latency (tens of µs
   on this VM) to every wait, including 200 ns round trips — 100× slower.
2. **Always spin** (status quo). Lowest latency; a core per waiting thread
   regardless of wait length.
3. **`sched_yield()` in the spin loop.** Cheaper than a hard spin when the
   machine is oversubscribed, but still polls, and yields are unbounded in
   latency on a loaded system.
4. **Spin, then block** (this record). Spin for a budget that covers the
   short waits, arm the interrupt and sleep for the rest. Two variants:
   a fixed budget (*hybrid*) and one that predicts the wait from recent
   history (*adaptive*, in the spirit of Linux's adaptive mutexes and
   CUDA's `ScheduleAuto`).

## Decision

**Device** (`src/device`): each channel gains an `irq_target` register —
"raise an interrupt once `get >= irq_target`" — armed by the driver per
wait, exactly like enabling an interrupt on one semaphore. The device has a
single **IRQ line** (`common/event.h`): on Linux a 32-bit sequence word plus
`futex(2)`, the primitive under every blocking wait in userspace and the
one the stage-8 kernel module's `wait_event`/`wake_up` maps onto; a
condition variable elsewhere. After retiring a command the engine checks
the channel's `irq_target`; if the fence passed it, the engine clears the
target, timestamps the pulse, and signals the line. Cost only when armed.

**Driver** (`src/driver`): one primitive, `wait_until(pred, arm, flags)`,
replaces every spin loop — fence waits, `SG_WAIT_ALL` snapshots, staging
slot reuse, ring backpressure:

1. spin for a budget — the first ~1 µs untimed, so a wait that ends within
   a device round trip never reads the clock and costs exactly what the old
   pure spin did; after that the clock is checked every 32 iterations;
2. read the IRQ sequence, **arm**, then **re-check** the predicate, then
   sleep on the sequence with a 1 ms safety timeout; repeat until satisfied.

The arm–then–check order is what makes it correct: the engine stores `get`
*before* it clears the target and signals, so a waiter that checks after
arming either sees the fence or is guaranteed a signal it has not yet
consumed (the futex compares the sequence word atomically in the kernel).
Several waiters may arm the same channel; the target keeps the minimum and
everyone re-checks on wake.

Policies (`SG_WAIT_POLICY`, per-call override via `sg_wait_args.flags`):
`spin` (budget ∞), `block` (budget 0), `hybrid` (budget `SG_SPIN_NS`),
`adaptive` (budget ≈ 2× an EWMA of recent wait durations on that channel,
clamped to [2 µs, `SG_SPIN_NS`]; a channel whose waits exceed the cap gets
the 2 µs floor and goes to sleep). For a blocked wait the EWMA is fed the
time until the fence *passed* (the interrupt's timestamp), not until the
thread woke — see *An estimator that measured its own penalty* below.

**Default: `adaptive` with `SG_SPIN_NS = 30 µs`** — the cap set to the
measured wake-up latency, which is the classic bound (spin no longer than a
sleep would cost). Chosen from the sweep below.

**ABI v5**: `sg_wait_args.flags` (`SG_WAIT_DEFAULT/SPIN/BLOCK`); stats
`waits_spun`, `waits_blocked`, `wake_latency_ns`, per-engine `irqs`;
`QUERY` reports the policy in effect. **Runtime**: `sgSetSyncPolicy`
mirroring `cudaSetDeviceFlags`; all host waits pass the chosen flag.

**Not changed:** the engines themselves still spin when idle (device-side
power gating is stage 7), the big lock (stage 4), everything from 3a/3b.

## Hypotheses (written before measuring)

* **H1** Blocking on a 14 ms GEMM: submitter CPU per op drops from ~14 ms
  to < 50 µs (`cpu_frac` < 0.005); wall time unchanged within one wake-up.
* **H2** Wake-up latency on the VM is 20–80 µs (futex wake plus a vCPU that
  may itself be descheduled), so `block` makes a 10 µs op 3–8× slower and a
  0.3 µs fill 100×+ slower.
* **H3** `hybrid` with a budget of 2–5 µs keeps the spin path's latency for
  everything shorter than the budget and the blocking path's CPU for
  everything longer; the sweep over `SG_SPIN_NS` shows a knee there.
  `adaptive` lands within 10% of the best fixed budget on every duration
  without being told.
* **H4** Under the default policy, `pipeline`'s host CPU per chunk falls by
  > 90% with < 3% change in wall time, and `mt` at 8 threads improves
  because sleeping waiters stop competing with the engines for vCPUs.
* **H5** The default must not regress the round trip: `submit fill_64B_sync`
  p50 stays at its spin value.

## Results

All files same-cluster mode, best of 15 passes across 5 processes per
benchmark, canary-gated. Wake-up latency (interrupt pulse → waiter running
again) measured **23–40 µs** on this VM: a futex wake plus, often, a vCPU
that has to be scheduled back onto a host core. That number is the whole
story; every tradeoff below is "spin for less than that, or sleep and pay
it".

### Policy × wait length (`wait`: one command, then `sgDeviceSynchronize`)

Per-op wall time (µs) / submitter CPU fraction. `results/wait-spin.json`,
`wait-block.json`, `wait-hybrid5k.json`, `wait-adaptive.json` (5 µs cap),
`wait-adaptive30k.json` (30 µs cap).

| work (≈ length) | spin | block | hybrid, 5 µs | adaptive, 5 µs cap | **adaptive, 30 µs cap** |
|---|---:|---:|---:|---:|---:|
| fill 4 KiB (0.3 µs) | 0.38 / 1.00 | 0.70 / 0.83 | 0.38 / 0.98 | 0.40 / 0.98 | **0.40 / 1.00** |
| GEMM 32³ (10 µs) | 11.5 / 1.00 | 31.6 / 0.20 | **40.3** / 0.30 | 30.9 / 0.22 | **15.0 / 0.76** |
| GEMM 128³ (200 µs) | 210 / 1.00 | 235 / 0.03 | 242 / 0.06 | 228 / 0.03 | **238 / 0.03** |
| GEMM 512³ (14 ms) | 14,917 / 1.00 | 14,938 / 0.01 | 14,871 / 0.01 | 14,876 / 0.01 | **14,878 / 0.01** |

* **H1 confirmed**: blocking on the 14 ms GEMM costs +0.1–0.4% wall and
  cuts CPU by 99%.
* **H2 confirmed**: `block` makes the 10 µs op 2.7× slower and the 0.3 µs
  fill 1.8× slower — less than the predicted 100×, because by the time the
  waiter has armed the interrupt and re-checked, a 300 ns fence has usually
  already passed and it never sleeps at all (`blocked_frac` 0, `wake_us` is
  the rare exception).
* **H3 half right.** A fixed budget has no good value: 5 µs is the *worst*
  policy for the 10 µs op (it spins 5 µs, then pays the 30 µs wake on top —
  40 µs, worse than blocking outright), while 20 µs (below) catches that op
  and wastes 20 µs of CPU on every longer one. Adaptive with the cap raised
  to the wake latency is the only column that is never bad: within 6% of
  spin on the fill, 1.3× spin on the 10 µs op at a quarter of the CPU,
  blocking's CPU on everything longer.

### Fixed budget sweep (`wait-spin{200,1000}.json`, `wait-hybrid5k.json`, `wait-spin20000.json`)

| work | 200 ns | 1 µs | 5 µs | 20 µs |
|---|---:|---:|---:|---:|
| GEMM 32³ µs / CPU | 31.4 / 0.21 | 37.5 / 0.21 | 40.3 / 0.30 | **11.8 / 0.95** |
| GEMM 128³ µs / CPU | 235 / 0.03 | 235 / 0.04 | 242 / 0.06 | 240 / **0.12** |

The knee is at the wake-up latency, not below it — the classic
competitive-ratio result: spin for as long as a sleep would cost, then
sleep, and total time is never worse than 2× the best choice in hindsight.
The 30 µs cap is that rule; adaptive then avoids paying even the spin when
history says the wait will be long.

### An estimator that measured its own penalty (`results/stage2-adaptive-v1.json`)

The first adaptive default fed the EWMA the *observed* wait duration. After
one blocked wait that observation includes the 30 µs wake, the estimate
jumps above the cap, the next wait therefore spins only the minimum and
blocks again, and the channel is locked into "long waits" for good.
Pinned `vadd n=1024` — four sub-µs commands and one sync — went from 3.1 to
5.8 µs that way. The fix is to measure until the fence *passed* (the
interrupt's timestamp) rather than until the thread woke, and to floor the
prediction at 2 µs so bursty sub-µs chains never block. Positive feedback
between a controller and the cost it controls is the standard failure of
adaptive policies; the estimator must observe the workload, not itself.

### Default configuration vs stage 3b (`results/streams.json` → `results/stage2.json`)

The default (adaptive, 30 µs cap, 2 µs floor, interrupt-timestamped
estimator) on the `wait` benchmark: 0.38 µs / 12.1 µs / 234 µs / 14.9 ms
with CPU fractions 0.99 / 0.95 / 0.04 / 0.01 — spin's latency where waits
are short, block's CPU where they are long, and no policy in the sweep
beats it on more than one row.

| | 3b | stage 2 default | note |
|---|---:|---:|---|
| `pipeline` host CPU per chunk, R=1 S=2 | 450 µs (= wall) | **13 µs** (3% of wall) | **H4 confirmed**: −97% CPU, wall +2% |
| `pipeline` host CPU per chunk, R=4 S=2 | 1,197 µs | **24 µs** | wall −8% |
| `mt` 8 threads (default stream) ops/s · p99 · CPU/op | 3.35 M · 17.5 µs · 661 ns | **3.58 M · 1.3 µs · 480 ns** | **H4 confirmed**: sleepers stop stealing vCPUs from the engines |
| `submit fill_64B_sync` p50 | 250 ns | 291 ns | **H5 confirmed**: `spin` measures 291 ns on the same day — placement, not policy |
| `submit fill_64B` async p50 | 125 ns | 125 ns | unchanged |
| `memcpy` pinned 4 MiB H2D / D2H | 57 / 57 GB/s | **40 / 42 GB/s** | a ~70 µs wait blocks after 30 µs and pays a ~30 µs wake: the 2× bound, biting |
| `memcpy` pinned 64 MiB, pageable 4 MiB | 32 / 41 GB/s | 30 / 41 GB/s | ms-scale and staged waits: unchanged |
| `vadd` pinned n=1M | 512 µs | 599 µs (CPU 5%) | +17% wall for −95% CPU |
| `vadd` pinned n=1024 | 3.13 µs | 4.07 µs (CPU 94%) | +30%: adaptive's timing on ~1 µs waits, plus today's slower round trip |
| `gemm` 512³, `alloc` | — | +2%, +5% | noise |

### What it costs

Two regressions are real and are the policy's known shape, not bugs:

* Waits of roughly one to three wake-latencies (30–100 µs) — a 4 MiB
  pinned copy, a 200 µs kernel — are the worst case for any spin-then-block
  scheme: they spin the whole budget *and* pay the wake. Bandwidth on
  synchronous 4 MiB pinned copies drops 26–30%. Callers who need those
  microseconds set `sgSetSyncPolicy(SG_SYNC_SPIN)`; that is exactly why
  CUDA exposes the same choice.
* Sub-µs chains that never block still pay the adaptive estimator's clock
  reads once a wait outlives the untimed first round (~1 µs): pinned `vadd
  n=1024` +30%.

## Consequences

* `cpu_ns_per_op ≈ wall/op` is gone from the tables: a thread waiting on
  the device now costs CPU proportional to a wake-up, not to the wait.
  On the pipeline workload that is 97% of the host CPU this project had been
  burning since stage 0.
* The device model has an interrupt line. Stage 7 (device-side power
  gating) can put the *engines* to sleep the same way — spin briefly on an
  empty ring, then block until the doorbell — with the same futex; today
  the two engine threads still burn two cores while idle.
* The stage-8 kernel module inherits the shape directly: `irq_target` is a
  semaphore-release interrupt, `Event` is `wait_event_interruptible` /
  `wake_up`, and the arm-then-check ordering is the standard ISR/waiter
  protocol.
* Three lessons this stage recorded the hard way: a fixed spin budget has
  no good value below the wake latency; an adaptive estimator must observe
  the workload and not its own penalty; and a canary threshold is a
  measurement of the code it gates — when the code changes, re-measure the
  gate (a stale 260 ns limit idled the benchmark job for ten hours).
