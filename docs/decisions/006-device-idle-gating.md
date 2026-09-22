# 006 — Device idle gating

## Context

Stage 2 ([004](004-interrupts-and-wait-policy.md)) fixed the host side of
"who burns a core waiting": a thread waiting on the device sleeps until an
interrupt. The device side never changed. Both engine threads have spun at
100% on empty runlists since stage 0 — an idle `softgpu` costs two full
cores — and every `dev_util` figure so far has been honest about *work*
and silent about *power*. Real GPUs clock-gate and power-gate idle engines
behind a hysteresis timer and pay for it with wake-up latency on the first
command after an idle period.

This stage builds that, in the same shape as stage 2 pointed the other way:
an engine spins briefly on an empty runlist, then sleeps on a futex until a
doorbell — or another engine's fence — wakes it.

## Problem

The device's idle power is 2.0 cores by construction, and nothing in the
model can trade it against latency.

## Options considered

1. **Leave the engines spinning.** Zero wake-up latency; two cores forever.
2. **`sched_yield()` on empty rounds.** Cheaper under oversubscription, still
   a busy loop, unbounded wake-up under load.
3. **Sleep after an idle budget, wake on doorbell** (this record), with the
   budget fixed (*hybrid*) or predicted from recent idle-gap lengths
   (*adaptive*), plus `spin` and `sleep` as the two ends of the curve.

## Decision

**Device** (`src/device`): each engine gets a doorbell `Event` (the stage 2
futex primitive) and an `asleep` flag. In the runlist's no-progress branch
the engine tracks how long the current idle stretch has lasted; past the
budget it **arms** (`asleep = true`), **re-checks** every channel's PUT and
every blocked head's fence, and only then sleeps (1 ms safety timeout). On
wake it clears the flag and resumes. Two wake sources: the driver's
`Device::doorbell(engine)` after every PUT store, and `wake_sleepers()`
after every retirement (a sleeping engine may be blocked on that fence).
`power_off` wakes everyone so they can exit.

**Dekker.** Driver: store PUT, fence, load `asleep`. Engine: store `asleep`,
fence, load PUTs. Both are store-then-load; without `seq_cst` fences on
both sides ARM64 may reorder either, and a doorbell rung between the
engine's check and its sleep is lost until the timeout. The device counts
such events (`missed_doorbells`: a timeout wake that finds work), the
`gating_no_missed_doorbells` test asserts zero, and
`SG_EXPERIMENT_NO_FENCE=1` removes the fences so the race can be measured
rather than asserted.

**Policy** (`SG_ENGINE_IDLE`, `SG_ENGINE_IDLE_NS`): `spin`, `sleep`,
`hybrid` (fixed budget), `adaptive` (EWMA of recent idle-gap lengths —
long gaps → sleep after a 2 µs floor, short gaps → spin up to 2× the
typical gap, capped). Default decided by the sweep below.

**Telemetry:** per-engine `sleep_cycles`, `wakeups`, `missed_doorbells`, and
`cpu_ns` — the engine thread's own `CLOCK_THREAD_CPUTIME_ID`, published on
every idle→busy transition, before each sleep, and about once a millisecond
while spinning. The benchmark reports it as `dev_cpu_cores` (engine CPU
per wall second; 2.0 = both engines spinning) — the device power proxy,
which until now was 2.0 in every row without being shown.

**ABI v7**: the stats above plus `QUERY.idle_policy`/`idle_ns`. No API
change.

**Not changed:** host wait policy, channels, locks, staging, allocator.

## Hypotheses (written before measuring)

* **H1** Idle device: engine CPU falls from 2.0 cores to < 0.05 under
  gating; during `wait`'s 14 ms GEMM the copy engine's CPU goes to ~0 while
  the compute engine stays at 1.0.
* **H2** The first command after a long idle pays a device wake-up of
  30–60 µs (futex wake plus a cold vCPU); after gaps shorter than the budget
  it costs nothing.
* **H3** Throughput benchmarks are unchanged within noise under the default
  — their inter-command gaps never reach the budget.
* **H4** The knee of the budget sweep sits at the wake-up latency again
  (~30 µs); adaptive matches the best fixed budget without tuning.
* **H5** Dropping the fences produces measurable lost doorbells on ARM64
  (kept as a negative experiment, `results/gated-nofence.json`).

## Results

_(from `results/gated.json`, `gate-{spin,sleep,h10us,h50us,h200us}.json`, `gated-nofence.json`)_

## Consequences

_(after results)_
