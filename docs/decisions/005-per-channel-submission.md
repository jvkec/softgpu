# 005 — Per-channel submission: removing the big driver lock

## Context

Since stage 0 every driver ioctl except `WAIT` has run under one
`std::mutex`. It was the right first choice — one shared mailbox, one
allocator — and each later stage kept it deliberately so its own effect
stayed attributable. The cost has been in every `mt` table: throughput
*falls* from 1 to 8 submitting threads (7.4 M → 3.6 M ops/s at stage 2)
while the compute engine sits at ~50% utilization, and submitter CPU per
operation climbs from ~150 to 500–700 ns as threads queue on the lock.

Stage 3b made the lock unnecessary for the hot path: channels are
independent rings with their own PUT/GET, and each stream already owns a
channel. What the lock still protected was a handful of shared structures —
the VRAM allocator, the pin registry, the staging pool, the deferred-release
list, and plain counters — none of which need to serialize *submission*.
Real drivers end up in the same place: per-channel submission with no
global lock (ultimately user-mode doorbells), small locks or RCU around the
rare shared state.

## Problem

One lock serializes all submitters regardless of whether they share
anything, and it is held across the whole of a pageable copy — including
the host `memcpy` into staging and the wait for a free slot.

## Options considered

1. **Keep the lock, shrink the critical section.** Helps pageable copies,
   does nothing for independent streams submitting tiny commands, which is
   the common case.
2. **Per-channel locks + small locks for shared state** (this record).
   Two streams on different channels never touch the same lock; the
   allocator, pin registry and staging pool get their own, taken shared on
   the read-mostly submit path.
3. **Lock-free per-channel ring (ticket reservation).** Producers claim a
   slot with `fetch_add` and publish in ticket order. No lock at all, at the
   price of the classic hazard: a producer descheduled between reserving
   and publishing stalls every later publisher. Built as `SG_SUBMIT_MODE=
   ticket` and measured against option 2 on a deliberately shared channel.
4. **User-mode submission** (map the ring and doorbell into the process,
   no ioctl). Where real GPUs end up; it is the stage-8 kernel module's
   `mmap` path and needs the kernel side to exist first.

## Decision

**Per-channel producers** (`src/driver/driver.cpp`): `put` mirrors become
atomics; each `(engine, channel)` gets a `Producer` on its own cache line
with a mutex (default) or a ticket counter. A submit holds only its
channel's lock for the whole command — all chunks of a pageable copy stay
contiguous — and nothing else for longer than a bookkeeping step.

**Shared state, small locks:**
* allocator + deferred-release list: one `shared_mutex`; `owns_range()`
  validation on the submit path takes it shared, `ALLOC`/`FREE`/reclaim
  exclusive;
* pin registry: `shared_mutex`; the per-copy `pinned()` lookup is shared;
* staging pool: a plain mutex around slot bookkeeping only. Slots carry a
  `busy` flag so two concurrent pageable copies can never be handed the
  same slot; the host `memcpy` and the wait for a slot's previous DMA both
  happen outside the lock;
* counters: relaxed atomics;
* snapshots (`FREE`, `UNPIN`, `SG_WAIT_ALL`, close): read the atomic PUTs
  with no lock. A snapshot of monotonic counters is "at least everything
  whose submit had returned before the call", which is exactly the promise.

Lock order where two are ever held: `pin_lock` → `alloc_lock` (UNPIN moves
a range onto the pending list). Channel locks nest inside nothing.

**Telemetry:** `lock_wait_ns` — accumulated only when a `try_lock` fails, so
the uncontended path pays no clock read. It is the before/after number.

**Runtime:** unchanged. A stream's mutex stays: a stream *is* an ordered
sequence, and threads sharing one serialize by definition (CUDA's default
stream has the same property). The `mt` benchmark's two variants — all
threads on the default stream, one stream per thread — separate that from
the driver.

**Not changed:** device model, channels/streams, staging geometry, wait
policy, allocator algorithm. ABI v6 adds only the stat and `QUERY.submit_mode`.

## Hypotheses (written before measuring)

* **H1** `mt`, one stream per thread: 8 threads from 3.9 M to ≥ 8 M ops/s,
  limited by the machine (6 vCPUs minus 2 always-on engines) and the compute
  engine's fill rate, not by a lock; `lock_ns_per_op → 0`.
* **H2** `mt`, all threads on the default stream: unchanged — the stream
  mutex serializes them, as designed.
* **H3** `ticket` vs `mutex` on one shared channel: lower p50 under light
  contention, worse p99 once threads exceed cores (a preempted reserver
  stalls the publishers behind it). The mutex stays the default unless the
  data says otherwise.
* **H4** Single-threaded numbers unchanged within noise: an uncontended
  mutex and an atomic cost the same ~20 ns.
* **H5** ThreadSanitizer finds at least one real race in the first cut —
  every stage that touched concurrency has (3a's `waits` counter, 3b's
  shared-ring ordering).

## Results

Stage 2 (`results/stage2.json`, `streams-mt.json`) vs stage 4
(`results/nolock.json`), same-cluster mode, best of 15 passes across 5
processes per benchmark, canary-gated. TSan and ASan clean across all eight
test variants.

### Scaling (`mt`: 4 KiB memset per op, 8 threads max on 6 vCPUs, 2 of them engines)

| threads | one stream per thread: before → after ops/s | p99 | compute-engine util | all on the default stream: before → after |
|---:|---:|---:|---:|---:|
| 1 | 5.38 M → 5.66 M | 542 → 208 ns | 0.85 → 0.90 | 7.55 M → 5.64 M |
| 2 | 5.15 M → 5.74 M | 1.1 µs → 0.5 µs | 0.96 → 0.96 | 4.00 M → 3.81 M |
| 4 | 3.40 M → **4.10 M** | 1.7 → 1.7 µs | 0.79 → 0.90 | 3.09 M → 2.82 M |
| 8 | 3.76 M → **4.29 M** | 18.0 → **1.9 µs** | 0.78 → **0.95** | 3.58 M → 2.94 M |

`lock_ns_per_op` is 0.000 in every row: no submitter ever waited on a driver
lock. With one stream per thread, 8 threads now run the compute engine at
95% — **the driver is no longer the bottleneck; the device is** (one engine
retiring 4 KiB fills at ~4.5 M/s including the runlist scan). That is also
why **H1's "≥ 8 M ops/s" was wrong**: the ceiling moved, but to the engine,
not to the machine. Tail latency at 8 threads fell 90%, which is the lock
convoy disappearing.

**H2 confirmed, with a cost.** Threads sharing the default stream are
serialized by the stream's own mutex exactly as before, and every row got
5–25% slower: the fine-grained locking is paid on a path that gains nothing
from it (below).

### Lock-free ticket vs mutex on one shared channel (`SG_CHANNELS=1`, one stream per thread)

| threads | mutex ops/s · p50 · p99 | ticket ops/s · p50 · p99 |
|---:|---:|---:|
| 1 | 5.23 M · 125 ns · 167 ns | 5.87 M · 125 ns · 167 ns |
| 2 | 4.85 M · 250 · 541 | 4.44 M · 250 · 917 |
| 4 | 2.92 M · 167 · 1,917 | 2.77 M · 1,041 · 1,667 |
| 8 | 3.30 M · 166 · 4,125 | **0.83 M · 7,541 · 86,748** |

**H3 confirmed, and then some.** Uncontended, the ticket path is 12% faster
(one `fetch_add` instead of a lock/unlock pair). At 8 producers on 4 free
cores it collapses: −75% throughput, p50 45×, p99 21×, CPU per op 8×. The
first version did not degrade — it **livelocked**: publishers spun waiting
for the head ticket, the head-ticket holder was descheduled by those very
spinners, and 8 threads burned 100% CPU forever (the benchmark ran five
hours before a timeout was added). A mutex degrades gracefully because a
blocked waiter gives up its core. The fix was spin-then-`yield` in the
publish wait, which turns the livelock into the slow but finite row above.
On separate channels (`results/nolock-ticket.json`) ticket and mutex are
within noise of each other, as expected — there is nothing to contend on.

**Decision: the per-channel mutex stays the default.** Lock-free publishing
only wins when producers never sleep, which is exactly the condition an
oversubscribed host violates.

### A second ordering bug TSan could not see

The first ticket implementation stored the driver's PUT mirror, then the
device's PUT register. Producer *T+1* could observe the mirror advance and
store its own doorbell before *T*'s landed, so the device register went
*backwards*, the engine's GET overtook PUT, and the engine executed stale
ring slots forever with `running_` already false. Every access involved was
atomic, so ThreadSanitizer reported nothing; a hung process with one thread
at 100% did. Two changes: the doorbell is now written *before* the ticket is
handed on, and the device faults a channel whose PUT is behind its GET
instead of running off the end of the ring — what real hardware does.
**H5 confirmed**, but not in the form predicted: the bugs of this stage were
ordering and liveness bugs, which a data-race detector is blind to by
construction. The `ticket_publish_storm` test (two producers per channel,
16 streams over 8 channels) reproduces the interleaving.

### What it costs on the uncontended path (`stage2.json` → `nolock.json`)

| | before | after |
|---|---:|---:|
| `submit fill_64B` async p50 | 125 ns | 166 ns |
| `submit fill_64B_sync` p50 | 291 | 292 |
| `alloc` malloc+free, idle / in flight | 250 / 875 ns | 292 / 1,125 ns |
| `memcpy`, `pipeline`, `vadd`, `gemm` | — | within ±2% (pinned 4 MiB +8%) |

**H4 refuted.** About 40 ns per submit: a `shared_mutex` reader lock for the
address-range check, the channel mutex, and a handful of relaxed atomics
replace one uncontended mutex. On a 125 ns path that is a third. The
scaling case wins it back many times over, but single-threaded callers pay
it. A cheaper read path for validation — an epoch/RCU-style snapshot of the
allocation map instead of a reader lock — is the obvious follow-up.

## Consequences

* The driver has no global lock. Submission on distinct channels shares
  nothing; the shared structures (allocator, pins, staging) are behind small
  locks taken shared on the hot path. `lock_ns_per_op` is the standing
  regression test for that claim.
* The bottleneck moved onto the device for the first time in the project's
  multithreaded case: one compute engine at 95%. Stage 6 (scheduling policy
  across channels) and any "more engines" experiment start from here.
* Per-channel producers are the shape user-mode submission needs: the
  stage-8 kernel module can `mmap` a channel's ring and doorbell into the
  process and skip the ioctl entirely, because nothing in submission now
  requires kernel-side serialization beyond the channel itself.
* Two of the three bugs this stage found were invisible to ThreadSanitizer
  — a doorbell ordering bug and a livelock — and were found by a hang.
  Timeouts on every benchmark step and test are now part of the harness;
  "the sanitizer is clean" is necessary and was never sufficient.
* Uncontended cost went up ~40 ns/op; the fix (lock-free validation reads)
  is known and deferred.
