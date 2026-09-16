# Stage 0 — Baseline

The baseline is the simplest architecture that is *correct*, *layered like
the real thing*, and *instrumented*. It is deliberately slow in specific,
well-understood ways so that every later stage fixes a measured problem
rather than an imagined one.

## Architecture

```
 application
     │  sg_runtime.h (C ABI)
 ┌───▼──────────────────────────────────────────┐
 │ runtime  (src/runtime)                       │  marshal args, map errors
 └───┬──────────────────────────────────────────┘
     │  sg_drv_open / sg_drv_ioctl / sg_drv_close   (syscall-shaped)
 ┌───▼──────────────────────────────────────────┐
 │ driver   (src/driver)                        │  big lock, VRAM allocator,
 │   validate → stage → mailbox → doorbell →    │  validation, staging copies,
 │   spin on `completed`                        │  one command in flight
 └───┬──────────────────────────────────────────┘
     │  Registers (MMIO model) + VRAM + host DMA addresses
 ┌───▼──────────────────────────────────────────┐
 │ device   (src/device)                        │  one engine thread, spins on
 │   FILL COPY_* VADD GEMM                      │  doorbell, executes, retires
 └──────────────────────────────────────────────┘
```

### The ABI (`include/softgpu/sg_ioctl.h`)

`sg_cmd` is a 64-byte, cache-line-sized command. `SG_IOC_SUBMIT` takes one
command plus an optional user host pointer. This header is plain C and is
the one thing meant to survive every stage, including the kernel module.

### The device (`src/device`)

A register file with a single-slot **mailbox**, a **doorbell** the driver
increments to submit, and a **completed** counter the device increments to
retire. One engine thread spins on the doorbell. Execution is real
(`memcpy`, loops), and time is accounted on a modeled 1 GHz clock:
`busy_cycles` while executing, `idle_cycles` while waiting. Idle is 100%
wasted power in the baseline — there is no idle state to enter.

Memory ordering: the doorbell store is `release` and the device's load is
`acquire`, so the mailbox contents are visible; the same pair on `completed`
publishes results back. On the ARM64 VM these are load-bearing.

### The driver (`src/driver`)

* **One mutex** around everything: allocation, freeing, submission.
* **One command in flight**: write mailbox, ring doorbell, busy-poll until
  the ticket retires. No interrupts, no queue.
* **Pageable copies**: H2D/D2H bounce through one 4 MiB staging buffer,
  one device command per chunk, fully serialized. Every byte is copied
  twice and the device idles while the host memcpy runs (and vice versa).
* **First-fit free-list allocator** with 256 B granularity and coalescing.
* **Full validation**: every address is checked against live allocations in
  the driver *and* bounds-checked in the device.

### The runtime (`src/runtime`)

Thin. Owns a descriptor, builds `sg_cmd`s, maps `-errno` to `sgError_t`.
`sgDeviceSynchronize()` exists but is a no-op because nothing is async yet.

## What we measure (`bench/`)

| benchmark | question it answers |
|-----------|--------------------|
| `submit`  | What does one round trip to the device cost? (p50/p99, CPU ns per op) |
| `memcpy`  | What does the data path cost per byte, and where are the cliffs? |
| `vadd`    | In a small real workload, what fraction is copies vs compute? |
| `gemm`    | When work is compute-bound, does the driver stop mattering? |
| `mt`      | How does throughput scale with concurrent submitters under one lock? |

Every row also reports `cpu_ns_per_op` (host CPU burned — our power proxy),
`dev_util` (device busy / (busy + idle)) and `dev_cmds_per_op` (how many
device commands one API call turned into).

## Expected weaknesses (hypotheses to confirm with numbers)

1. Submit latency is dominated by two cross-core cache-line handoffs
   (doorbell, completed) plus the mutex — several hundred ns to low µs, and
   the CPU cost per op is ~100% of wall time because we spin.
2. `memcpy` bandwidth is capped by the serialized memcpy→DMA→memcpy chain;
   large copies will show a visible cost per 4 MiB chunk.
3. `vadd` at small `n` is almost entirely submission overhead; at large `n`
   it is dominated by copies, not the kernel.
4. `mt` throughput does not scale — it may go *down* with more threads due
   to lock convoying.
5. Device utilization is low across the board except in `gemm 512`.

The measured results and the confirmation/refutation of each hypothesis are
recorded in `docs/decisions/000-baseline-results.md` once the first run on
the VM is in.
