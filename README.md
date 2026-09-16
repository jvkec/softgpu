# softgpu

A GPU driver stack in miniature, built to be measured.

`softgpu` is a software-modeled accelerator — VRAM, a register file, a command
engine — plus the full software stack that sits on top of real GPUs: a
kernel-mode driver, a user-mode runtime with a CUDA-shaped API, tests and
microbenchmarks. No hardware is involved, but every architectural decision
mirrors a real one, and every decision is made the same way: **measure the
baseline, identify the bottleneck, change the design, measure again.**

The project is a sequence of stages. Each stage lives in
[`docs/decisions/`](docs/decisions/) as an architecture decision record with
before/after numbers from [`results/`](results/).

| Stage | Topic | Status |
|------:|-------|--------|
| 0 | [Baseline](docs/00-baseline.md): mailbox submission, big lock, staging copies, busy-poll | [done](docs/decisions/000-baseline-results.md) |
| 1 | Command ring + doorbell batching | |
| 2 | Fences and interrupts vs polling; hybrid wait policy | |
| 3 | Pinned host memory, DMA engine, copy/compute overlap (streams) | |
| 4 | Multiple channels, lock-free submission, racing with TSan | |
| 5 | Device virtual memory: page tables, demand paging (`userfaultfd`), prefetch | |
| 6 | Context scheduling: time slicing, priorities, preemption | |
| 7 | Power management: idle gating, interrupt coalescing, energy-vs-latency | |
| 8 | Linux kernel module driver (`/dev/softgpu`, `ioctl`, `mmap`) | |

## Layout

```
include/softgpu/sg_ioctl.h    driver <-> runtime ABI (plain C, shared with the kernel module)
include/softgpu/sg_runtime.h  public runtime API
src/device/                   the "hardware": VRAM, registers, execution engine
src/driver/                   kernel-mode driver stand-in: allocation, validation, submission
src/runtime/                  user-mode runtime implementing sg_runtime.h
bench/                        sgbench: submit / memcpy / vadd / gemm / mt
tests/                        sgtest: correctness against CPU references
docs/                         architecture notes and decision records
results/                      committed benchmark output, one JSON per tag
scripts/vm.sh                 build / test / bench on the Linux VM
scripts/report.py             render / diff results JSON as markdown tables
```

Layering is enforced by the build: `sg_runtime` links only `sg_driver`, which
links only `sg_device`. The runtime cannot reach the device except through the
ioctl-shaped interface in `src/driver/sg_driver.h`.

## Build

Requires a C++17 compiler, CMake ≥ 3.22 and Ninja. Linux is the target;
macOS builds for development.

```sh
cmake --preset release && cmake --build --preset release
./build/release/sgtest
./build/release/sgbench all --json results/baseline.json
```

Sanitizer builds: `cmake --preset tsan` / `cmake --preset asan` (these use clang; GCC's libtsan fails under ASLR on aarch64).

On the VM (see `scripts/vm.sh`):

```sh
scripts/vm.sh test
scripts/vm.sh bench baseline all
```

## API sketch

```c
sgInit();
sgDevPtr a, b, c;
sgMalloc(&a, n * 4); sgMalloc(&b, n * 4); sgMalloc(&c, n * 4);
sgMemcpyH2D(a, ha, n * 4);
sgMemcpyH2D(b, hb, n * 4);
sgVaddF32(c, a, b, n);
sgMemcpyD2H(hc, c, n * 4);
sgDeviceSynchronize();
sgShutdown();
```
