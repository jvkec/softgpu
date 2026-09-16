#pragma once
// The "hardware": a software model of a simple accelerator.
//
// It owns a region of VRAM and a register file, and runs one execution thread
// that pulls commands out of the register file and executes them. Nothing
// above this layer touches VRAM or registers except through the paths real
// hardware would expose: MMIO registers and DMA to host addresses the driver
// hands it.

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include "softgpu/sg_ioctl.h"

namespace softgpu::device {

// Memory-mapped register file ("BAR0"). Each group lives on its own cache
// line so the producer and consumer sides do not false-share.
//
// STAGE 1: commands live in a ring in system memory that the driver
// allocates and programs into `ring_base`/`ring_mask`. The driver advances
// `put` (the doorbell: "commands up to here are valid"); the device advances
// `get` as it retires them. `get` therefore doubles as the fence register:
// command number N has retired once get >= N + 1. This is the GPFIFO
// PUT/GET scheme real GPUs use.
struct alignas(64) Registers {
    // ---- configuration: written by the driver before power_on() ------------
    uint64_t ring_base = 0; // address of sg_cmd[ring_mask + 1]
    uint32_t ring_mask = 0; // depth - 1; depth is a power of two

    // ---- driver -> device -------------------------------------------------
    alignas(64) std::atomic<uint64_t> put{0};

    // ---- device -> driver -------------------------------------------------
    alignas(64) std::atomic<uint64_t> get{0};
    std::atomic<int32_t> sticky_error{0}; // first -errno since reset, or 0

    // ---- telemetry --------------------------------------------------------
    alignas(64) std::atomic<uint64_t> busy_cycles{0};
    std::atomic<uint64_t> idle_cycles{0};
    std::atomic<uint64_t> cmds_executed{0};
    std::atomic<uint64_t> batches{0}; // idle -> busy transitions
    std::atomic<uint32_t> stats_gen{0}; // bumped by reset_stats(); engine restarts its idle timer
};

class Device {
public:
    explicit Device(uint64_t vram_bytes);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Registers& regs() { return regs_; }
    uint64_t vram_size() const { return vram_size_; }

    // Bring the execution engine up / down. power_off() blocks until the
    // engine thread has exited; any command in flight is completed first.
    // `cpu` >= 0 pins the engine thread to that CPU (Linux only); the
    // benchmark harness uses it to make cross-core placement reproducible.
    void power_on(int cpu = -1);
    void power_off();

    void reset_stats();

private:
    void run();
    int execute(const sg_cmd& cmd);
    bool vram_range_ok(uint64_t off, uint64_t len) const;

    Registers regs_;
    std::unique_ptr<uint8_t[]> vram_;
    uint64_t vram_size_;
    std::atomic<bool> running_{false};
    std::thread engine_;
};

} // namespace softgpu::device
