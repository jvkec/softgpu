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
struct alignas(64) Registers {
    // ---- driver -> device -------------------------------------------------
    // BASELINE: a single command slot. The driver writes the command, then
    // rings the doorbell. Exactly one command may be in flight.
    sg_cmd mailbox{};
    alignas(64) std::atomic<uint32_t> doorbell{0}; // written by driver: submit ticket

    // ---- device -> driver -------------------------------------------------
    alignas(64) std::atomic<uint32_t> completed{0};  // last retired ticket
    std::atomic<int32_t> last_error{0};              // 0 or -errno of last command

    // ---- telemetry --------------------------------------------------------
    alignas(64) std::atomic<uint64_t> busy_cycles{0};
    std::atomic<uint64_t> idle_cycles{0};
    std::atomic<uint64_t> cmds_executed{0};
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
    void power_on();
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
