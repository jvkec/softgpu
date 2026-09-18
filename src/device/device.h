#pragma once
// The "hardware": a software model of a simple accelerator.
//
// It owns a region of VRAM and a register file, and runs one execution thread
// per engine that pulls commands out of that engine's ring and executes them.
// Nothing above this layer touches VRAM or registers except through the paths
// real hardware would expose: MMIO registers and DMA to host addresses the
// driver hands it.
//
// STAGE 3b: several engines, each serving several channels. Engine 0 is
// the compute engine (FILL, VADD, GEMM); engines 1.. are copy engines
// (COPY_*). A channel is a ring with its own PUT/GET and fence, executed in
// order. An engine round-robins its channels (a "runlist") and skips a
// channel whose head is a SG_OP_WAIT_FENCE that is not yet satisfied — the
// semaphore acquire blocks that channel, not the engine.

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "softgpu/sg_ioctl.h"

namespace softgpu::device {

// One channel's memory-mapped registers. Each group lives on its own cache
// line so the producer and consumer sides do not false-share.
//
// Commands live in a ring in system memory that the driver allocates and
// programs into `ring_base`/`ring_mask`. The driver advances `put` (the
// doorbell: "commands up to here are valid"); the engine advances `get` as
// it retires them. `get` therefore doubles as the fence register: command
// number N has retired once get >= N + 1. This is the GPFIFO PUT/GET scheme
// real GPUs use.
struct alignas(64) Channel {
    // ---- configuration: written by the driver before power_on() ------------
    uint64_t ring_base = 0; // address of sg_cmd[ring_mask + 1]
    uint32_t ring_mask = 0; // depth - 1; depth is a power of two

    // ---- driver -> engine -------------------------------------------------
    alignas(64) std::atomic<uint64_t> put{0};

    // ---- engine -> driver -------------------------------------------------
    alignas(64) std::atomic<uint64_t> get{0};
    std::atomic<int32_t> sticky_error{0}; // first -errno since reset, or 0
};

// One engine's telemetry (the engine thread accounts across its channels).
struct alignas(64) EngineStats {
    std::atomic<uint64_t> busy_cycles{0}; // executing commands
    std::atomic<uint64_t> wait_cycles{0}; // work pending but every channel head blocked on a WAIT
    std::atomic<uint64_t> idle_cycles{0}; // nothing queued on any channel
    std::atomic<uint64_t> cmds_executed{0};
    std::atomic<uint64_t> batches{0};   // idle -> busy transitions
    std::atomic<uint32_t> stats_gen{0}; // bumped by reset_stats(); engine restarts its idle timer
};

class Device {
public:
    // One compute engine plus `copy_engines` copy engines, `channels` rings each.
    Device(uint64_t vram_bytes, uint32_t copy_engines, uint32_t channels);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    uint32_t num_engines() const { return num_engines_; }
    uint32_t num_channels() const { return num_channels_; }
    Channel& channel(uint32_t engine, uint32_t ch) { return ch_[engine][ch]; }
    EngineStats& stats(uint32_t engine) { return stats_[engine]; }
    uint64_t vram_size() const { return vram_size_; }

    // Bring the engines up / down. power_off() blocks until every engine
    // thread has exited; commands in flight are completed first. `cpu` >= 0
    // pins the compute engine's thread to that CPU (Linux only).
    void power_on(int cpu = -1);
    void power_off();

    void reset_stats();

private:
    void run(uint32_t engine);
    int execute(uint32_t engine, const sg_cmd& cmd);
    bool vram_range_ok(uint64_t off, uint64_t len) const;

    Channel ch_[SG_MAX_ENGINES][SG_MAX_CHANNELS];
    EngineStats stats_[SG_MAX_ENGINES];
    uint32_t num_engines_;
    uint32_t num_channels_;
    std::unique_ptr<uint8_t[]> vram_;
    uint64_t vram_size_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> engines_;
};

} // namespace softgpu::device
