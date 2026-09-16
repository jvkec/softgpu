#pragma once
#include <chrono>
#include <cstdint>

namespace softgpu {

// The device's notion of time. We model a 1 GHz device clock, so one cycle is
// one nanosecond of host wall time. Everything reported as "cycles" in stats
// is derived from this clock, which makes busy/idle accounting comparable
// across stages regardless of how the driver waits.
using DeviceClock = std::chrono::steady_clock;

inline uint64_t now_cycles() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            DeviceClock::now().time_since_epoch()).count());
}

} // namespace softgpu
