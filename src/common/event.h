#pragma once
// A wakeable event: the device's interrupt line, and the primitive every
// blocking wait in the driver sleeps on.
//
// Linux: a 32-bit sequence word plus futex(2) — the same primitive under
// pthread condition variables and what the stage-8 kernel module maps to
// wait_event()/wake_up(). Waiters read `seq`, check their predicate, then
// FUTEX_WAIT(seq_seen): the kernel compares the word atomically, so a signal
// between the check and the sleep cannot be lost. Elsewhere: a condition
// variable with the same contract.

#include <atomic>
#include <chrono>
#include <cstdint>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <climits>
#else
#include <condition_variable>
#include <mutex>
#endif

namespace softgpu {

class Event {
public:
    uint32_t seq() const { return seq_.load(std::memory_order_acquire); }

    // Publish "something happened" and wake every waiter.
    void signal() {
#if defined(__linux__)
        seq_.fetch_add(1, std::memory_order_release);
        syscall(SYS_futex, &seq_, FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#else
        {
            std::lock_guard<std::mutex> g(m_);
            seq_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
#endif
    }

    // Sleep until seq() != seen or timeout_ns elapses. May return spuriously;
    // callers re-check their predicate.
    void wait(uint32_t seen, uint64_t timeout_ns) {
#if defined(__linux__)
        timespec ts{time_t(timeout_ns / 1000000000ull), long(timeout_ns % 1000000000ull)};
        syscall(SYS_futex, &seq_, FUTEX_WAIT_PRIVATE, seen, &ts, nullptr, 0);
#else
        std::unique_lock<std::mutex> l(m_);
        cv_.wait_for(l, std::chrono::nanoseconds(timeout_ns),
                     [&] { return seq_.load(std::memory_order_acquire) != seen; });
#endif
    }

private:
    std::atomic<uint32_t> seq_{0};
#if !defined(__linux__)
    std::mutex m_;
    std::condition_variable cv_;
#endif
};

} // namespace softgpu
