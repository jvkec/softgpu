#pragma once
// Small architecture-specific helpers shared by the device model and driver.

namespace softgpu {

// Hint to the core that we are in a spin loop. Keeps the spinning thread from
// hogging pipeline resources shared with its SMT sibling; does not yield to
// the OS. Used where "hardware" would poll a register.
inline void cpu_relax() {
#if defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
    asm volatile("pause" ::: "memory");
#else
    asm volatile("" ::: "memory");
#endif
}

} // namespace softgpu
