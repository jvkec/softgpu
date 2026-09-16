#pragma once
// First-fit free-list allocator for device memory.
//
// BASELINE: simplest correct thing. No size classes, no per-context pools,
// O(#free blocks) allocation. It is protected by the driver's global lock.

#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>

namespace softgpu::driver {

class VramAllocator {
public:
    VramAllocator(uint64_t size, uint64_t align);

    std::optional<uint64_t> alloc(uint64_t bytes);
    bool free(uint64_t addr);

    // True if [addr, addr+len) lies entirely inside one live allocation.
    bool owns_range(uint64_t addr, uint64_t len) const;

    uint64_t bytes_live() const { return live_bytes_; }

private:
    uint64_t align_;
    std::map<uint64_t, uint64_t> free_;           // start -> length, coalesced
    std::unordered_map<uint64_t, uint64_t> live_; // start -> length
    uint64_t live_bytes_ = 0;
};

} // namespace softgpu::driver
