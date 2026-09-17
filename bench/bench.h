#pragma once
// Shared benchmark plumbing: timing, percentiles, CPU accounting, JSON out.

#include <time.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "softgpu/sg_runtime.h"

namespace bench {

using Clock = std::chrono::steady_clock;

inline double ns_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

// user+sys CPU time of the *calling thread* in ns. This is our proxy for
// host-side power: a driver that spins burns CPU even when the device is the
// bottleneck. Per-thread on purpose — process-wide accounting would include
// the device engine's spin loop, which is the simulated hardware, not the
// driver (its cost is modeled separately as device idle cycles).
inline double thread_cpu_ns() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1e9 + double(ts.tv_nsec);
}

inline double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t i = static_cast<size_t>(p / 100.0 * double(v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

// One row of results. `metrics` is deliberately a flat map so the JSON writer
// and the table printer stay trivial.
struct Row {
    std::string bench;
    std::string variant;
    std::map<std::string, double> metrics;
};

struct Report {
    std::vector<Row> rows;

    void print() const {
        for (const auto& r : rows) {
            std::printf("%-10s %-22s", r.bench.c_str(), r.variant.c_str());
            for (const auto& [k, v] : r.metrics) std::printf("  %s=%.3g", k.c_str(), v);
            std::printf("\n");
        }
    }

    bool write_json(const std::string& path, const std::string& tag) const {
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return false;
        std::fprintf(f, "{\n  \"tag\": \"%s\",\n  \"rows\": [\n", tag.c_str());
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            std::fprintf(f, "    {\"bench\": \"%s\", \"variant\": \"%s\"", r.bench.c_str(),
                         r.variant.c_str());
            for (const auto& [k, v] : r.metrics) std::fprintf(f, ", \"%s\": %.6g", k.c_str(), v);
            std::fprintf(f, "}%s\n", i + 1 < rows.size() ? "," : "");
        }
        std::fprintf(f, "  ]\n}\n");
        std::fclose(f);
        return true;
    }
};

// Snapshot of runtime stats + submitter CPU around a measured region.
// Single-threaded benchmarks get CPU from the calling thread; multi-threaded
// ones sum their workers' thread CPU into `cpu_override`.
struct Window {
    sgStats_t s0{}, s1{};
    double cpu0 = 0, cpu1 = 0;
    double cpu_override = -1;
    Clock::time_point t0, t1;

    void begin() {
        sgDeviceSynchronize(); // warm-up work must not leak into the window
        sgResetStats();
        sgGetStats(&s0);
        cpu0 = thread_cpu_ns();
        t0 = Clock::now();
    }
    void end() {
        t1 = Clock::now();
        cpu1 = thread_cpu_ns();
        sgGetStats(&s1);
    }
    double wall_ns() const { return std::chrono::duration<double, std::nano>(t1 - t0).count(); }
    double cpu_ns() const { return cpu_override >= 0 ? cpu_override : cpu1 - cpu0; }
    double dev_busy() const { return double(s1.device_busy_cycles - s0.device_busy_cycles); }
    double dev_idle() const { return double(s1.device_idle_cycles - s0.device_idle_cycles); }
    double dev_cmds() const { return double(s1.device_cmds_executed - s0.device_cmds_executed); }
    double dev_batches() const { return double(s1.device_batches - s0.device_batches); }
    double waits() const { return double(s1.driver_waits - s0.driver_waits); }
    double stalls() const { return double(s1.driver_stalls - s0.driver_stalls); }
    double staging_waits() const { return double(s1.staging_waits - s0.staging_waits); }
    double bytes_direct() const { return double(s1.bytes_direct - s0.bytes_direct); }
    double bytes_staged() const { return double(s1.bytes_staged - s0.bytes_staged); }

    // Fill the standard metrics every benchmark reports.
    void fill(Row& r, double ops) const {
        r.metrics["wall_ms"] = wall_ns() / 1e6;
        r.metrics["cpu_ns_per_op"] = cpu_ns() / ops;
        r.metrics["dev_util"] = dev_busy() / std::max(1.0, dev_busy() + dev_idle());
        r.metrics["dev_cmds_per_op"] = dev_cmds() / ops;
        r.metrics["waits_per_op"] = waits() / ops;
        r.metrics["stalls_per_op"] = stalls() / ops;
        r.metrics["staging_waits_per_op"] = staging_waits() / ops;
        if (bytes_direct() + bytes_staged() > 0)
            r.metrics["direct_frac"] = bytes_direct() / (bytes_direct() + bytes_staged());
        r.metrics["avg_batch"] = dev_cmds() / std::max(1.0, dev_batches());
    }
};

struct Options {
    bool quick = false;
    std::string json;
    std::string tag = "baseline";
    int threads_max = 8;
    int repeat = 1; // run the whole selection N times, report per-metric medians
};

void die_on(sgError_t e, const char* what);

void bench_submit(const Options&, Report&);
void bench_batch(const Options&, Report&);
void bench_memcpy(const Options&, Report&);
void bench_vadd(const Options&, Report&);
void bench_gemm(const Options&, Report&);
void bench_mt(const Options&, Report&);
void bench_alloc(const Options&, Report&);

} // namespace bench
