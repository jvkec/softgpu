// The five baseline microbenchmarks. Each one is designed to expose one
// dimension of the driver stack:
//
//   submit  - fixed cost of getting one command to the device and back
//   memcpy  - data-movement path (staging, chunking, copies per byte)
//   vadd    - a realistic tiny workload: copy in, compute, copy out
//   gemm    - compute-bound work, to see when submission cost stops mattering
//   mt      - scaling with concurrent submitters (the big lock)

#include <atomic>
#include <thread>
#include <vector>

#include "bench.h"

namespace bench {

namespace {

sgDevPtr must_malloc(size_t bytes) {
    sgDevPtr p = 0;
    die_on(sgMalloc(&p, bytes), "sgMalloc");
    return p;
}

void latency_row(Row& r, std::vector<double>& samples) {
    r.metrics["p50_ns"] = percentile(samples, 50);
    r.metrics["p99_ns"] = percentile(samples, 99);
    r.metrics["max_ns"] = percentile(samples, 100);
}

} // namespace

void bench_submit(const Options& o, Report& rep) {
    const int iters = o.quick ? 5000 : 50000;
    sgDevPtr d = must_malloc(4096);

    struct Case { const char* name; sgError_t (*fn)(sgDevPtr); } cases[] = {
        {"nop", [](sgDevPtr) { return sgDeviceSynchronize(); }}, // API-only floor
        {"fill_64B", [](sgDevPtr p) { return sgMemset(p, 0, 64); }},
        {"fill_4KiB", [](sgDevPtr p) { return sgMemset(p, 0, 4096); }},
    };
    for (auto& c : cases) {
        std::vector<double> lat;
        lat.reserve(iters);
        for (int i = 0; i < 200; ++i) c.fn(d); // warm up
        Window w;
        w.begin();
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            die_on(c.fn(d), c.name);
            lat.push_back(ns_since(t0));
        }
        w.end();
        Row r{"submit", c.name, {}};
        latency_row(r, lat);
        r.metrics["ops_per_s"] = iters / (w.wall_ns() / 1e9);
        w.fill(r, iters);
        rep.rows.push_back(r);
    }
    sgFree(d);
}

void bench_memcpy(const Options& o, Report& rep) {
    const size_t sizes[] = {4u << 10, 64u << 10, 1u << 20, 4u << 20, 16u << 20, 64u << 20};
    for (size_t bytes : sizes) {
        // Keep total traffic per case roughly constant so runs take similar time.
        const size_t budget = (o.quick ? 256u : 1024u) << 20;
        const int iters = std::max<int>(3, int(budget / bytes));
        std::vector<uint8_t> host(bytes, 0x5a);
        sgDevPtr d = must_malloc(bytes);

        for (int dir = 0; dir < 2; ++dir) {
            auto op = [&] {
                return dir == 0 ? sgMemcpyH2D(d, host.data(), bytes)
                                : sgMemcpyD2H(host.data(), d, bytes);
            };
            op();
            Window w;
            w.begin();
            for (int i = 0; i < iters; ++i) die_on(op(), "memcpy");
            w.end();
            char name[64];
            std::snprintf(name, sizeof name, "%s_%zuKiB", dir == 0 ? "h2d" : "d2h", bytes >> 10);
            Row r{"memcpy", name, {}};
            r.metrics["GBps"] = double(bytes) * iters / w.wall_ns();
            r.metrics["ns_per_op"] = w.wall_ns() / iters;
            w.fill(r, iters);
            rep.rows.push_back(r);
        }
        sgFree(d);
    }
}

void bench_vadd(const Options& o, Report& rep) {
    const uint32_t ns[] = {1u << 10, 1u << 16, 1u << 20, 1u << 24};
    for (uint32_t n : ns) {
        const size_t bytes = size_t(n) * 4;
        const int iters = o.quick ? 3 : std::max<int>(5, int((256u << 20) / bytes));
        std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n);
        sgDevPtr da = must_malloc(bytes), db = must_malloc(bytes), dc = must_malloc(bytes);

        double t_in = 0, t_k = 0, t_out = 0;
        Window w;
        w.begin();
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            die_on(sgMemcpyH2D(da, a.data(), bytes), "h2d");
            die_on(sgMemcpyH2D(db, b.data(), bytes), "h2d");
            auto t1 = Clock::now();
            die_on(sgVaddF32(dc, da, db, n), "vadd");
            auto t2 = Clock::now();
            die_on(sgMemcpyD2H(c.data(), dc, bytes), "d2h");
            auto t3 = Clock::now();
            t_in += std::chrono::duration<double, std::nano>(t1 - t0).count();
            t_k += std::chrono::duration<double, std::nano>(t2 - t1).count();
            t_out += std::chrono::duration<double, std::nano>(t3 - t2).count();
        }
        w.end();
        char name[32];
        std::snprintf(name, sizeof name, "n=%u", n);
        Row r{"vadd", name, {}};
        r.metrics["us_per_iter"] = w.wall_ns() / iters / 1e3;
        r.metrics["frac_h2d"] = t_in / (t_in + t_k + t_out);
        r.metrics["frac_kernel"] = t_k / (t_in + t_k + t_out);
        r.metrics["frac_d2h"] = t_out / (t_in + t_k + t_out);
        w.fill(r, iters);
        rep.rows.push_back(r);
        sgFree(da); sgFree(db); sgFree(dc);
    }
}

void bench_gemm(const Options& o, Report& rep) {
    const uint32_t dims[] = {32, 128, 512};
    for (uint32_t d : dims) {
        const size_t bytes = size_t(d) * d * 4;
        const int iters = o.quick ? 3 : (d >= 512 ? 5 : 200);
        sgDevPtr da = must_malloc(bytes), db = must_malloc(bytes), dc = must_malloc(bytes);
        die_on(sgMemset(da, 0, bytes), "memset");
        die_on(sgMemset(db, 0, bytes), "memset");
        sgGemmF32(dc, da, db, d, d, d);
        Window w;
        w.begin();
        for (int i = 0; i < iters; ++i) die_on(sgGemmF32(dc, da, db, d, d, d), "gemm");
        w.end();
        char name[32];
        std::snprintf(name, sizeof name, "%ux%ux%u", d, d, d);
        Row r{"gemm", name, {}};
        r.metrics["GFLOPS"] = 2.0 * d * d * d * iters / w.wall_ns();
        r.metrics["us_per_op"] = w.wall_ns() / iters / 1e3;
        w.fill(r, iters);
        rep.rows.push_back(r);
        sgFree(da); sgFree(db); sgFree(dc);
    }
}

void bench_mt(const Options& o, Report& rep) {
    const int per_thread = o.quick ? 2000 : 20000;
    for (int T = 1; T <= o.threads_max; T *= 2) {
        std::vector<sgDevPtr> bufs(T);
        for (auto& b : bufs) b = must_malloc(4096);

        std::atomic<int> go{0};
        std::vector<std::thread> ts;
        std::vector<std::vector<double>> lats(T);
        std::vector<double> cpu(T, 0.0);
        Window w;
        w.begin();
        for (int t = 0; t < T; ++t)
            ts.emplace_back([&, t] {
                lats[t].reserve(per_thread);
                while (!go.load(std::memory_order_acquire)) {}
                const double c0 = thread_cpu_ns();
                for (int i = 0; i < per_thread; ++i) {
                    auto t0 = Clock::now();
                    die_on(sgMemset(bufs[t], i, 4096), "memset");
                    lats[t].push_back(ns_since(t0));
                }
                cpu[t] = thread_cpu_ns() - c0;
            });
        go.store(1, std::memory_order_release);
        for (auto& th : ts) th.join();
        w.end();
        w.cpu_override = 0;
        for (double c : cpu) w.cpu_override += c;

        std::vector<double> all;
        for (auto& l : lats) all.insert(all.end(), l.begin(), l.end());
        char name[32];
        std::snprintf(name, sizeof name, "threads=%d", T);
        Row r{"mt", name, {}};
        r.metrics["ops_per_s"] = double(T) * per_thread / (w.wall_ns() / 1e9);
        latency_row(r, all);
        w.fill(r, double(T) * per_thread);
        rep.rows.push_back(r);
        for (auto b : bufs) sgFree(b);
    }
}

} // namespace bench
