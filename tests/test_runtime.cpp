// Correctness tests for the runtime -> driver -> device path. No framework:
// a CHECK macro and a process exit code are all ctest needs.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "softgpu/sg_runtime.h"

static int g_failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++g_failures;                                                                \
        }                                                                                \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == SG_OK)

static void test_memcpy_roundtrip(size_t bytes) {
    std::vector<uint8_t> src(bytes), dst(bytes, 0);
    for (size_t i = 0; i < bytes; ++i) src[i] = static_cast<uint8_t>(i * 31 + 7);
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, bytes));
    CHECK_OK(sgMemcpyH2D(d, src.data(), bytes));
    CHECK_OK(sgMemcpyD2H(dst.data(), d, bytes));
    CHECK(src == dst);
    CHECK_OK(sgFree(d));
}

static void test_memset_and_d2d() {
    const size_t n = 1 << 16;
    sgDevPtr a = 0, b = 0;
    CHECK_OK(sgMalloc(&a, n));
    CHECK_OK(sgMalloc(&b, n));
    CHECK_OK(sgMemset(a, 0xAB, n));
    CHECK_OK(sgMemcpyD2D(b, a, n));
    std::vector<uint8_t> out(n);
    CHECK_OK(sgMemcpyD2H(out.data(), b, n));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 0xAB; }));
    CHECK_OK(sgFree(a));
    CHECK_OK(sgFree(b));
}

static void test_vadd() {
    const uint32_t n = 100003; // not a multiple of anything convenient
    std::vector<float> a(n), b(n), c(n);
    for (uint32_t i = 0; i < n; ++i) { a[i] = float(i) * 0.5f; b[i] = 1000.0f - float(i); }
    sgDevPtr da = 0, db = 0, dc = 0;
    CHECK_OK(sgMalloc(&da, n * 4));
    CHECK_OK(sgMalloc(&db, n * 4));
    CHECK_OK(sgMalloc(&dc, n * 4));
    CHECK_OK(sgMemcpyH2D(da, a.data(), n * 4));
    CHECK_OK(sgMemcpyH2D(db, b.data(), n * 4));
    CHECK_OK(sgVaddF32(dc, da, db, n));
    CHECK_OK(sgMemcpyD2H(c.data(), dc, n * 4));
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; ++i) ok = (c[i] == a[i] + b[i]);
    CHECK(ok);
    CHECK_OK(sgFree(da)); CHECK_OK(sgFree(db)); CHECK_OK(sgFree(dc));
}

static void test_gemm() {
    const uint32_t m = 37, n = 53, k = 29;
    std::vector<float> a(m * k), b(k * n), c(m * n), ref(m * n, 0.0f);
    for (size_t i = 0; i < a.size(); ++i) a[i] = float((i * 7) % 13) - 6.0f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = float((i * 5) % 11) - 5.0f;
    for (uint32_t i = 0; i < m; ++i)
        for (uint32_t j = 0; j < n; ++j) {
            float s = 0;
            for (uint32_t p = 0; p < k; ++p) s += a[i * k + p] * b[p * n + j];
            ref[i * n + j] = s;
        }
    sgDevPtr da = 0, db = 0, dc = 0;
    CHECK_OK(sgMalloc(&da, a.size() * 4));
    CHECK_OK(sgMalloc(&db, b.size() * 4));
    CHECK_OK(sgMalloc(&dc, c.size() * 4));
    CHECK_OK(sgMemcpyH2D(da, a.data(), a.size() * 4));
    CHECK_OK(sgMemcpyH2D(db, b.data(), b.size() * 4));
    CHECK_OK(sgGemmF32(dc, da, db, m, n, k));
    CHECK_OK(sgMemcpyD2H(c.data(), dc, c.size() * 4));
    bool ok = true;
    for (size_t i = 0; i < c.size() && ok; ++i) ok = std::fabs(c[i] - ref[i]) < 1e-3f;
    CHECK(ok);
    CHECK_OK(sgFree(da)); CHECK_OK(sgFree(db)); CHECK_OK(sgFree(dc));
}

static void test_validation() {
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, 4096));
    uint8_t buf[16] = {};
    // Past the end of the allocation.
    CHECK(sgMemset(d, 0, 4097) == SG_ERR_INVALID_ADDRESS);
    CHECK(sgMemcpyH2D(d + 4090, buf, 16) == SG_ERR_INVALID_ADDRESS);
    // Never allocated.
    CHECK(sgMemcpyD2H(buf, 100 << 20, 16) == SG_ERR_INVALID_ADDRESS);
    // Double free / bogus free.
    CHECK_OK(sgFree(d));
    CHECK(sgFree(d) == SG_ERR_INVALID_VALUE);
    CHECK(sgFree(12345) == SG_ERR_INVALID_VALUE);
    // Zero-size and null.
    CHECK(sgMalloc(&d, 0) == SG_ERR_INVALID_VALUE);
    CHECK(sgMalloc(nullptr, 16) == SG_ERR_INVALID_VALUE);
}

static void test_alloc_reuse_and_oom() {
    // Fill VRAM, free the middle, make sure coalescing gives it back.
    const size_t chunk = 64u << 20;
    sgDevPtr p[4] = {};
    for (auto& x : p) CHECK_OK(sgMalloc(&x, chunk));
    sgDevPtr extra = 0;
    CHECK(sgMalloc(&extra, 1) == SG_ERR_OUT_OF_MEMORY);
    CHECK_OK(sgFree(p[1]));
    CHECK_OK(sgFree(p[2]));
    CHECK_OK(sgMalloc(&extra, 2 * chunk)); // only satisfiable if coalesced
    CHECK_OK(sgFree(extra));
    CHECK_OK(sgFree(p[0]));
    CHECK_OK(sgFree(p[3]));
}

static void test_async_ordering() {
    // Thousands of async fills — far more than any ring depth — then one
    // sync. The device must apply them in order and the last one must win.
    const size_t n = 4096;
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, n));
    for (int i = 0; i < 20000; ++i) CHECK_OK(sgMemset(d, i & 0xff, n));
    CHECK_OK(sgDeviceSynchronize());
    std::vector<uint8_t> out(n);
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == (19999 & 0xff); }));

    // Data dependency through the queue without an explicit sync in between.
    sgDevPtr a = 0, b = 0, c = 0;
    const uint32_t m = 1 << 14;
    CHECK_OK(sgMalloc(&a, m * 4)); CHECK_OK(sgMalloc(&b, m * 4)); CHECK_OK(sgMalloc(&c, m * 4));
    std::vector<float> ha(m, 3.0f), hb(m, 4.0f), hc(m);
    CHECK_OK(sgMemcpyH2D(a, ha.data(), m * 4));
    CHECK_OK(sgMemcpyH2D(b, hb.data(), m * 4));
    CHECK_OK(sgVaddF32(c, a, b, m));
    CHECK_OK(sgMemcpyD2D(a, c, m * 4));   // reuse a as scratch: a = c
    CHECK_OK(sgVaddF32(c, a, b, m));      // c = (3+4)+4
    CHECK_OK(sgMemcpyD2H(hc.data(), c, m * 4));
    CHECK(std::all_of(hc.begin(), hc.end(), [](float v) { return v == 11.0f; }));

    // Sync with nothing outstanding is a no-op that does not block.
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgGetStats(&s1));
    CHECK(s1.driver_waits == s0.driver_waits);

    CHECK_OK(sgFree(d)); CHECK_OK(sgFree(a)); CHECK_OK(sgFree(b)); CHECK_OK(sgFree(c));
}

static void test_pinned_roundtrip(size_t bytes) {
    uint8_t *src = nullptr, *dst = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&src), bytes));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&dst), bytes));
    for (size_t i = 0; i < bytes; ++i) src[i] = static_cast<uint8_t>(i * 13 + 5);
    std::memset(dst, 0, bytes);
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, bytes));
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyH2D(d, src, bytes));
    CHECK_OK(sgMemcpyD2H(dst, d, bytes));
    CHECK_OK(sgGetStats(&s1));
    CHECK(std::memcmp(src, dst, bytes) == 0);
    CHECK(s1.bytes_direct - s0.bytes_direct == 2 * bytes); // both went direct
    CHECK(s1.bytes_staged == s0.bytes_staged);
    CHECK_OK(sgFree(d));
    CHECK_OK(sgFreeHost(src));
    CHECK_OK(sgFreeHost(dst));
}

static void test_host_register() {
    const size_t n = 1 << 20;
    std::vector<uint8_t> buf(n + 4096), out(n);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = uint8_t(i);
    sgDevPtr d = 0;
    CHECK_OK(sgMalloc(&d, n));
    CHECK_OK(sgHostRegister(buf.data(), n)); // pin only the first n bytes
    CHECK(sgHostRegister(buf.data() + 100, 10) == SG_ERR_INVALID_VALUE); // overlap

    sgStats_t s0{}, s1{}, s2{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyH2D(d, buf.data(), n)); // fully inside the pin: direct
    CHECK_OK(sgGetStats(&s1));
    CHECK(s1.bytes_direct - s0.bytes_direct == n);
    // Straddles the end of the pinned range: must fall back to staging and
    // still be correct.
    CHECK_OK(sgMemcpyH2D(d, buf.data() + 4096, n));
    CHECK_OK(sgGetStats(&s2));
    CHECK(s2.bytes_staged - s1.bytes_staged == n);
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::memcmp(out.data(), buf.data() + 4096, n) == 0);

    // Unregister while a DMA from the range is in flight, then reuse it.
    CHECK_OK(sgMemcpyH2DAsync(d, buf.data(), n, nullptr));
    CHECK_OK(sgHostUnregister(buf.data()));
    CHECK(sgHostUnregister(buf.data()) == SG_ERR_INVALID_VALUE);
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgMemcpyD2H(out.data(), d, n));
    CHECK(std::memcmp(out.data(), buf.data(), n) == 0);
    CHECK_OK(sgFree(d));
}

static void test_async_pinned_pipeline() {
    // H2D a, H2D b (async, pinned), VADD, D2H c (async, pinned), one sync.
    const uint32_t n = 1 << 18;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&a), n * 4));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&b), n * 4));
    CHECK_OK(sgMallocHost(reinterpret_cast<void**>(&c), n * 4));
    for (uint32_t i = 0; i < n; ++i) { a[i] = float(i); b[i] = 2.0f * float(i); c[i] = -1.0f; }
    sgDevPtr da = 0, db = 0, dc = 0;
    CHECK_OK(sgMalloc(&da, n * 4)); CHECK_OK(sgMalloc(&db, n * 4)); CHECK_OK(sgMalloc(&dc, n * 4));
    sgStats_t s0{}, s1{};
    CHECK_OK(sgGetStats(&s0));
    CHECK_OK(sgMemcpyH2DAsync(da, a, n * 4, nullptr));
    CHECK_OK(sgMemcpyH2DAsync(db, b, n * 4, nullptr));
    CHECK_OK(sgVaddF32(dc, da, db, n));
    CHECK_OK(sgMemcpyD2HAsync(c, dc, n * 4, nullptr));
    CHECK_OK(sgDeviceSynchronize());
    CHECK_OK(sgGetStats(&s1));
    bool ok = true;
    for (uint32_t i = 0; i < n && ok; ++i) ok = (c[i] == 3.0f * float(i));
    CHECK(ok);
    CHECK(s1.driver_waits - s0.driver_waits == 1); // only the final sync blocked
    // A non-null stream is rejected until streams exist.
    CHECK(sgMemcpyH2DAsync(da, a, 4, reinterpret_cast<sgStream_t>(1)) == SG_ERR_INVALID_VALUE);
    CHECK_OK(sgFree(da)); CHECK_OK(sgFree(db)); CHECK_OK(sgFree(dc));
    CHECK_OK(sgFreeHost(a)); CHECK_OK(sgFreeHost(b)); CHECK_OK(sgFreeHost(c));
}

static void test_deferred_free() {
    // A long GEMM is writing `c` when we free it and immediately reallocate.
    // The allocator must not hand the same memory out until the GEMM retired,
    // otherwise the GEMM's late writes would clobber our memset.
    const uint32_t d = 512;
    const size_t bytes = size_t(d) * d * 4;
    sgDevPtr a = 0, b = 0, c = 0;
    CHECK_OK(sgMalloc(&a, bytes)); CHECK_OK(sgMalloc(&b, bytes)); CHECK_OK(sgMalloc(&c, bytes));
    CHECK_OK(sgMemset(a, 0, bytes)); CHECK_OK(sgMemset(b, 0, bytes));
    CHECK_OK(sgGemmF32(c, a, b, d, d, d)); // ~14 ms, async
    CHECK_OK(sgFree(c));
    sgDevPtr c2 = 0;
    CHECK_OK(sgMalloc(&c2, bytes));
    // With first-fit and everything else live, c2 either reuses c's slot
    // (only safe if the driver deferred the free) or takes fresh space.
    CHECK_OK(sgMemset(c2, 0x7f, bytes));
    CHECK_OK(sgDeviceSynchronize());
    std::vector<uint8_t> out(bytes);
    CHECK_OK(sgMemcpyD2H(out.data(), c2, bytes));
    CHECK(std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 0x7f; }));
    // Now that the GEMM has retired, the original c has been reclaimed and,
    // being the lowest free block, is what first-fit hands out next.
    CHECK_OK(sgFree(c2));
    sgDevPtr c3 = 0;
    CHECK_OK(sgMalloc(&c3, bytes));
    CHECK(c3 == c);
    CHECK_OK(sgFree(a)); CHECK_OK(sgFree(b)); CHECK_OK(sgFree(c3));
}

static void test_concurrent_pinned_submitters() {
    const int T = 8, iters = 100;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            const size_t n = 1 << 16;
            uint8_t *src = nullptr, *dst = nullptr;
            if (sgMallocHost(reinterpret_cast<void**>(&src), n) != SG_OK ||
                sgMallocHost(reinterpret_cast<void**>(&dst), n) != SG_OK) { bad[t] = 1; return; }
            sgDevPtr d = 0;
            if (sgMalloc(&d, n) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters && !bad[t]; ++i) {
                std::memset(src, t * 16 + (i & 15), n);
                if (sgMemcpyH2D(d, src, n) != SG_OK || sgMemcpyD2H(dst, d, n) != SG_OK ||
                    std::memcmp(src, dst, n) != 0)
                    bad[t] = 1;
            }
            sgFree(d);
            sgFreeHost(src);
            sgFreeHost(dst);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
}

static void test_concurrent_submitters() {
    // Many threads hammering the driver; every thread verifies its own data.
    const int T = 8, iters = 200;
    std::vector<std::thread> ts;
    std::vector<int> bad(T, 0);
    for (int t = 0; t < T; ++t)
        ts.emplace_back([t, &bad] {
            const size_t n = 4096;
            std::vector<uint8_t> src(n, uint8_t(t + 1)), dst(n);
            sgDevPtr d = 0;
            if (sgMalloc(&d, n) != SG_OK) { bad[t] = 1; return; }
            for (int i = 0; i < iters; ++i) {
                src[i % n] = uint8_t(i);
                if (sgMemcpyH2D(d, src.data(), n) != SG_OK ||
                    sgMemcpyD2H(dst.data(), d, n) != SG_OK || src != dst) { bad[t] = 1; break; }
            }
            sgFree(d);
        });
    for (auto& th : ts) th.join();
    CHECK(std::accumulate(bad.begin(), bad.end(), 0) == 0);
}

int main() {
    CHECK(sgInit() == SG_OK);
    CHECK(sgInit() == SG_ERR_ALREADY_INITIALIZED);

    struct { const char* name; void (*fn)(); } tests[] = {
        {"memcpy_roundtrip_small", [] { test_memcpy_roundtrip(1); test_memcpy_roundtrip(4096); }},
        {"memcpy_roundtrip_multi_chunk", [] { test_memcpy_roundtrip((4u << 20) + 12345); }},
        {"memcpy_roundtrip_large", [] { test_memcpy_roundtrip(64u << 20); }},
        {"memset_and_d2d", test_memset_and_d2d},
        {"vadd", test_vadd},
        {"gemm", test_gemm},
        {"validation", test_validation},
        {"alloc_reuse_and_oom", test_alloc_reuse_and_oom},
        {"async_ordering", test_async_ordering},
        {"pinned_roundtrip", [] { test_pinned_roundtrip(1); test_pinned_roundtrip(4096); test_pinned_roundtrip(64u << 20); }},
        {"host_register", test_host_register},
        {"async_pinned_pipeline", test_async_pinned_pipeline},
        {"deferred_free", test_deferred_free},
        {"concurrent_pinned_submitters", test_concurrent_pinned_submitters},
        {"concurrent_submitters", test_concurrent_submitters},
    };
    for (auto& t : tests) {
        int before = g_failures;
        t.fn();
        std::printf("[%s] %s\n", g_failures == before ? " OK " : "FAIL", t.name);
    }

    CHECK(sgShutdown() == SG_OK);
    CHECK(sgShutdown() == SG_ERR_NOT_INITIALIZED);
    std::printf("%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
