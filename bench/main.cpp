#include <cstdlib>
#include <cstring>
#include <string>

#include "bench.h"

namespace bench {
void die_on(sgError_t e, const char* what) {
    if (e != SG_OK) {
        std::fprintf(stderr, "%s: %s\n", what, sgErrorString(e));
        std::exit(2);
    }
}
} // namespace bench

static void usage() {
    std::fprintf(stderr,
                 "usage: sgbench [submit|memcpy|vadd|gemm|mt|all]... [--quick] [--json FILE] "
                 "[--tag NAME] [--threads N]\n");
    std::exit(1);
}

int main(int argc, char** argv) {
    bench::Options opt;
    std::vector<std::string> which;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick") opt.quick = true;
        else if (a == "--json" && i + 1 < argc) opt.json = argv[++i];
        else if (a == "--tag" && i + 1 < argc) opt.tag = argv[++i];
        else if (a == "--threads" && i + 1 < argc) opt.threads_max = std::atoi(argv[++i]);
        else if (a[0] == '-') usage();
        else which.push_back(a);
    }
    if (which.empty()) which.push_back("all");

    bench::die_on(sgInit(), "sgInit");
    bench::Report rep;
    for (const auto& w : which) {
        bool all = w == "all";
        if (all || w == "submit") bench::bench_submit(opt, rep);
        if (all || w == "memcpy") bench::bench_memcpy(opt, rep);
        if (all || w == "vadd") bench::bench_vadd(opt, rep);
        if (all || w == "gemm") bench::bench_gemm(opt, rep);
        if (all || w == "mt") bench::bench_mt(opt, rep);
        if (!all && w != "submit" && w != "memcpy" && w != "vadd" && w != "gemm" && w != "mt") usage();
    }
    sgShutdown();

    rep.print();
    if (!opt.json.empty()) {
        if (!rep.write_json(opt.json, opt.tag)) { std::perror("write_json"); return 3; }
        std::fprintf(stderr, "wrote %s\n", opt.json.c_str());
    }
    return 0;
}
