#!/usr/bin/env bash
# Build, test and benchmark on the Linux VM from the Mac.
#
# The repo is shared into the VM at $SG_VM_SRC; build trees live on the VM's
# own disk ($SG_VM_BUILD) so artifacts never touch the shared mount or git.
# Results are copied back into results/ so they can be committed.
#
#   scripts/vm.sh build  [preset]        # configure + build (default: release)
#   scripts/vm.sh test   [preset]        # build + ctest
#   scripts/vm.sh bench  <tag> [args..]  # build release, run sgbench, save results/<tag>.json
#                                        #   SG_ENV="SG_RING_DEPTH=16" to set env vars for the run
#                                        #   SG_RUNS=5 separate processes are merged (best pass per row)
#                                        #   SG_FORCE=1 to bench even on a loaded host
#   scripts/vm.sh canary                 # print the round-trip canary (ns); ~210 / ~590 on a quiet host
#   scripts/vm.sh shell                  # ssh into the VM at the source dir
set -euo pipefail

VM=${SG_VM:-softgpu-vm}
SRC=${SG_VM_SRC:-/mnt/softgpu}
BUILD=${SG_VM_BUILD:-\$HOME/build/softgpu}
HERE=$(cd "$(dirname "$0")/.." && pwd)

remote() { ssh -o LogLevel=error "$VM" "$@"; }

preset_flags() {
  case "$1" in
    release) echo "-DCMAKE_BUILD_TYPE=Release" ;;
    debug)   echo "-DCMAKE_BUILD_TYPE=Debug" ;;
    tsan)    echo "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DSG_SANITIZE=thread -DCMAKE_CXX_COMPILER=clang++" ;;
    asan)    echo "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DSG_SANITIZE=address,undefined -DCMAKE_CXX_COMPILER=clang++" ;;
    *) echo "unknown preset: $1" >&2; exit 1 ;;
  esac
}

configure_and_build() {
  local preset=$1
  # Mirrors CMakePresets.json, but with the build dir on VM-local disk.
  remote "set -e; cd $SRC && \
    cmake -S . -B $BUILD/$preset -G Ninja $(preset_flags "$preset") >/dev/null && \
    cmake --build $BUILD/$preset --parallel"
}

# The VM's vCPUs are ordinary host threads: a busy Mac makes every number
# worse, and macOS load average does not predict it well. The guard is a
# canary: one device round trip. It is bimodal (~210 ns when the two threads
# share a CPU cluster, ~590 ns across clusters), so the limit is set above
# both modes and only catches genuinely heavy load; placement noise is
# handled by sgbench --repeat.
host_load() { sysctl -n vm.loadavg | awk '{print $2}'; }

canary_ns() {
  remote "cd $BUILD/release && ./sgbench submit --quick 2>/dev/null" \
    | awk '/fill_64B_sync/ { for (i = 1; i <= NF; i++) if ($i ~ /^p50_ns=/) { sub("p50_ns=", "", $i); print $i } }'
}

check_host_quiet() {
  local ns limit=${SG_CANARY_NS:-800}
  ns=$(canary_ns)
  if [ -z "${SG_FORCE:-}" ] && awk -v v="$ns" -v l="$limit" 'BEGIN{exit !(v+0 > l+0)}'; then
    echo "canary round trip is ${ns} ns (limit ${limit}); host is busy, numbers would be noise." >&2
    echo "quiet the host, or SG_FORCE=1 to override." >&2
    exit 2
  fi
  echo "canary round trip ${ns} ns, host load $(host_load): ok"
}

# Merge several runs of the same selection into one results file, keeping,
# per row, every metric from the run with the lowest wall_ms (see the note in
# bench/main.cpp on bimodal placement), and attach provenance.
merge() {
  python3 - "$@" <<'PYEOF'
import json, sys, datetime
out, load, env, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
ds = [json.load(open(p)) for p in runs]
best = ds[0]
for d in ds[1:]:
    for i, row in enumerate(d["rows"]):
        if row["wall_ms"] < best["rows"][i]["wall_ms"]:
            best["rows"][i] = row
best["meta"] = {"host_loadavg_1m": float(load), "env": env, "runs": len(runs),
                "date": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(best, open(out, "w"), indent=1)
PYEOF
}

cmd=${1:-build}; shift || true
case "$cmd" in
  build) configure_and_build "${1:-release}" ;;
  test)  p=${1:-release}; configure_and_build "$p"; remote "cd $BUILD/$p && ctest --output-on-failure" ;;
  bench)
    tag=${1:?usage: vm.sh bench <tag> [sgbench args]}; shift || true
    configure_and_build release
    check_host_quiet
    mkdir -p "$HERE/results"
    runs=${SG_RUNS:-5}
    tmp=$(mktemp -d)
    for k in $(seq 1 "$runs"); do
      remote "cd $SRC && env ${SG_ENV:-} $BUILD/release/sgbench $* --tag $tag --json /tmp/sg-$tag-$k.json >/dev/null 2>&1; sleep 2"
      scp -q "$VM:/tmp/sg-$tag-$k.json" "$tmp/$k.json"
    done
    merge "$HERE/results/$tag.json" "$(host_load)" "${SG_ENV:-}" "$tmp"/*.json
    rm -rf "$tmp"
    echo "saved results/$tag.json ($runs runs, host load $(host_load))"
    ;;
  canary) configure_and_build release >/dev/null; echo "$(canary_ns) ns" ;;
  shell) ssh -t "$VM" "cd $SRC && exec \$SHELL -l" ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
