#!/usr/bin/env bash
# Build, test and benchmark on the Linux VM from the Mac.
#
# The repo is shared into the VM over virtiofs at $SG_VM_SRC; build trees live
# on the VM's own disk ($SG_VM_BUILD) so artifacts never touch the shared
# mount or git. Results are copied back into results/ so they can be committed.
#
#   scripts/vm.sh build  [preset]        # configure + build (default: release)
#   scripts/vm.sh test   [preset]        # build + ctest
#   scripts/vm.sh bench  <tag> [args..]  # build release, run sgbench, save results/<tag>.json
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

cmd=${1:-build}; shift || true
case "$cmd" in
  build) configure_and_build "${1:-release}" ;;
  test)  p=${1:-release}; configure_and_build "$p"; remote "cd $BUILD/$p && ctest --output-on-failure" ;;
  bench)
    tag=${1:?usage: vm.sh bench <tag> [sgbench args]}; shift || true
    configure_and_build release
    mkdir -p "$HERE/results"
    remote "cd $SRC && nproc && uname -srm && $BUILD/release/sgbench $* --tag $tag --json /tmp/sg-$tag.json"
    scp -q "$VM:/tmp/sg-$tag.json" "$HERE/results/$tag.json"
    echo "saved results/$tag.json"
    ;;
  shell) ssh -t "$VM" "cd $SRC && exec \$SHELL -l" ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
