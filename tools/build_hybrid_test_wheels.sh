#!/bin/bash
# Builds the mamba-ssm and causal-conv1d wheels that the hybrid-model tests
# compare against, from pinned commits, into $1. Run by docker/Dockerfile's
# hybrid-test-wheels stage; the torch to build against must already be installed.
# .buildkite/scripts/check-hybrid-test-deps.py checks the same commits.
set -euo pipefail

out_dir=$1
mamba_commit=f1493ff6e9335160eb134eb67e59f8e4d9adefd6 # v2.3.0
causal_conv1d_commit=da6dbaa9fd5a919967f14d3fd031da1288ad5025 # v1.6.0

# Fetch an exact commit (tried twice) and fail unless HEAD is that commit.
fetch_commit() {
    local repo=$1 commit=$2 dir=$3
    rm -rf "$dir" && git init -q "$dir"
    git -C "$dir" fetch -q --depth 1 "$repo" "$commit" \
        || { sleep 30; git -C "$dir" fetch -q --depth 1 "$repo" "$commit"; }
    git -C "$dir" checkout -q --detach FETCH_HEAD
    test "$(git -C "$dir" rev-parse HEAD)" = "$commit"
}

# Compile only sm_80 (A100, and L4's sm_89) and sm_90 (H100/H200): the only
# GPUs that run these tests. Upstream hard-codes 9 targets on CUDA 13.
only_ci_gpu_targets() {
    python3 - "$1" <<'EOF'
import sys

path = sys.argv[1]
source = open(path).read()
anchor = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
assert source.count(anchor) == 1, f"{path}: arch patch anchor not found"
patch = (
    '    cc_flag = [f for f in cc_flag if f != "-gencode" and not f.startswith("arch=")]\n'
    '    cc_flag += ["-gencode", "arch=compute_80,code=sm_80",'
    ' "-gencode", "arch=compute_90,code=sm_90"]\n'
)
open(path, "w").write(source.replace(anchor, patch + anchor))
EOF
}

mkdir -p "$out_dir"

fetch_commit https://github.com/state-spaces/mamba "$mamba_commit" /tmp/mamba-src
# torch>=2.14 nightly needs C++20 for ATen headers (pytorch/pytorch#178150);
# mamba v2.3.0 pins -std=c++17.
sed -i 's/-std=c++17/-std=c++20/g' /tmp/mamba-src/setup.py
only_ci_gpu_targets /tmp/mamba-src/setup.py
# FORCE_BUILD: without it, setup.py downloads an unpinned prebuilt wheel.
MAMBA_FORCE_BUILD=TRUE uv build --wheel --no-build-isolation \
    --python /opt/venv/bin/python3 --out-dir "$out_dir" /tmp/mamba-src

fetch_commit https://github.com/Dao-AILab/causal-conv1d "$causal_conv1d_commit" /tmp/causal-conv1d-src
only_ci_gpu_targets /tmp/causal-conv1d-src/setup.py
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv build --wheel --no-build-isolation \
    --python /opt/venv/bin/python3 --out-dir "$out_dir" /tmp/causal-conv1d-src

printf 'mamba_ssm %s\ncausal_conv1d %s\n' "$mamba_commit" "$causal_conv1d_commit" \
    > "$out_dir/commits.txt"
rm -rf /tmp/mamba-src /tmp/causal-conv1d-src
