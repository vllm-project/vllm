#!/bin/bash
# The ROCm image content keys, sourced by build.sh and read by ci-infra's "ROCm base image cache"
# job, which needs the stack key of a PR before it is built. Needs STACK and SDF (the stack
# Dockerfile) set, and a git repository.
#
# Ks (stack key): hash of the stack Dockerfile + therock_wheels.py.
#   A Ks cache miss triggers a full stack rebuild (hours).
# Kr (runtime key): Ks + the portion of Dockerfile.rocm above the per-commit
#   marker + all pip requirement files.
#   A Kr cache miss triggers a runtime rebuild (minutes).

_trw=$( [[ $STACK == rocm ]] && echo tools/vllm-rocm/therock_wheels.py || true )
# keys_at <rev>  — print "Ks Kr" for the files at <rev>. In $(), set -e is off: failures are returned.
keys_at() {
  local rev=$1 Ks Kr
  # The layer format (compression, and mtimes as built: see the bake file) is part of every image's
  # identity: an image left in an older format would hand later builds a second copy of the same layers.
  Ks=$( { echo "layers: zstd-3, mtimes kept"; git ls-tree "$rev" -- "$SDF" ${_trw:+"$_trw"}; } | sha256sum | cut -c1-16) \
    || return 1
  Kr=$(
    {
      echo "$Ks"
      git show "$rev:docker/Dockerfile.rocm" | sed -n '1,/^# ==== per-commit stages/p' || exit 1
      git ls-tree "$rev" -- \
        requirements/common.txt \
        requirements/rocm.txt \
        requirements/test/rocm.txt \
        rust-toolchain.toml \
        tests/vllm_test_utils \
        tools/install_torchcodec_rocm.sh
    } | sha256sum | cut -c1-16
  ) || return 1
  echo "$Ks $Kr"
}
