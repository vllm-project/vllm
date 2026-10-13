#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build ROCm images with docker/docker-bake-rocm.hcl. Pushes only to $ROCM_CI_REPO and its cache.
#   build.sh <stack> keys     print the stack/runtime keys and whether their images exist (on-call)
#   build.sh <stack> base     ensure stack + runtime              -> meta rocm-<stack>-{stack,stack-wheels,runtime}
#   build.sh <stack> ci       this build's test image, CI arches  -> $ROCM_CI_REPO:build-<build id>
#   build.sh <stack> check    CI-arch build + import smoke, no push (pre-merge signal for the non-default stack)
#   build.sh <stack> csrc     all-arch csrc compile into the cache (release, serialized per stack)
#   build.sh <stack> release  all-arch image + wheels             -> rocm/vllm-ci:<stack>-<commit>{-wheels,-base,}
# <stack>: rocm (TheRock ROCm 10, default) | rocm_72 (apt ROCm 7.2). Force a rebuild: bump ARG REFRESH.
# Trust is decided by ci-infra, not here: it sets ROCM_CI_REPO (rocm/vllm-ci for trusted builds, else rocm/vllm-ci-pr).
set -euo pipefail
export STACK=$1 PUSH=true CACHE=rocm/vllm-ci-cache CACHE_TO=rocm/vllm-ci-cache:latest
MODE=$2 REPO=${ROCM_CI_REPO:?set by ci-infra} PFX=""
if [[ $REPO != rocm/vllm-ci ]]; then PFX="pr${BUILDKITE_PULL_REQUEST:-local}-" CACHE_TO=$REPO:cache-pr${BUILDKITE_PULL_REQUEST:-local}-; fi
case $STACK in   # literal paths (ci_selector)
  rocm) SDF=docker/Dockerfile.rocm_base ;; rocm_72) SDF=docker/Dockerfile.rocm_72_base ;; *) exit 2 ;; esac

if ! docker buildx inspect vllm-rocm >/dev/null 2>&1; then   # prefer the host buildkitd (and its GC policy)
  if [[ -S /run/buildkit/buildkitd.sock ]]; then drv=(--driver remote unix:///run/buildkit/buildkitd.sock)
  else drv=(--driver docker-container ${BUILDKITD_CONFIG:+--buildkitd-config "$BUILDKITD_CONFIG"}); fi
  docker buildx create --name vllm-rocm "${drv[@]}" >/dev/null || docker buildx inspect vllm-rocm >/dev/null  # race
fi
export BUILDX_BUILDER=vllm-rocm
# setuptools_scm (version stage) needs a v* tag in reach; agents clone shallow and tagless. Tags move: --force.
git fetch -q --force --tags --filter=tree:0 origin
git describe --tags --match 'v[0-9]*' >/dev/null 2>&1 || git fetch -q --unshallow --filter=tree:0 origin

# digest <ref>: print its digest; status 1 iff the tag does not exist; any other registry error fails the job.
# kill $$ is used (not exit 1) because digest() runs inside $() subshells where exit only exits the subshell.
digest() {
  local o; o=$(docker buildx imagetools inspect "$1" --format '{{.Manifest.Digest}}' 2>&1) && { echo "$o"; return; }
  grep -qiE 'not found|manifest unknown' <<<"$o" && return 1
  echo "registry error for $1: $o" >&2; kill $$
}
label()  { docker buildx imagetools inspect "$1" --format "{{index .Image.Config.Labels \"$2\"}}"; }
bake()   { docker buildx bake -f docker/docker-bake-rocm.hcl "$@"; }
meta()   { buildkite-agent meta-data "$@"; }
# Ks: the stack recipe. Kr: Ks + the runtime half of Dockerfile.rocm + the pip inputs.
# therock_wheels.py is only used by the rocm stack; include it in Ks only for that stack.
_TRW=$( [[ $STACK == rocm ]] && echo tools/vllm-rocm/therock_wheels.py || true )
Ks=$(git ls-tree HEAD -- "$SDF" $_TRW | sha256sum | cut -c1-16)
Kr=$( { echo "$Ks"; sed '/^# ==== per-commit stages/q' docker/Dockerfile.rocm
        git ls-tree HEAD -- requirements/common.txt requirements/rocm.txt requirements/test/rocm.txt \
          tools/install_torchcodec_rocm.sh; } | sha256sum | cut -c1-16)

# ensure <bake target|group> <tag> [<sibling suffix>]: print ref@digest of the trusted image, else of ours, else
# build + push ours. With a suffix (stack: -wheels) the image counts as present only if the sibling exists too.
ensure() {
  local r d
  for r in "rocm/vllm-ci:$2" "$REPO:$PFX$2"; do
    d=$(digest "$r") && { [[ -z ${3:-} ]] || digest "$r$3" >/dev/null; } && { echo "$r@$d"; return; }
  done
  TAG=$REPO:$PFX$2 bake "$1" >&2
  d=$(digest "$REPO:$PFX$2") && echo "$REPO:$PFX$2@$d"     # fails the step if the push is not visible
}
load() { STACK_IMAGE=$(meta get "rocm-$STACK-stack"); STACK_WHEELS=$(meta get "rocm-$STACK-stack-wheels")
         RUNTIME=$(meta get "rocm-$STACK-runtime"); export STACK_IMAGE STACK_WHEELS RUNTIME; }

case $MODE in
keys)
  for t in "stack-$STACK-$Ks" "runtime-$STACK-$Kr"; do echo "$t $(digest "rocm/vllm-ci:$t" || echo MISSING)"; done ;;
base)
  STACK_IMAGE=$(ensure base "stack-$STACK-$Ks" -wheels); export STACK_IMAGE
  RUNTIME=$(ensure runtime "runtime-$STACK-$Kr"); w=${STACK_IMAGE%@*}-wheels
  meta set "rocm-$STACK-stack" "$STACK_IMAGE"; meta set "rocm-$STACK-runtime" "$RUNTIME"
  meta set "rocm-$STACK-stack-wheels" "$w@$(digest "$w")" ;;
ci)
  load; export TAG=$REPO:build-$BUILDKITE_BUILD_ID
  [[ ${AITER_NIGHTLY:-0} != 1 ]] || export AITER_SPEC=${AITER_SPEC:?set by the AITER nightly schedule}
  bake ci ;;
check) load; PUSH=false bake test-smoke ;;
csrc)  load; PUSH=false bake csrc-all ;;
release)
  [[ $REPO == rocm/vllm-ci ]] || { echo "release mode is trusted-only" >&2; exit 1; }
  load; export TAG=rocm/vllm-ci:$STACK-$BUILDKITE_COMMIT
  PUSH=false bake version; VLLM_VERSION=$(cat .rocm-version/vllm-version.txt); export VLLM_VERSION
  # Reuse only a finished build of the same version: a final tag often sits on its rc's commit.
  if digest "$TAG" >/dev/null && [[ $(label "$TAG" org.opencontainers.image.version) == "$VLLM_VERSION" ]]; then exit 0; fi
  bake wheels openai-smoke                               # -wheels first (csrc: cache hit from rel-csrc)
  docker buildx imagetools create -t "$TAG-base" "$RUNTIME"
  bake vllm-openai ;;                                    # the image tag last: it marks the set complete
*) exit 2 ;;
esac
