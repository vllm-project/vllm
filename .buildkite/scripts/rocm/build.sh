#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Build and promote ROCm images via docker/docker-bake-rocm.hcl.
# Pushes only to $ROCM_CI_REPO and the BuildKit cache that goes with it.
#
# Usage: build.sh <stack> <mode>
#
# Modes:
#   keys     Print current Ks/Kr keys and whether their images exist (on-call debug).
#   base     Ensure stack, runtime and test-deps images exist; write their refs to Buildkite metadata.
#   ci       Build and push this build's per-commit test image.
#   release  Release image + wheels from source into the release repo (release pipeline, AWS agents).
#   release-base  Only the release stack and runtime (a serialized step, so one run builds a new key).
#
# Stack: rocm (TheRock ROCm 10). Force a stack rebuild: bump ARG REFRESH
#        in docker/Dockerfile.rocm_base
#
# Trust: ROCM_CI_REPO is set by ci-infra (CI) or release-pipeline.yaml (release).
#   Trusted main builds: rocm/vllm-ci
#   PR / fork builds:    rocm/vllm-dev    (pr<N>- tags and caches; trusted builds never read it)
#   Other untrusted:     rocm/vllm-dev    (br-<branch>-<hash>- tags and caches: nightlies, branch builds)
#   Release pipeline:    its ECR repo     (never reads rocm/vllm-ci or its cache: it builds the
#                                          stack and runtime from source, with only its own ECR cache;
#                                          csrc objects come from the shared S3 sccache, as in CUDA)
# ROCM_CI_TRUSTED_REPO / ROCM_CI_CACHE_REPO only point a local run at a sandbox registry.
set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments and trust
# ---------------------------------------------------------------------------
STACK=$1
MODE=$2
REPO=${ROCM_CI_REPO:?ROCM_CI_REPO must be set (by ci-infra or release-pipeline.yaml)}
TRUSTED=${ROCM_CI_TRUSTED_REPO:-rocm/vllm-ci}

export STACK PUSH=true
export CACHE=${ROCM_CI_CACHE_REPO:-rocm/vllm-ci-cache}  # every CI build reads the trusted cache
export CACHE_TO=$CACHE:                     # ref prefix trusted builds write (HCL appends <stack>-<name>)
export BUILDKIT_PROGRESS=${BUILDKIT_PROGRESS:-plain}  # TTY progress redraws bloat CI logs

# PR builds write to a per-PR cache and prefix their image tags. The release pipeline's agents
# cannot write rocm/vllm-ci or its cache: release writes its images and its own cache to $REPO,
# so the next release reuses this one's all-arch compile.
PFX=""
if [[ $MODE == release* ]]; then
  CACHE="" CACHE_TO=$REPO:cache-   # and reads only that cache: CI agents also run PR code
elif [[ $REPO != "$TRUSTED" ]]; then
  if [[ ${BUILDKITE_PULL_REQUEST:-false} != false ]]; then
    PFX="pr$BUILDKITE_PULL_REQUEST-"
  else   # not a PR (AITER nightly, branch or manual builds): one namespace per branch
    br=${BUILDKITE_BRANCH:-local}   # the hash keeps e.g. feat/x and feat-x apart
    PFX="br-$(printf %s "$br" | tr -c 'A-Za-z0-9.-' - | cut -c1-40)-$(printf %s "$br" | sha256sum | cut -c1-8)-"
  fi
  CACHE_TO=$REPO:cache-$PFX
  # Only the compile stages' cache: everything else is cheap or in the trusted cache.
  export CACHE_CI=false
fi

# Stack Dockerfile — literal path used by ci-infra ci_selector (images.py).
case $STACK in
  rocm) SDF=docker/Dockerfile.rocm_base ;;
  *) echo "Unknown stack: $STACK" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# BuildKit builder
# ---------------------------------------------------------------------------
# Prefer the host buildkitd (it has its own GC policy and cache). Fall back
# to a docker-container builder if no socket is present. Release RUN steps must reach EC2 IMDS
# for sccache's S3 credentials: their fallback builder uses the host network, under its own name.
B=vllm-rocm net=()
[[ $MODE == release* ]] && B=vllm-rocm-release net=(--driver-opt network=host)
if ! docker buildx inspect "$B" >/dev/null 2>&1; then
  if [[ -S /run/buildkit/buildkitd.sock ]]; then
    drv=(--driver remote unix:///run/buildkit/buildkitd.sock)
  else
    drv=(--driver docker-container "${net[@]}" \
      ${BUILDKITD_CONFIG:+--buildkitd-config "$BUILDKITD_CONFIG"})
  fi
  # || inspect: tolerate a race where another job created it between the check and create.
  docker buildx create --name "$B" "${drv[@]}" >/dev/null \
    || docker buildx inspect "$B" >/dev/null
fi
export BUILDX_BUILDER=$B
docker buildx version

# ---------------------------------------------------------------------------
# Build context — an export of HEAD
# ---------------------------------------------------------------------------
# Bake builds .rocm-src, not the checkout: the content keys below hash HEAD, and BuildKit keys
# COPY on file modes, which drift on shared agents. umask 022 gives every agent git's modes.
[[ $(git rev-parse HEAD) == $(git rev-parse "${BUILDKITE_COMMIT:-HEAD}^{commit}") ]] \
  || { echo "HEAD is not BUILDKITE_COMMIT ($BUILDKITE_COMMIT)" >&2; exit 1; }
umask 022
rm -rf .rocm-src && mkdir .rocm-src
git archive HEAD | tar -x --no-same-permissions -C .rocm-src -f -
export CONTEXT=.rocm-src

# ---------------------------------------------------------------------------
# Git tags — required by the version stage
# ---------------------------------------------------------------------------
# setuptools_scm needs a reachable v* tag. Agents clone without tags, partially
# and with their objects in a host mirror (--reference) that the build context
# cannot see; --force updates tags that moved (e.g. a re-tag after a release
# candidate). The context has no .git: the version stage reads .git_archival.txt.
git fetch -q --force --tags --filter=tree:0 origin
git describe --tags --match 'v[0-9]*' >/dev/null 2>&1 \
  || git fetch -q --unshallow --filter=tree:0 origin
printf 'node: %s\nnode-date: %s\ndescribe-name: %s\n' "$(git rev-parse HEAD)" \
  "$(git log -1 --format=%cI)" "$(git describe --tags --long --abbrev=40 --match 'v[0-9]*')" \
  > .rocm-src/.git_archival.txt

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# digest <ref>
#   Print the sha256 digest of <ref>. Returns 1 if the tag does not exist.
#   The "Digest:" line of the plain output is parsed and validated (as the old CI
#   scripts did): --format output and stray stderr are not safe to splice into a ref.
#   Any other registry error, after retries, exits 75 (the steps retry it); in a $()
#   subshell that becomes the substitution's status, which ensure() and set -e pass on.
digest() {
  local out rc d try
  for try in 1 2 3; do
    rc=0
    out=$(docker buildx imagetools inspect "$1" 2>&1) || rc=$?
    d=$(awk '$1 == "Digest:" { print $2; exit }' <<<"$out")
    if ((rc == 0)) && [[ $d =~ ^sha256:[0-9a-f]{64}$ ]]; then echo "$d"; return 0; fi
    if ((rc != 0)) && grep -qiE 'manifest unknown|no such manifest|name unknown|not found|does not exist' <<<"$out"; then
      return 1
    fi
    ((try == 3)) || sleep 5
  done
  echo "registry error for $1 (exit $rc): $out" >&2
  exit 75
}

# label <ref> <key>  — read one image label
label() { docker buildx imagetools inspect "$1" \
            --format "{{index .Image.Config.Labels \"$2\"}}"; }

# bake <targets...>  — invoke docker buildx bake with the project HCL
bake()   { docker buildx bake -f docker/docker-bake-rocm.hcl "$@"; }

# meta <get|set> <key> [value]  — Buildkite build metadata
meta()   { buildkite-agent meta-data "$@"; }

# pushed <bake-metadata-file> <target>  — the digest that bake pushed for <target>
pushed() {
  local d
  d=$(awk -F'"' -v t="$2" '/^  "/ { k = $2 } k == t && $2 == "containerimage.digest" { print $4 }' "$1")
  [[ $d =~ ^sha256:[0-9a-f]{64}$ ]] && echo "$d"
}

# ---------------------------------------------------------------------------
# Content keys
# ---------------------------------------------------------------------------
# Ks (stack key): hash of the stack Dockerfile + therock_wheels.py.
#   A Ks cache miss triggers a full stack rebuild (hours).
# Kr (runtime key): Ks + the portion of Dockerfile.rocm above the per-commit
#   marker + all pip requirement files.
#   A Kr cache miss triggers a runtime rebuild (minutes).

_trw=$( [[ $STACK == rocm ]] && echo tools/vllm-rocm/therock_wheels.py || true )
# The layer format (compression, and mtimes as built: see the bake file) is part of every image's
# identity: an image left in an older format would hand later builds a second copy of the same layers.
Ks=$( { echo "layers: zstd-3, mtimes kept"; git ls-tree HEAD -- "$SDF" ${_trw:+"$_trw"}; } | sha256sum | cut -c1-16)
Kr=$(
  {
    echo "$Ks"
    sed '/^# ==== per-commit stages/q' .rocm-src/docker/Dockerfile.rocm
    git ls-tree HEAD -- \
      requirements/common.txt \
      requirements/rocm.txt \
      requirements/test/rocm.txt \
      rust-toolchain.toml \
      tests/vllm_test_utils \
      tools/install_torchcodec_rocm.sh
  } | sha256sum | cut -c1-16
)
echo "build.sh $STACK $MODE: repo=$REPO trusted=$TRUSTED Ks=$Ks Kr=$Kr"

# ---------------------------------------------------------------------------
# ensure <bake-target> <tag-suffix> [<sibling-suffix>]
# ---------------------------------------------------------------------------
# Resolve a content-keyed image without building if it already exists.
# Resolution order:
#   1. Trusted registry (rocm/vllm-ci:<tag>) — safe for any CI build to reuse. Not for release:
#      CI agents also run PR code, so release never ships an image it did not build itself.
#   2. This repo's own tag ($REPO:$PFX<tag>) — PR builds and the release pipeline write here.
#   3. Neither found: build via bake and push.
# A <sibling-suffix> means the image only counts as present when the sibling
# tag also exists (used by stack: must have both the image and -wheels); then
# both pinned refs are printed, space-separated.
ensure() {
  local target=$1 tag=$2 sibling=${3:-} ref d ds refs rc mf
  refs=("$REPO:$PFX$tag")
  [[ $MODE == release* || $REPO == "$TRUSTED" ]] || refs=("$TRUSTED:$tag" "${refs[@]}")

  for ref in "${refs[@]}"; do
    if d=$(digest "$ref") && { [[ -z $sibling ]] || ds=$(digest "$ref$sibling"); }; then
      echo "Reusing $ref@$d" >&2
      echo "$ref@$d${sibling:+ $ref$sibling@$ds}"
      return 0
    else
      rc=$?; ((rc == 1)) || return $rc   # a registry error is not a miss
    fi
  done

  # Neither copy found — build and push.
  ref=$REPO:$PFX$tag
  echo "--- :docker: Building $target as $ref" >&2
  # Explicit checks: $(ensure ...) runs in a subshell, where set -e does not apply.
  mf=$(mktemp)
  TAG=$ref bake --metadata-file "$mf" "$target" >&2 || { rm -f "$mf"; echo "building $ref failed" >&2; return 1; }
  # The digests this bake pushed (base: stack-<stack>, stack-wheels), not the tags', which by now
  # may hold a concurrent build's push of the same key.
  if ! d=$(pushed "$mf" "${target/#base/stack-$STACK}") \
      || { [[ -n $sibling ]] && ! ds=$(pushed "$mf" stack-wheels); }; then
    rm -f "$mf"; echo "bake recorded no digest for $ref" >&2; return 1
  fi
  rm -f "$mf"
  echo "$ref@$d${sibling:+ $ref$sibling@$ds}"
}

# load — read stack/runtime refs from Buildkite metadata (written by base mode).
load() {
  STACK_IMAGE=$(meta get "rocm-$STACK-stack")
  STACK_WHEELS=$(meta get "rocm-$STACK-stack-wheels")
  RUNTIME=$(meta get "rocm-$STACK-runtime")
  TEST_DEPS=$(meta get "rocm-$STACK-test-deps")
  export STACK_IMAGE STACK_WHEELS RUNTIME TEST_DEPS
  for ref in "$STACK_IMAGE" "$STACK_WHEELS" "$RUNTIME" "$TEST_DEPS"; do
    [[ $ref =~ @sha256:[0-9a-f]{64}$ ]] || { echo "not a pinned image ref in the metadata: '$ref'" >&2; exit 1; }
  done
}

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
case $MODE in

  keys)
    # Print current content keys and registry status (useful for on-call triage).
    for t in "stack-$STACK-$Ks" "runtime-$STACK-$Kr" "test-deps-$STACK-$Kr"; do
      d=$(digest "$TRUSTED:$t") || { (($? == 1)) || exit 75; d=MISSING; }
      echo "$t  $d"
    done
    ;;

  base)
    # Ensure the stack and runtime images exist; record their digests in metadata
    # so the ci mode can pull them without re-resolving.
    # The "base" bake group builds the stack and its -wheels sibling together.
    STACK_IMAGE=$(ensure base "stack-$STACK-$Ks" -wheels)
    STACK_WHEELS=${STACK_IMAGE#* } STACK_IMAGE=${STACK_IMAGE%% *}
    export STACK_IMAGE
    RUNTIME=$(ensure runtime "runtime-$STACK-$Kr")
    export RUNTIME
    TEST_DEPS=$(ensure test-deps "test-deps-$STACK-$Kr")

    meta set "rocm-$STACK-stack"        "$STACK_IMAGE"
    meta set "rocm-$STACK-runtime"      "$RUNTIME"
    meta set "rocm-$STACK-test-deps"    "$TEST_DEPS"
    meta set "rocm-$STACK-stack-wheels" "$STACK_WHEELS"
    echo "stack=$STACK_IMAGE runtime=$RUNTIME test-deps=$TEST_DEPS stack-wheels=$STACK_WHEELS"
    ;;

  ci)
    # Build and push the per-commit test image for GPU test steps.
    load
    export TAG=$REPO:build-$BUILDKITE_BUILD_ID
    # AITER nightly (AITER_NIGHTLY=1 from the schedule): the test stage installs the newest
    # AITER nightly that fits the image; an AITER_SPEC in the environment pins one instead.
    rm -rf .rocm-wheel .rocm-kernel-map
    bake ci
    # For the python-only install test (run-amd-test.sh downloads it); not in the image.
    buildkite-agent artifact upload ".rocm-wheel/vllm-*.whl"
    # CI test selection (VLLM_KERNEL_SYMBOL_MAP=1, set by ci-infra): the kernel symbol map of this
    # build's csrc, from the build cache. Optional: a missing map never fails the image build.
    if [[ ${VLLM_KERNEL_SYMBOL_MAP:-0} == 1 ]]; then
      if bake kernel-symbol-map && [[ -s .rocm-kernel-map/kernel_symbol_map.rocm.json.gz ]] \
          && (cd .rocm-kernel-map && buildkite-agent artifact upload kernel_symbol_map.rocm.json.gz); then
        :
      else
        echo "ROCm kernel symbol map export or upload failed; continuing without it" >&2
      fi
    fi
    ;;

  release|release-base)
    # Build and push the release image + wheels into $REPO. The stack and runtime come from this
    # repo only (built here once per key, from the same Dockerfiles and keys CI tests), never from
    # rocm/vllm-ci or its cache.
    STACK_IMAGE=$(ensure base "stack-$STACK-$Ks" -wheels)
    STACK_WHEELS=${STACK_IMAGE#* } STACK_IMAGE=${STACK_IMAGE%% *}
    export STACK_IMAGE   # before ensure runtime: a runtime built here must use this stack
    RUNTIME=$(ensure runtime "runtime-$STACK-$Kr")
    [[ $MODE == release ]] || exit 0
    export STACK_WHEELS RUNTIME
    export TAG=$REPO:$STACK-$BUILDKITE_COMMIT

    # Resolve the vLLM version from the commit (setuptools_scm).
    PUSH=false bake version
    VLLM_VERSION=$(cat .rocm-version/vllm-version.txt)
    export VLLM_VERSION

    # Skip if this exact version was already released (handles rc → final re-tags).
    if digest "$TAG" >/dev/null \
        && [[ $(label "$TAG" org.opencontainers.image.version) == "$VLLM_VERSION" ]]; then
      echo "Release image $TAG already exists with version $VLLM_VERSION — skipping." >&2
      exit 0
    fi

    # Wheels first: their all-arch csrc compile is the long part, and vllm-openai reuses it.
    bake wheels openai-smoke
    # Alias the runtime as the -base tag (the base-nightly consumer reference).
    docker buildx imagetools create -t "$TAG-base" "$RUNTIME"
    # Push the vllm-openai tag last — its presence marks the set as complete.
    bake vllm-openai
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
