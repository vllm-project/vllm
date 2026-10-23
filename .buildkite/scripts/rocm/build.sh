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
#   release  Release image + wheels on the CI-tested layers (release pipeline, AWS agents).
#
# Stack: rocm (TheRock ROCm 10). Force a stack rebuild: bump ARG REFRESH
#        in docker/Dockerfile.rocm_base.
#
# Trust: ROCM_CI_REPO is set by ci-infra (CI) or release-pipeline.yaml (release).
#   Trusted main builds: rocm/vllm-ci
#   PR / fork builds:    rocm/vllm-dev    (pr<N>- tags and caches; trusted builds never read it)
#   Release pipeline:    its ECR repo     (reads rocm/vllm-ci and its cache, writes neither; own cache in ECR)
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
export CACHE=${ROCM_CI_CACHE_REPO:-rocm/vllm-ci-cache}  # every build reads the trusted cache
export CACHE_TO=$CACHE:                     # ref prefix trusted builds write (HCL appends <stack>-<name>)
export BUILDKIT_PROGRESS=${BUILDKIT_PROGRESS:-plain}  # TTY progress redraws bloat CI logs

# PR builds write to a per-PR cache and prefix their image tags. The release pipeline's agents
# cannot write rocm/vllm-ci or its cache: release writes its images and its own cache to $REPO,
# so the next release reuses this one's all-arch compile.
PFX=""
if [[ $MODE == release ]]; then
  CACHE_TO=$REPO:cache-
elif [[ $REPO != "$TRUSTED" ]]; then
  PFX="pr${BUILDKITE_PULL_REQUEST:-local}-"
  CACHE_TO=$REPO:cache-pr${BUILDKITE_PULL_REQUEST:-local}-
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
# to a docker-container builder if no socket is present.
if ! docker buildx inspect vllm-rocm >/dev/null 2>&1; then
  if [[ -S /run/buildkit/buildkitd.sock ]]; then
    drv=(--driver remote unix:///run/buildkit/buildkitd.sock)
  else
    drv=(--driver docker-container \
      ${BUILDKITD_CONFIG:+--buildkitd-config "$BUILDKITD_CONFIG"})
  fi
  # || inspect: tolerate a race where another job created it between the check and create.
  docker buildx create --name vllm-rocm "${drv[@]}" >/dev/null \
    || docker buildx inspect vllm-rocm >/dev/null
fi
export BUILDX_BUILDER=vllm-rocm
docker buildx version

# ---------------------------------------------------------------------------
# Git tags — required by the version stage
# ---------------------------------------------------------------------------
# setuptools_scm needs a reachable v* tag. Agents clone without tags, partially
# and with their objects in a host mirror (--reference) that the build context
# cannot see; --force updates tags that moved (e.g. a re-tag after a release
# candidate). The version stage reads .git_archival.txt instead of .git.
git fetch -q --force --tags --filter=tree:0 origin
git describe --tags --match 'v[0-9]*' >/dev/null 2>&1 \
  || git fetch -q --unshallow --filter=tree:0 origin
printf 'node: %s\nnode-date: %s\ndescribe-name: %s\n' "$(git rev-parse HEAD)" \
  "$(git log -1 --format=%cI)" "$(git describe --tags --long --abbrev=40 --match 'v[0-9]*')" \
  > .git_archival.txt

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# digest <ref>
#   Print the sha256 digest of <ref>. Returns 1 if the tag does not exist.
#   The "Digest:" line of the plain output is parsed and validated (as the old CI
#   scripts did): --format output and stray stderr are not safe to splice into a ref.
#   Any other registry error, after retries, is fatal: kill $$ instead of exit 1
#   because this function is called from $() subshells where exit only exits the subshell.
digest() {
  local out rc d try
  for try in 1 2 3; do
    rc=0
    out=$(docker buildx imagetools inspect "$1" 2>&1) || rc=$?
    d=$(awk '$1 == "Digest:" { print $2; exit }' <<<"$out")
    if ((rc == 0)) && [[ $d =~ ^sha256:[0-9a-f]{64}$ ]]; then echo "$d"; return 0; fi
    if ((rc != 0)) && grep -qiE 'not found|manifest unknown' <<<"$out"; then return 1; fi
    ((try == 3)) || sleep 5
  done
  echo "registry error for $1 (exit $rc): $out" >&2
  kill $$
}

# label <ref> <key>  — read one image label
label() { docker buildx imagetools inspect "$1" \
            --format "{{index .Image.Config.Labels \"$2\"}}"; }

# bake <targets...>  — invoke docker buildx bake with the project HCL
bake()   { docker buildx bake -f docker/docker-bake-rocm.hcl "$@"; }

# meta <get|set> <key> [value]  — Buildkite build metadata
meta()   { buildkite-agent meta-data "$@"; }

# ---------------------------------------------------------------------------
# Content keys
# ---------------------------------------------------------------------------
# Ks (stack key): hash of the stack Dockerfile + therock_wheels.py.
#   A Ks cache miss triggers a full stack rebuild (hours).
# Kr (runtime key): Ks + the portion of Dockerfile.rocm above the per-commit
#   marker + all pip requirement files.
#   A Kr cache miss triggers a runtime rebuild (minutes).

_trw=$( [[ $STACK == rocm ]] && echo tools/vllm-rocm/therock_wheels.py || true )
# The layer compression is part of every image's identity: an image left in an older
# compression would hand later builds a second copy of the same layers.
Ks=$( { echo "layers: zstd-3"; git ls-tree HEAD -- "$SDF" ${_trw:+"$_trw"}; } | sha256sum | cut -c1-16)
Kr=$(
  {
    echo "$Ks"
    sed '/^# ==== per-commit stages/q' docker/Dockerfile.rocm
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
#   1. Trusted registry (rocm/vllm-ci:<tag>) — safe for any build to reuse.
#   2. This repo's own tag ($REPO:$PFX<tag>) — PR builds and the release pipeline write here.
#   3. Neither found: build via bake and push.
# A <sibling-suffix> means the image only counts as present when the sibling
# tag also exists (used by stack: must have both the image and -wheels).
ensure() {
  local target=$1 tag=$2 sibling=${3:-} ref d

  for ref in "$TRUSTED:$tag" "$REPO:$PFX$tag"; do
    if d=$(digest "$ref") && { [[ -z $sibling ]] || digest "$ref$sibling" >/dev/null; }; then
      echo "Reusing $ref@$d" >&2
      echo "$ref@$d"
      return 0
    fi
  done

  # Neither copy found — build and push.
  ref=$REPO:$PFX$tag
  echo "--- :docker: Building $target as $ref" >&2
  # Explicit checks: $(ensure ...) runs in a subshell, where set -e does not apply.
  TAG=$ref bake "$target" >&2 || { echo "building $ref failed" >&2; return 1; }
  d=$(digest "$ref") || { echo "$ref is missing from the registry after its push" >&2; return 1; }
  echo "$ref@$d"
}

# load — read stack/runtime refs from Buildkite metadata (written by base mode).
load() {
  STACK_IMAGE=$(meta get "rocm-$STACK-stack")
  STACK_WHEELS=$(meta get "rocm-$STACK-stack-wheels")
  RUNTIME=$(meta get "rocm-$STACK-runtime")
  TEST_DEPS=$(meta get "rocm-$STACK-test-deps")
  export STACK_IMAGE STACK_WHEELS RUNTIME TEST_DEPS
}

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
case $MODE in

  keys)
    # Print current content keys and registry status (useful for on-call triage).
    for t in "stack-$STACK-$Ks" "runtime-$STACK-$Kr" "test-deps-$STACK-$Kr"; do
      echo "$t  $(digest "$TRUSTED:$t" || echo MISSING)"
    done
    ;;

  base)
    # Ensure the stack and runtime images exist; record their digests in metadata
    # so the ci mode can pull them without re-resolving.
    # The "base" bake group builds the stack and its -wheels sibling together.
    STACK_IMAGE=$(ensure base "stack-$STACK-$Ks" -wheels)
    export STACK_IMAGE
    RUNTIME=$(ensure runtime "runtime-$STACK-$Kr")
    export RUNTIME
    TEST_DEPS=$(ensure test-deps "test-deps-$STACK-$Kr")
    STACK_WHEELS=${STACK_IMAGE%@*}-wheels
    STACK_WHEELS=$STACK_WHEELS@$(digest "$STACK_WHEELS")

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
    rm -rf .rocm-wheel
    bake ci
    # For the python-only install test (run-amd-test.sh downloads it); not in the image.
    buildkite-agent artifact upload ".rocm-wheel/vllm-*.whl"
    ;;

  release)
    # Build and push the release image + wheels on the CI-tested layers, reused by digest from
    # rocm/vllm-ci; a key CI has not built yet is built here into $REPO. Only the all-arch
    # compile and the release layers are new.
    STACK_IMAGE=$(ensure base "stack-$STACK-$Ks" -wheels)
    export STACK_IMAGE   # before ensure runtime: a runtime built here must use this stack
    STACK_WHEELS=${STACK_IMAGE%@*}-wheels
    STACK_WHEELS=$STACK_WHEELS@$(digest "$STACK_WHEELS")
    RUNTIME=$(ensure runtime "runtime-$STACK-$Kr")
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
