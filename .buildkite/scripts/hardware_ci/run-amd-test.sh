#!/bin/bash
# Run one AMD CI job in its Kubernetes pod.
#
# The container image ($ROCM_CI_REPO:build-<build-id>) already contains vLLM,
# all test dependencies, and /vllm-workspace. This script:
#   1. Sets up per-job local caches and validates the HF mount.
#   2. Optionally reinstalls a fresher wheel from a Buildkite artifact.
#   3. Acquires the DPX GPU partition lock when running on a DPX agent.
#   4. Runs a preflight check (imports + GPU count).
#   5. Executes $VLLM_TEST_COMMANDS under a wall-clock timeout.
#   6. Writes a diagnostics bundle on failure.
set -o pipefail

# ---------------------------------------------------------------------------
# Inputs — set by the pipeline / ci-infra template
# ---------------------------------------------------------------------------
commands=${VLLM_TEST_COMMANDS:?VLLM_TEST_COMMANDS must be set by the pipeline}
gpus=${VLLM_CI_EXPECTED_GPU_COUNT:-1}
job=${BUILDKITE_JOB_ID:-local}
# Short: Ray puts AF_UNIX sockets under $TMPDIR, whose paths may not exceed 107 bytes.
root=/tmp/vllm-${job##*-}
root=${root:0:22}

# ---------------------------------------------------------------------------
# Cache directories — all under $root so they are job-local and isolated.
# HF hub is the one exception: it lives on the node's persistent NFS mount.
# ---------------------------------------------------------------------------
export HF_HOME=${HF_HOME:-/home/buildkite-agent/huggingface}
export TIKTOKEN_RS_CACHE_DIR=$HF_HOME/tiktoken-rs-cache
export PYTHONPATH=/vllm-workspace
export PYTORCH_ROCM_ARCH=""                        # let vLLM detect arches at runtime
export TMPDIR=$root/tmp
export VLLM_RPC_BASE_PATH=/tmp
export TORCHINDUCTOR_CACHE_DIR=$root/inductor
export TRITON_CACHE_DIR=$root/triton
export VLLM_CACHE_ROOT=$root/vllm
export XDG_CACHE_HOME=$root/xdg
export HF_DATASETS_CACHE=$root/datasets
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-300}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}

mkdir -p \
  "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE" \
  "$TIKTOKEN_RS_CACHE_DIR" || exit 1

# HF_HOME must be a persistent mount, not rootfs (catches misconfigured nodes).
[[ $(findmnt -n -T "$HF_HOME" -o TARGET) != / ]] \
  || { echo "HF_HOME is on rootfs — it must be a persistent node mount" >&2; exit 1; }

# hf-xet state corrupts on NFS; redirect it to a local tmpdir when needed.
if [[ $(findmnt -n -T "$HF_HOME" -o FSTYPE) == nfs* ]]; then
  export HF_XET_CACHE=$root/hf-xet
  export HF_XET_HIGH_PERFORMANCE=0
  export HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE=0
fi

# ---------------------------------------------------------------------------
# Terminal and pytest settings
# ---------------------------------------------------------------------------
# Keep CI log output colorized.
: "${TERM:=xterm-256color}" "${FORCE_COLOR:=1}" "${CLICOLOR_FORCE:=1}" "${PY_COLORS:=1}"
export TERM FORCE_COLOR CLICOLOR_FORCE PY_COLORS

# Fault-handler dumps stacks after 25 min; pytest-timeout kills a single test
# after 40 min (PYTEST_TIMEOUT=2400 s). Both are overridable by the pipeline.
export PYTHONFAULTHANDLER=1
export PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-2400}
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} \
  --color=yes \
  --durations=25 \
  --durations-min=1.0 \
  -o faulthandler_timeout=1500 \
  --timeout-method=thread"

# A native fault prints its backtrace, and once pytest has its result, a fault in native
# library teardown (where it has crashed with SIGSEGV after every test passed) exits with
# that result instead of failing the job (amd_ci_teardown.py).
ci_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$ci_dir:$PYTHONPATH"
export PYTEST_ADDOPTS="$PYTEST_ADDOPTS -p amd_ci_teardown"

[[ $gpus == 0 ]] && export VLLM_TARGET_DEVICE=cpu

# ---------------------------------------------------------------------------
# Wheel reinstall — python_only_compile.sh path
# ---------------------------------------------------------------------------
# python_only_compile.sh reinstalls vLLM around this build's wheel, which build.sh uploads as an
# artifact of the image build step (kept out of the image: only this test needs it).
if [[ $commands == *python_only_compile.sh* ]]; then
  wheel_dir=$(mktemp -d)
  buildkite-agent artifact download ".rocm-wheel/vllm-*.whl" "$wheel_dir" --step image-build-amd \
    || exit 1
  VLLM_PRECOMPILED_WHEEL_LOCATION=$(echo "$wheel_dir"/.rocm-wheel/vllm-*.whl)
  VLLM_VERSION_OVERRIDE=$(python3 -P -c \
    'import importlib.metadata as m; print(m.version("vllm"))') || exit 1
  export VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE VLLM_PRECOMPILED_WHEEL_LOCATION
fi

# ---------------------------------------------------------------------------
# DPX GPU partition lock
# ---------------------------------------------------------------------------
# DPX nodes expose exactly one GPU partition per pod. A host-wide lock file
# prevents a second job from landing on the same device simultaneously.
_acquire_dpx_lock() {
  local dev=(/dev/dri/renderD*)
  [[ ${#dev[@]} == 1 ]] \
    || { echo "DPX: expected exactly one render device, found ${#dev[@]}" >&2; return 1; }
  mkdir -p "$HF_HOME/.vllm-dpx-locks"
  exec 9>>"$HF_HOME/.vllm-dpx-locks/$(stat -Lc %t-%T "${dev[0]}").lock"
  flock -n 9 || { echo "DPX GPU collision on ${dev[0]}" >&2; return 1; }
}

if [[ ${BUILDKITE_AGENT_META_DATA_QUEUE:-} == *dpx* && $gpus != 0 ]]; then
  _acquire_dpx_lock || exit 1
fi

# ---------------------------------------------------------------------------
# Lifecycle logging
# ---------------------------------------------------------------------------
# Structured events for infra dashboards keyed on [amd-ci-teardown].
# Usage: _teardown_log <event> [key=val ...]
_teardown_log() {
  local event=$1; shift
  printf '[amd-ci-teardown] event=%s' "$event" >&2
  (($#)) && printf ' %s' "$@" >&2
  printf '\n' >&2
}

# ---------------------------------------------------------------------------
# Diagnostics bundle (always collected on failure)
# ---------------------------------------------------------------------------
# Writes to artifacts/amd-gpu-diagnostics/<job>/diagnostics.log, which the
# pipeline's artifact_paths uploads even when the step fails.
diagnostics() {
  local d=artifacts/amd-gpu-diagnostics/$job
  mkdir -p "$d"
  {
    date
    echo "node=${VLLM_CI_K8S_NODE_NAME:-?}  pod=${VLLM_CI_K8S_POD_NAME:-?}"
    echo "--- amd-smi ---"
    timeout 10 amd-smi list
    echo "--- rocm-smi pids ---"
    timeout 10 rocm-smi --showpids
    echo "--- top processes by RSS ---"
    ps -eo pid,etime,rss,args --sort=-rss | head -30
    echo "--- disk / memory ---"
    df -h /tmp /dev/shm "$HF_HOME"
    free -g
    echo "--- devices ---"
    ls -l /dev/kfd /dev/dri
  } >"$d/diagnostics.log" 2>&1
}

# ---------------------------------------------------------------------------
# Preflight: verify imports and GPU visibility
# ---------------------------------------------------------------------------
# -P: the checkout (cwd) has a vllm/ source tree that would shadow the installed package.
python3 -P -c "import vllm, torch, transformers, ray" || exit 1

if [[ $gpus != 0 ]]; then
  rocminfo >/dev/null || { diagnostics; exit 1; }
  python3 - <<EOF || { diagnostics; exit 1; }
import torch
n = torch.cuda.device_count()
assert torch.version.hip and n == $gpus, \
    f"preflight: expected $gpus ROCm GPU(s), found {n}"
EOF
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
# Scrub orchestration vars before handing control to test commands.
# (VLLM_CI_* that are test knobs are intentionally left — see tests/ci_envs.py)
unset VLLM_TEST_COMMANDS VLLM_CI_EXPECTED_GPU_COUNT VLLM_CI_DIAGNOSTICS_DIR

_teardown_log workload_started "timeout_s=${CONTAINER_TIMEOUT_S:-0}"

rc=0
if [[ ${CONTAINER_TIMEOUT_S:-0} -gt 0 ]]; then
  # timeout fires TERM, then KILL after 10 s — mirrors the old docker run path.
  # Not --foreground: the whole process group (pytest, engine workers) must stop, not just bash.
  # That group is in the background of the job's PTY, where reading the terminal stops it
  # (SIGTTIN: apt, transformers' trust_remote_code prompt), so stdin is /dev/null.
  timeout --verbose --signal=TERM --kill-after=10s \
    "${CONTAINER_TIMEOUT_S}s" bash -o pipefail -c "$commands" </dev/null || rc=$?
else
  bash -o pipefail -c "$commands" </dev/null || rc=$?
fi

[[ $rc == 5   ]] && rc=0   # pytest exit 5 = no tests collected, not a failure
[[ $rc == 124 ]] && _teardown_log deadline_exhausted \
  "configured_s=${CONTAINER_TIMEOUT_S:-0}" "status=$rc"
_teardown_log workload_finished "status=$rc"

[[ $rc != 0 ]] && diagnostics
exit $rc
