#!/bin/bash
# Run one AMD CI job in its Kubernetes pod. The image ($ROCM_CI_REPO:build-<build id>) already holds vLLM,
# the test deps and /vllm-workspace; this sets up caches, checks the GPUs and runs $VLLM_TEST_COMMANDS.
set -o pipefail
commands=${VLLM_TEST_COMMANDS:?set by the pipeline}
gpus=${VLLM_CI_EXPECTED_GPU_COUNT:-1}
job=${BUILDKITE_JOB_ID:-local} root=/tmp/vllm-ci-${BUILDKITE_JOB_ID:-local}

# --- Environment ---------------------------------------------------------
# Job-local compile caches; the HF hub cache is the node's persistent mount.
export HF_HOME=${HF_HOME:-/home/buildkite-agent/huggingface} PYTORCH_ROCM_ARCH="" PYTHONPATH=/vllm-workspace \
  TMPDIR=$root/tmp VLLM_RPC_BASE_PATH=/tmp TORCHINDUCTOR_CACHE_DIR=$root/inductor TRITON_CACHE_DIR=$root/triton \
  VLLM_CACHE_ROOT=$root/vllm XDG_CACHE_HOME=$root/xdg HF_DATASETS_CACHE=$root/datasets \
  HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-300} HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export TIKTOKEN_RS_CACHE_DIR=$HF_HOME/tiktoken-rs-cache
# Terminal / color: keep CI logs readable and pytest output colorized.
: "${TERM:=xterm-256color}" "${FORCE_COLOR:=1}" "${CLICOLOR_FORCE:=1}" "${PY_COLORS:=1}"
export TERM FORCE_COLOR CLICOLOR_FORCE PY_COLORS
# Dump stacks after 25 minutes, stop a single test after 40.
export PYTHONFAULTHANDLER=1 PYTEST_TIMEOUT=${PYTEST_TIMEOUT:-2400} PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} --color=yes \
  --durations=25 --durations-min=1.0 -o faulthandler_timeout=1500 --timeout-method=thread"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME" \
  "$HF_DATASETS_CACHE" "$TIKTOKEN_RS_CACHE_DIR" || exit 1
[[ $(findmnt -n -T "$HF_HOME" -o TARGET) != / ]] || { echo "HF_HOME must be a persistent mount" >&2; exit 1; }
if [[ $(findmnt -n -T "$HF_HOME" -o FSTYPE) == nfs* ]]; then   # hf-xet state must stay off NFS
  export HF_XET_CACHE=$root/hf-xet HF_XET_HIGH_PERFORMANCE=0 HF_XET_RECONSTRUCTION_USE_VECTORED_WRITE=0
fi
[[ $gpus == 0 ]] && export VLLM_TARGET_DEVICE=cpu

# --- Wheel reinstall (python_only_compile.sh path) -----------------------
# /opt/vllm-wheels/vllm-*.whl is baked into the image at build time.
# If VLLM_CI_USE_ARTIFACTS=1 and a fresher wheel was uploaded as a Buildkite
# artifact (e.g. from a same-build wheels step), download and reinstall it.
if [[ $commands == *python_only_compile.sh* ]]; then
  VLLM_VERSION_OVERRIDE=$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))') || exit 1
  export VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE
  if [[ ${VLLM_CI_USE_ARTIFACTS:-0} == 1 ]] && command -v buildkite-agent >/dev/null 2>&1; then
    _artifact_glob="${VLLM_CI_ARTIFACT_GLOB:-artifacts/vllm-rocm-install/vllm-rocm-install.tar.gz}"
    _artifact_dir=$(mktemp -d -t vllm-rocm-artifact.XXXXXX)
    if buildkite-agent artifact download "$_artifact_glob" "$_artifact_dir" 2>/dev/null; then
      _whl=$(find "$_artifact_dir" -name 'vllm-*.whl' | head -1)
      [[ -n $_whl ]] || { tar -xzf "$_artifact_dir"/*.tar.gz -C "$_artifact_dir" 2>/dev/null; _whl=$(find "$_artifact_dir" -name 'vllm-*.whl' | head -1); }
      if [[ -n $_whl ]]; then
        echo "--- Installing vLLM wheel from artifact: $(basename "$_whl")"
        pip install --no-deps --force-reinstall "$_whl" || exit 1
        VLLM_VERSION_OVERRIDE=$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))') || exit 1
        export VLLM_VERSION_OVERRIDE
      fi
    fi
    rm -rf "$_artifact_dir"
  fi
fi

# --- DPX GPU partition lock -----------------------------------------------
if [[ ${BUILDKITE_AGENT_META_DATA_QUEUE:-} == *dpx* && $gpus != 0 ]]; then
  dev=(/dev/dri/renderD*); [[ ${#dev[@]} == 1 ]] || { echo "DPX: expected one render device" >&2; exit 1; }
  mkdir -p "$HF_HOME/.vllm-dpx-locks"
  exec 9>>"$HF_HOME/.vllm-dpx-locks/$(stat -Lc %t-%T "${dev[0]}").lock"
  flock -n 9 || { echo "DPX GPU collision on ${dev[0]}" >&2; exit 1; }
fi

# --- Lifecycle logging ----------------------------------------------------
_teardown_log() { printf '[amd-ci-teardown] event=%s' "$1" >&2; shift; (($#)) && printf ' %s' "$@" >&2; printf '\n' >&2; }

# --- Diagnostics (always-on failure bundle) --------------------------------
diagnostics() {
  local d=artifacts/amd-gpu-diagnostics/$job; mkdir -p "$d"
  { date; echo "node=${VLLM_CI_K8S_NODE_NAME:-?} pod=${VLLM_CI_K8S_POD_NAME:-?}"
    timeout 10 amd-smi list; timeout 10 rocm-smi --showpids; ps -eo pid,etime,rss,args --sort=-rss | head -30
    df -h /tmp /dev/shm "$HF_HOME"; free -g; ls -l /dev/kfd /dev/dri; } >"$d/diagnostics.log" 2>&1
}

# --- Preflight: image imports and GPU count --------------------------------
python3 -c "import vllm, torch, transformers, ray" || exit 1
if [[ $gpus != 0 ]] && ! { rocminfo >/dev/null && python3 -c "import torch; n = torch.cuda.device_count(); \
    assert torch.version.hip and n == $gpus, f'expected $gpus GPUs, found {n}'"; }; then
  diagnostics; exit 1
fi

# --- Run ------------------------------------------------------------------
# Orchestration vars stay out of vLLM's environment (other VLLM_CI_* are test knobs, see tests/ci_envs.py).
unset VLLM_TEST_COMMANDS VLLM_CI_EXPECTED_GPU_COUNT VLLM_CI_DIAGNOSTICS_DIR
_teardown_log workload_started "timeout_s=${CONTAINER_TIMEOUT_S:-0}"
rc=0
if [[ ${CONTAINER_TIMEOUT_S:-0} -gt 0 ]]; then
  timeout --verbose --foreground --signal=TERM --kill-after=10s "${CONTAINER_TIMEOUT_S}s" \
    bash -o pipefail -c "$commands" || rc=$?
else
  bash -o pipefail -c "$commands" || rc=$?
fi
[[ $rc == 5 ]] && rc=0                      # pytest: no tests collected
[[ $rc == 124 ]] && _teardown_log deadline_exhausted "configured_s=${CONTAINER_TIMEOUT_S:-0}" "status=$rc"
_teardown_log workload_finished "status=$rc"
[[ $rc != 0 ]] && diagnostics
exit $rc
