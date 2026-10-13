#!/bin/bash
# Run one AMD CI job in its Kubernetes pod. The image ($ROCM_CI_REPO:build-<build id>) already holds vLLM,
# the test deps and /vllm-workspace; this sets up caches, checks the GPUs and runs $VLLM_TEST_COMMANDS.
set -o pipefail
commands=${VLLM_TEST_COMMANDS:?set by the pipeline}
gpus=${VLLM_CI_EXPECTED_GPU_COUNT:-1}
job=${BUILDKITE_JOB_ID:-local} root=/tmp/vllm-ci-${BUILDKITE_JOB_ID:-local}

# Job-local compile caches; the HF hub cache is the node's persistent mount.
export HF_HOME=${HF_HOME:-/home/buildkite-agent/huggingface} PYTORCH_ROCM_ARCH="" PYTHONPATH=/vllm-workspace \
  TMPDIR=$root/tmp VLLM_RPC_BASE_PATH=/tmp TORCHINDUCTOR_CACHE_DIR=$root/inductor TRITON_CACHE_DIR=$root/triton \
  VLLM_CACHE_ROOT=$root/vllm XDG_CACHE_HOME=$root/xdg HF_DATASETS_CACHE=$root/datasets \
  HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-300} HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export TIKTOKEN_RS_CACHE_DIR=$HF_HOME/tiktoken-rs-cache
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
if [[ $commands == *python_only_compile.sh* ]]; then   # reinstalls from /opt/vllm-wheels; /vllm-workspace has no .git
  VLLM_VERSION_OVERRIDE=$(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))') || exit 1
  export VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE
fi

# DPX: one GPU partition per pod; a host-wide lock refuses a second job on the same device.
if [[ ${BUILDKITE_AGENT_META_DATA_QUEUE:-} == *dpx* && $gpus != 0 ]]; then
  dev=(/dev/dri/renderD*); [[ ${#dev[@]} == 1 ]] || { echo "DPX: expected one render device" >&2; exit 1; }
  mkdir -p "$HF_HOME/.vllm-dpx-locks"
  exec 9>>"$HF_HOME/.vllm-dpx-locks/$(stat -Lc %t-%T "${dev[0]}").lock"
  flock -n 9 || { echo "DPX GPU collision on ${dev[0]}" >&2; exit 1; }
fi

diagnostics() {   # small, always-on failure bundle for infra triage (step artifact_paths uploads it)
  local d=artifacts/amd-gpu-diagnostics/$job; mkdir -p "$d"
  { date; echo "node=${VLLM_CI_K8S_NODE_NAME:-?} pod=${VLLM_CI_K8S_POD_NAME:-?}"
    timeout 10 amd-smi list; timeout 10 rocm-smi --showpids; ps -eo pid,etime,rss,args --sort=-rss | head -30
    df -h /tmp /dev/shm "$HF_HOME"; free -g; ls -l /dev/kfd /dev/dri; } >"$d/diagnostics.log" 2>&1
}

# Preflight: the image imports and sees exactly the GPUs the step requested.
python3 -c "import vllm, torch, transformers, ray" || exit 1
if [[ $gpus != 0 ]] && ! { rocminfo >/dev/null && python3 -c "import torch; n = torch.cuda.device_count(); \
    assert torch.version.hip and n == $gpus, f'expected $gpus GPUs, found {n}'"; }; then
  diagnostics; exit 1
fi

# Orchestration vars stay out of vLLM's environment (other VLLM_CI_* are test knobs, see tests/ci_envs.py).
unset VLLM_TEST_COMMANDS VLLM_CI_EXPECTED_GPU_COUNT VLLM_CI_DIAGNOSTICS_DIR
bash -o pipefail -c "$commands"; rc=$?
[[ $rc == 5 ]] && rc=0                      # pytest: no tests collected
[[ $rc != 0 ]] && diagnostics
exit $rc
