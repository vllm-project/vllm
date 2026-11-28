#!/bin/bash
# Started once per node by serve_multinode.sbatch. The node rank is SLURM_NODEID.
set -euo pipefail

CONTAINER_CMD=${CONTAINER_CMD:-apptainer}  # or singularity

# Pass a variable into the container. --env splits values on commas, so use the
# environment prefixes instead (Apptainer and SingularityCE read different ones).
pass_env() {
  export "APPTAINERENV_$1=$2" "SINGULARITYENV_$1=$2"
}

# Keep TMPDIR short: vLLM puts ZeroMQ IPC sockets there and Unix socket paths
# are limited to ~107 bytes, which deep per-job scratch paths can exceed.
mkdir -p "/tmp/vllm-$SLURM_JOB_ID"
pass_env TMPDIR "/tmp/vllm-$SLURM_JOB_ID"
# Ignore packages in ~/.local: the container shares $HOME with the host, and a
# user-site install of the same Python version would shadow the image's packages.
pass_env PYTHONNOUSERSITE 1
# Compute nodes often have no internet access: download weights beforehand.
pass_env HF_HOME "$WORKDIR/hf_cache"
pass_env HF_HUB_OFFLINE 1
# Keep compile caches out of $HOME, which is often small or slow on clusters.
pass_env VLLM_CACHE_ROOT "$WORKDIR/cache/vllm"
pass_env XDG_CACHE_HOME "$WORKDIR/cache/xdg"
# Triton defaults to ~/.triton, even when VLLM_CACHE_ROOT is set. Keep it node-local:
# with a cold cache on a shared filesystem, workers on several nodes compiling the
# same kernels can read each other's half-written cache files (JSONDecodeError).
pass_env TRITON_CACHE_DIR "/tmp/vllm-$SLURM_JOB_ID/triton"
# Optional: pin NCCL/RCCL and Gloo to the high-speed interconnect, e.g.
#   pass_env NCCL_SOCKET_IFNAME hsn0,hsn1,hsn2,hsn3
#   pass_env GLOO_SOCKET_IFNAME hsn0

# AMD GPUs: Slurm may expose the allocation as ROCR_VISIBLE_DEVICES. Importing Ray
# (if installed) raises on that variable; express the same GPUs as HIP_VISIBLE_DEVICES.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ] && [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
  unset ROCR_VISIBLE_DEVICES
fi

HEADLESS=()
[ "$SLURM_NODEID" = 0 ] || HEADLESS=(--headless)
# Split the optional flag strings into arrays (they may hold several flags).
read -r -a GPU_FLAGS <<< "${GPU_FLAG:-}"
read -r -a EXTRA_ARGS <<< "${VLLM_ARGS:-}"

# GPU_FLAG: --nv on NVIDIA. On AMD the devices are usually visible without --rocm,
# which injects host libraries that may need a newer glibc than the image has.
exec "$CONTAINER_CMD" exec "${GPU_FLAGS[@]}" --bind "$WORKDIR" "$SIF" \
  vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
    --nnodes "$NNODES" --node-rank "$SLURM_NODEID" \
    --master-addr "$HEAD" --master-port "$MASTER_PORT" \
    --port "$API_PORT" "${EXTRA_ARGS[@]}" "${HEADLESS[@]}"
