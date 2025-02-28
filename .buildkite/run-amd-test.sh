#!/bin/bash

# This script runs test inside the corresponding ROCm docker container.
set -o pipefail

# Print ROCm version
echo "--- Confirming Clean Initial State"
while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

echo "--- ROCm info"
rocminfo

# cleanup older docker images
cleanup_docker() {
  # Get Docker's root directory
  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory."
    exit 1
  fi
  echo "Docker root directory: $docker_root"
  # Check disk usage of the filesystem where Docker's root directory is located
  disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
  # Define the threshold
  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
    # Remove dangling images (those that are not tagged and not used by any container)
    docker image prune -f
    # Remove unused volumes / force the system prune for old images as well.
    docker volume prune -f && docker system prune --force --filter "until=72h" --all
    echo "Docker images and volumes cleanup completed."
  else
    echo "Disk usage is below $threshold%. No cleanup needed."
  fi
}

# Call the cleanup docker function
cleanup_docker

echo "--- Resetting GPUs"

echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

echo "--- Pulling container" 
image_name="rocm/vllm-ci:${BUILDKITE_COMMIT}"
container_name="rocm_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"
docker pull "${image_name}"

remove_docker_container() {
   docker rm -f "${container_name}" || docker image rm -f "${image_name}" || true
}
trap remove_docker_container EXIT

echo "--- Running container"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

commands=$@
echo "Commands:$commands"
#ignore certain kernels tests
if [[ $commands == *" kernels "* ]]; then
  commands="${commands} \
  --ignore=kernels/test_attention.py \
  --ignore=kernels/test_attention_selector.py \
  --ignore=kernels/test_blocksparse_attention.py \
  --ignore=kernels/test_causal_conv1d.py \
  --ignore=kernels/test_cutlass.py \
  --ignore=kernels/test_encoder_decoder_attn.py \
  --ignore=kernels/test_flash_attn.py \
  --ignore=kernels/test_flashinfer.py \
  --ignore=kernels/test_int8_quant.py \
  --ignore=kernels/test_machete_gemm.py \
  --ignore=kernels/test_mamba_ssm.py \
  --ignore=kernels/test_marlin_gemm.py \
  --ignore=kernels/test_moe.py \
  --ignore=kernels/test_prefix_prefill.py \
  --ignore=kernels/test_rand.py \
  --ignore=kernels/test_sampler.py \
  --ignore=kernels/test_cascade_flash_attn.py \
  --ignore=kernels/test_mamba_mixer2.py"
fi

#ignore certain Entrypoints tests
if [[ $commands == *" entrypoints/openai "* ]]; then
  commands=${commands//" entrypoints/openai "/" entrypoints/openai \
  --ignore=entrypoints/openai/test_accuracy.py \
  --ignore=entrypoints/openai/test_audio.py \
  --ignore=entrypoints/openai/test_encoder_decoder.py \
  --ignore=entrypoints/openai/test_embedding.py \
  --ignore=entrypoints/openai/test_oot_registration.py "}
fi

PARALLEL_JOB_COUNT=8
# check if the command contains shard flag, we will run all shards in parallel because the host have 8 GPUs. 
if [[ $commands == *"--shard-id="* ]]; then
  # assign job count as the number of shards used   
  commands=${commands//"--num-shards= "/"--num-shards=${PARALLEL_JOB_COUNT} "}
  for GPU in $(seq 0 $(($PARALLEL_JOB_COUNT-1))); do
    # assign shard-id for each shard
    commands_gpu=${commands//"--shard-id= "/"--shard-id=${GPU} "}
    echo "Shard ${GPU} commands:$commands_gpu"
    docker run \
        --device /dev/kfd --device /dev/dri \
        --network host \
        --shm-size=16gb \
        --rm \
        -e HIP_VISIBLE_DEVICES="${GPU}" \
        -e HF_TOKEN \
        -e AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY \
        -v "${HF_CACHE}:${HF_MOUNT}" \
        -e "HF_HOME=${HF_MOUNT}" \
        --name "${container_name}_${GPU}" \
        "${image_name}" \
        /bin/bash -c "${commands_gpu}" \
        |& while read -r line; do echo ">>Shard $GPU: $line"; done &
    PIDS+=($!)
  done
  #wait for all processes to finish and collect exit codes
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
    STATUS+=($?)
  done
  for st in "${STATUS[@]}"; do
    if [[ ${st} -ne 0 ]]; then
      echo "One of the processes failed with $st"
      exit "${st}"
    fi
  done
else
  docker run \
          --device /dev/kfd --device /dev/dri \
          --network host \
          --shm-size=16gb \
          --rm \
          -e HIP_VISIBLE_DEVICES=0 \
          -e HF_TOKEN \
          -e AWS_ACCESS_KEY_ID \
          -e AWS_SECRET_ACCESS_KEY \
          -v "${HF_CACHE}:${HF_MOUNT}" \
          -e "HF_HOME=${HF_MOUNT}" \
          --name "${container_name}" \
          "${image_name}" \
          /bin/bash -c "${commands}"
fi
