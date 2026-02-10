#!/bin/bash

# This script runs test inside the corresponding ROCm docker container.
set -o pipefail

# Export Python path
export PYTHONPATH=".."

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

cleanup_network() {
  for node in $(seq 0 $((NUM_NODES-1))); do
    if docker pr -a -q -f name="node${node}" | grep -q .; then
      docker stop "node${node}"
    fi
  done
  if docker network ls | grep docker-net; then
    docker network rm docker-net
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
echo "Raw commands: $commands"

commands=${commands//"pytest -v -s basic_correctness/test_basic_correctness.py"/"pytest -v -s basic_correctness/test_basic_correctness.py"}

if [[ $commands == *"pytest -v -s models/test_registry.py"* ]]; then
  commands=${commands//"pytest -v -s models/test_registry.py"/"pytest -v -s models/test_registry.py -k 'not BambaForCausalLM and not GritLM and not Mamba2ForCausalLM and not Zamba2ForCausalLM'"}
fi

commands=${commands//"pytest -v -s compile/test_basic_correctness.py"/"pytest -v -s compile/test_basic_correctness.py"}

if [[ $commands == *"pytest -v -s lora"* ]]; then
  commands=${commands//"pytest -v -s lora"/"VLLM_ROCM_CUSTOM_PAGED_ATTN=0 pytest -v -s lora"}
fi

#ignore certain kernels tests
if [[ $commands == *" kernels/core"* ]]; then
  commands="${commands} \
  --ignore=kernels/core/test_fused_quant_layernorm.py \
  --ignore=kernels/core/test_permute_cols.py"
fi

if [[ $commands == *" kernels/attention"* ]]; then
  commands="${commands} \
  --ignore=kernels/attention/test_attention_selector.py \
  --ignore=kernels/attention/test_encoder_decoder_attn.py \
  --ignore=kernels/attention/test_flash_attn.py \
  --ignore=kernels/attention/test_flashinfer.py \
  --ignore=kernels/attention/test_prefix_prefill.py \
  --ignore=kernels/attention/test_cascade_flash_attn.py \
  --ignore=kernels/attention/test_mha_attn.py \
  --ignore=kernels/attention/test_lightning_attn.py \
  --ignore=kernels/attention/test_attention.py"
fi

if [[ $commands == *" kernels/quantization"* ]]; then
  commands="${commands} \
  --ignore=kernels/quantization/test_int8_quant.py \
  --ignore=kernels/quantization/test_machete_mm.py \
  --ignore=kernels/quantization/test_block_fp8.py \
  --ignore=kernels/quantization/test_block_int8.py \
  --ignore=kernels/quantization/test_marlin_gemm.py \
  --ignore=kernels/quantization/test_cutlass_scaled_mm.py \
  --ignore=kernels/quantization/test_int8_kernel.py"
fi

if [[ $commands == *" kernels/mamba"* ]]; then
  commands="${commands} \
  --ignore=kernels/mamba/test_mamba_mixer2.py \
  --ignore=kernels/mamba/test_causal_conv1d.py \
  --ignore=kernels/mamba/test_mamba_ssm_ssd.py"
fi

if [[ $commands == *" kernels/moe"* ]]; then
  commands="${commands} \
  --ignore=kernels/moe/test_moe.py \
  --ignore=kernels/moe/test_cutlass_moe.py \
  --ignore=kernels/moe/test_triton_moe_ptpc_fp8.py"
fi

#ignore certain Entrypoints/openai tests
if [[ $commands == *" entrypoints/openai "* ]]; then
  commands=${commands//" entrypoints/openai "/" entrypoints/openai \
  --ignore=entrypoints/openai/test_audio.py \
  --ignore=entrypoints/openai/test_shutdown.py \
  --ignore=entrypoints/openai/test_completion.py \
  --ignore=entrypoints/openai/test_models.py \
  --ignore=entrypoints/openai/test_lora_adapters.py \
  --ignore=entrypoints/openai/test_return_tokens_as_ids.py \
  --ignore=entrypoints/openai/test_root_path.py \
  --ignore=entrypoints/openai/test_tokenization.py \
  --ignore=entrypoints/openai/test_prompt_validation.py "}
fi

#ignore certain Entrypoints/llm tests
if [[ $commands == *" entrypoints/llm "* ]]; then
  commands=${commands//" entrypoints/llm "/" entrypoints/llm \
  --ignore=entrypoints/llm/test_chat.py \
  --ignore=entrypoints/llm/test_accuracy.py \
  --ignore=entrypoints/llm/test_init.py \
  --ignore=entrypoints/llm/test_prompt_validation.py "}
fi

commands=$(echo "$commands" | sed 's/ \\ / /g')
echo "Final commands: $commands"

# --ignore=entrypoints/openai/test_encoder_decoder.py \
# --ignore=entrypoints/openai/test_embedding.py \
# --ignore=entrypoints/openai/test_oot_registration.py
# --ignore=entrypoints/openai/test_accuracy.py \
# --ignore=entrypoints/openai/test_models.py <= Fails on MI250 but passes on MI300 as of 2025-03-13


MYPYTHONPATH=".."

# Test that we're launching on the machine that has
# proper access to GPUs
render_gid=$(getent group render | cut -d: -f3)
if [[ -z "$render_gid" ]]; then
  echo "Error: 'render' group not found. This is required for GPU access." >&2
  exit 1
fi

if [[ $commands == *"VLLM_TEST_GROUP_NAME=mi325_4-2-node-tests-4-gpus-in-total"* ]]; then

  export DCKR_VER=$(docker --version | sed 's/Docker version \(.*\), build .*/\1/')

  if [[ "$commands" =~ ^(.*)"["(.*)"] && ["(.*)"]"$ ]]; then
      prefix=$( echo "${BASH_REMATCH[1]}" | sed 's/;//g')
      echo "PREFIX: ${prefix}"
      export composite_command="(command rocm-smi || true)"
      myIFS=$IFS
      IFS=','
      read -ra node0 <<< ${BASH_REMATCH[2]}
      read -ra node1 <<< ${BASH_REMATCH[3]}
      IFS=$myIFS
      for i in "${!node0[@]}";do 
        command_node_0=$(echo ${node0[i]} | sed 's/\"//g')
        command_node_1=$(echo ${node1[i]} | sed 's/\"//g')
        
        export commands="./.buildkite/scripts/run-multi-node-test.sh /vllm-workspace/tests 2 2 ${image_name} '${command_node_0}' '${command_node_1}'"
        echo "COMMANDS: ${commands}"
        composite_command=$(echo "${composite_command} && ${commands}")
      done
      /bin/bash -c "${composite_command}"
      cleanup_network
  else
      echo "Failed to parse node commands! Exiting."
      cleanup_network
      exit 111
  fi
else
  echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"
  docker run \
          --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
          --group-add $(getent group render | cut -d: -f3) \
          --network=host \
          --shm-size=16gb \
          --group-add "$render_gid" \
          --rm \
          -e HF_TOKEN \
          -e AWS_ACCESS_KEY_ID \
          -e AWS_SECRET_ACCESS_KEY \
          -v "${HF_CACHE}:${HF_MOUNT}" \
          -e "HF_HOME=${HF_MOUNT}" \
          -e "PYTHONPATH=${MYPYTHONPATH}" \
          --name "${container_name}" \
          "${image_name}" \
          /bin/bash -c "${commands}"
fi
