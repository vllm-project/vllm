#!/bin/bash

# This script runs tests inside the corresponding ROCm docker container.
# It handles both single-node and multi-node test configurations.
#
# Multi-node detection: Instead of matching on fragile group names, we detect
# multi-node jobs structurally by looking for the bracket command syntax
# "[node0_cmds] && [node1_cmds]" or via the NUM_NODES environment variable.
#
###############################################################################
# QUOTING / COMMAND PASSING
#
# Passing commands as positional arguments ($*) is fragile when the command
# string itself contains double quotes, e.g.:
#
#   bash run-amd-test.sh "export FLAGS="value" && pytest -m "not slow""
#
# The outer shell resolves the nested quotes *before* this script runs, so
# the script receives mangled input it cannot fully recover.
#
# Preferred: pass commands via the VLLM_TEST_COMMANDS environment variable:
#
#   export VLLM_TEST_COMMANDS='export FLAGS="value" && pytest -m "not slow"'
#   bash run-amd-test.sh
#
# Single-quoted assignment preserves all inner double quotes verbatim.
# The $* path is kept for backward compatibility but callers should migrate.
###############################################################################
set -o pipefail

# Export Python path
export PYTHONPATH=".."

###############################################################################
# Helper Functions
###############################################################################

wait_for_clean_gpus() {
  local timeout=${1:-300}
  local start=$SECONDS
  echo "--- Waiting for clean GPU state (timeout: ${timeout}s)"
  while true; do
    if grep -q clean /opt/amdgpu/etc/gpu_state; then
      echo "GPUs state is \"clean\""
      return
    fi
    if (( SECONDS - start >= timeout )); then
      echo "Error: GPUs did not reach clean state within ${timeout}s" >&2
      exit 1
    fi
    sleep 3
  done
}

cleanup_docker() {
  # Get Docker's root directory
  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory."
    exit 1
  fi
  echo "Docker root directory: $docker_root"

  disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
    docker image prune -f
    docker volume prune -f && docker system prune --force --filter "until=72h" --all
    echo "Docker images and volumes cleanup completed."
  else
    echo "Disk usage is below $threshold%. No cleanup needed."
  fi
}

cleanup_network() {
  local max_nodes=${NUM_NODES:-2}
  for node in $(seq 0 $((max_nodes - 1))); do
    if docker ps -a -q -f name="node${node}" | grep -q .; then
      docker stop "node${node}" || true
    fi
  done
  if docker network ls | grep -q docker-net; then
    docker network rm docker-net || true
  fi
}

is_multi_node() {
  local cmds="$1"
  # Primary signal: NUM_NODES environment variable set by the pipeline
  if [[ "${NUM_NODES:-1}" -gt 1 ]]; then
    return 0
  fi
  # Fallback: detect the bracket syntax structurally
  # Pattern: [...] && [...] (per-node command arrays)
  if [[ "$cmds" =~ \[.*\].*\&\&.*\[.*\] ]]; then
    return 0
  fi
  return 1
}

###############################################################################
# Pytest marker/keyword re-quoting
#
# When commands are passed through Buildkite -> shell -> $* -> bash -c,
# quotes around multi-word pytest -m/-k expressions get stripped:
#   pytest -v -s -m 'not cpu_test' v1/core
# becomes:
#   pytest -v -s -m not cpu_test v1/core
#
# pytest then interprets "cpu_test" as a file path, not part of the marker.
#
# This function detects unquoted expressions after -m/-k and re-quotes them
# by collecting tokens until a recognizable boundary is reached:
#   - test path (contains '/')
#   - test file (ends with '.py')
#   - another pytest flag (--xxx or -x single-char flags)
#   - command separator (&& || ; |)
#   - environment variable assignment (FOO=bar)
#
# Single-word markers (e.g. -m cpu_test, -m hybrid_model) pass through
# unquoted since they have no spaces and work fine.
#
# Already-quoted expressions (containing literal single quotes) are passed
# through untouched to avoid double-quoting values injected by
# apply_rocm_test_overrides.
#
# NOTE: This ONLY fixes -m/-k flags. It cannot recover arbitrary inner
# double-quotes stripped by the calling shell (see header comment).
# Use VLLM_TEST_COMMANDS to avoid the problem entirely.
###############################################################################
re_quote_pytest_markers() {
  local input="$1"
  local output=""
  local collecting=false
  local marker_buf=""

  # Flatten newlines for consistent tokenization
  local flat="${input//$'\n'/ }"

  # Disable globbing to prevent *.py etc. from expanding during read -ra
  local restore_glob
  restore_glob="$(shopt -p -o noglob 2>/dev/null || true)"
  set -o noglob
  local -a words
  read -ra words <<< "$flat"
  eval "$restore_glob"

  for word in "${words[@]}"; do
    if $collecting; then
      # If the token we're about to collect already contains a literal
      # single quote, the expression was already quoted upstream.
      # Flush and stop collecting.
      if [[ "$word" == *"'"* ]]; then
        if [[ -n "$marker_buf" ]]; then
          # Should not normally happen (partial buf + quote), flush raw
          output+="${marker_buf} "
          marker_buf=""
        fi
        output+="${word} "
        collecting=false
        continue
      fi

      local is_boundary=false
      case "$word" in
        # Command separators
        "&&"|"||"|";"|"|")
          is_boundary=true ;;
        # Long flags (--ignore, --shard-id, etc.)
        --*)
          is_boundary=true ;;
        # Short flags (-v, -s, -x, etc.) but NOT negative marker tokens
        # like "not" which don't start with "-". Also skip -k/-m which
        # would start a new marker (handled below).
        -[a-zA-Z])
          is_boundary=true ;;
        # Test path (contains /)
        */*)
          is_boundary=true ;;
        # Test file (ends with .py, possibly with ::method)
        *.py|*.py::*)
          is_boundary=true ;;
        # Environment variable assignment preceding a command (FOO=bar)
        *=*)
          # Only treat as boundary if it looks like VAR=value, not
          # pytest filter expressions like num_gpus=2 inside markers
          if [[ "$word" =~ ^[A-Z_][A-Z0-9_]*= ]]; then
            is_boundary=true
          fi
          ;;
      esac

      if $is_boundary; then
        # Flush the collected marker expression
        if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
          output+="'${marker_buf}' "
        else
          output+="${marker_buf} "
        fi
        collecting=false
        marker_buf=""
        # Check if this boundary word itself starts a new -m/-k
        if [[ "$word" == "-m" || "$word" == "-k" ]]; then
          output+="${word} "
          collecting=true
        else
          output+="${word} "
        fi
      else
        # Accumulate into marker buffer
        if [[ -n "$marker_buf" ]]; then
          marker_buf+=" ${word}"
        else
          marker_buf="${word}"
        fi
      fi
    elif [[ "$word" == "-m" || "$word" == "-k" ]]; then
      output+="${word} "
      collecting=true
      marker_buf=""
    else
      output+="${word} "
    fi
  done

  # Flush any trailing marker expression (marker at end of command)
  if $collecting && [[ -n "$marker_buf" ]]; then
    if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
      output+="'${marker_buf}'"
    else
      output+="${marker_buf}"
    fi
  fi

  echo "${output% }"
}

###############################################################################
# ROCm-specific pytest command rewrites
#
# These apply ignore flags and environment overrides for tests that are not
# yet supported or behave differently on ROCm hardware. Kept as a single
# function so new exclusions are easy to add in one place.
###############################################################################

apply_rocm_test_overrides() {
  local cmds="$1"

  # --- Model registry filter ---
  if [[ $cmds == *"pytest -v -s models/test_registry.py"* ]]; then
    cmds=${cmds//"pytest -v -s models/test_registry.py"/"pytest -v -s models/test_registry.py -k 'not BambaForCausalLM and not GritLM and not Mamba2ForCausalLM and not Zamba2ForCausalLM'"}
  fi

  # --- LoRA: disable custom paged attention ---
  if [[ $cmds == *"pytest -v -s lora"* ]]; then
    cmds=${cmds//"pytest -v -s lora"/"VLLM_ROCM_CUSTOM_PAGED_ATTN=0 pytest -v -s lora"}
  fi

  # --- Kernel ignores ---
  if [[ $cmds == *" kernels/core"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/core/test_fused_quant_layernorm.py \
    --ignore=kernels/core/test_permute_cols.py"
  fi

  if [[ $cmds == *" kernels/attention"* ]]; then
    cmds="${cmds} \
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

  if [[ $cmds == *" kernels/quantization"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/quantization/test_int8_quant.py \
    --ignore=kernels/quantization/test_machete_mm.py \
    --ignore=kernels/quantization/test_block_fp8.py \
    --ignore=kernels/quantization/test_block_int8.py \
    --ignore=kernels/quantization/test_marlin_gemm.py \
    --ignore=kernels/quantization/test_cutlass_scaled_mm.py \
    --ignore=kernels/quantization/test_int8_kernel.py"
  fi

  if [[ $cmds == *" kernels/mamba"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/mamba/test_mamba_mixer2.py \
    --ignore=kernels/mamba/test_causal_conv1d.py \
    --ignore=kernels/mamba/test_mamba_ssm_ssd.py"
  fi

  if [[ $cmds == *" kernels/moe"* ]]; then
    cmds="${cmds} \
    --ignore=kernels/moe/test_moe.py \
    --ignore=kernels/moe/test_cutlass_moe.py \
    --ignore=kernels/moe/test_triton_moe_ptpc_fp8.py"
  fi

  # --- Entrypoint ignores ---
  if [[ $cmds == *" entrypoints/openai "* ]]; then
    cmds=${cmds//" entrypoints/openai "/" entrypoints/openai \
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

  if [[ $cmds == *" entrypoints/llm "* ]]; then
    cmds=${cmds//" entrypoints/llm "/" entrypoints/llm \
    --ignore=entrypoints/llm/test_chat.py \
    --ignore=entrypoints/llm/test_accuracy.py \
    --ignore=entrypoints/llm/test_init.py \
    --ignore=entrypoints/llm/test_prompt_validation.py "}
  fi

  # Clean up escaped newlines from --ignore appends
  cmds=$(echo "$cmds" | sed 's/ \\ / /g')

  echo "$cmds"
}

###############################################################################
# Main
###############################################################################

# --- GPU initialization ---
echo "--- Confirming Clean Initial State"
wait_for_clean_gpus

echo "--- ROCm info"
rocminfo

# --- Docker housekeeping ---
cleanup_docker

echo "--- Resetting GPUs"
echo "reset" > /opt/amdgpu/etc/gpu_state
wait_for_clean_gpus

# --- Pull test image ---
echo "--- Pulling container"
image_name="rocm/vllm-ci:${BUILDKITE_COMMIT}"
docker pull "${image_name}"

remove_docker_container() {
  docker rm -f "${container_name}" || docker image rm -f "${image_name}" || true
}
trap remove_docker_container EXIT

# --- Prepare commands ---
echo "--- Running container"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

# ---- Command source selection ----
# Prefer VLLM_TEST_COMMANDS (preserves all inner quoting intact).
# Fall back to $* for backward compatibility, but warn that inner
# double-quotes will have been stripped by the calling shell.
if [[ -n "${VLLM_TEST_COMMANDS:-}" ]]; then
  commands="${VLLM_TEST_COMMANDS}"
  echo "Commands sourced from VLLM_TEST_COMMANDS (quoting preserved)"
else
  commands="$*"
  if [[ -z "$commands" ]]; then
    echo "Error: No test commands provided." >&2
    echo "Usage:" >&2
    echo "  Preferred:  VLLM_TEST_COMMANDS='...' bash $0" >&2
    echo "  Legacy:     bash $0 \"commands here\"" >&2
    exit 1
  fi
  echo "Commands sourced from positional args (legacy mode)"
  echo "WARNING: Inner double-quotes in the command string may have been"
  echo "  stripped by the calling shell. If you see syntax errors, switch to:"
  echo "  export VLLM_TEST_COMMANDS='your commands here'"
  echo "  bash $0"
fi

echo "Raw commands: $commands"

# Fix quoting before ROCm overrides (so overrides see correct structure)
commands=$(re_quote_pytest_markers "$commands")
echo "After re-quoting: $commands"

commands=$(apply_rocm_test_overrides "$commands")
echo "Final commands: $commands"

MYPYTHONPATH=".."

# Verify GPU access
render_gid=$(getent group render | cut -d: -f3)
if [[ -z "$render_gid" ]]; then
  echo "Error: 'render' group not found. This is required for GPU access." >&2
  exit 1
fi

# --- RDMA device passthrough (conditional) ---
# If the host has RDMA devices, pass them through so tests like
# test_moriio_connector can access ibverbs. On hosts without RDMA
# hardware the tests will gracefully skip via _rdma_available().
RDMA_FLAGS=""
if [ -d /dev/infiniband ]; then
  echo "RDMA devices detected on host, enabling passthrough"
  RDMA_FLAGS="--device /dev/infiniband --cap-add=IPC_LOCK"
else
  echo "No RDMA devices found on host, RDMA tests will be skipped"
fi

# --- Route: multi-node vs single-node ---
if is_multi_node "$commands"; then
  echo "--- Multi-node job detected"
  export DCKR_VER=$(docker --version | sed 's/Docker version \(.*\), build .*/\1/')

  # Parse the bracket syntax:  prefix ; [node0_cmds] && [node1_cmds]
  #   BASH_REMATCH[1] = prefix (everything before first bracket)
  #   BASH_REMATCH[2] = comma-separated node0 commands
  #   BASH_REMATCH[3] = comma-separated node1 commands
  if [[ "$commands" =~ ^(.*)\[(.*)"] && ["(.*)\]$ ]]; then
    prefix=$(echo "${BASH_REMATCH[1]}" | sed 's/;//g')
    echo "PREFIX: ${prefix}"

    export composite_command="(command rocm-smi || true)"
    saved_IFS=$IFS
    IFS=','
    read -ra node0 <<< "${BASH_REMATCH[2]}"
    read -ra node1 <<< "${BASH_REMATCH[3]}"
    IFS=$saved_IFS

    if [[ ${#node0[@]} -ne ${#node1[@]} ]]; then
      echo "Warning: node0 has ${#node0[@]} commands, node1 has ${#node1[@]}. They will be paired by index."
    fi

    for i in "${!node0[@]}"; do
      command_node_0=$(echo "${node0[i]}" | sed 's/\"//g')
      command_node_1=$(echo "${node1[i]}" | sed 's/\"//g')

      step_cmd="./.buildkite/scripts/run-multi-node-test.sh /vllm-workspace/tests 2 2 ${image_name} '${command_node_0}' '${command_node_1}'"
      echo "COMMANDS: ${step_cmd}"
      composite_command="${composite_command} && ${step_cmd}"
    done

    /bin/bash -c "${composite_command}"
    cleanup_network
  else
    echo "Multi-node job detected but failed to parse bracket command syntax."
    echo "Expected format: prefix ; [node0_cmd1, node0_cmd2] && [node1_cmd1, node1_cmd2]"
    echo "Got: $commands"
    cleanup_network
    exit 111
  fi
else
  echo "--- Single-node job"
  echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"
  docker run \
    --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
    $RDMA_FLAGS \
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
