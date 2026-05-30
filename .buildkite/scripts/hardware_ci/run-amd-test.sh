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

# Export Python path for commands that run directly on the host. Containerized
# tests set this to /vllm-workspace below so spawned Python processes do not
# depend on their current working directory.
export PYTHONPATH="${PYTHONPATH:-..}"

###############################################################################
# Helper Functions
###############################################################################

report_docker_usage() {
  echo "--- Docker usage"
  docker system df || true
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

handle_pytest_exit() {
  local exit_code=$1
  if [ "$exit_code" -eq 5 ]; then
    echo "Pytest exit code 5 (no tests collected) - treating as success."
    exit 0
  fi
  exit "$exit_code"
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
# through untouched to avoid double-quoting well-formed shell fragments.
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

  # Strip backslash-newline continuations, then flatten remaining newlines
  local flat="${input//$'\\\n'/ }"
  flat="${flat//$'\n'/ }"

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
        # Line-continuation artifact
        "\\")
          is_boundary=true ;;
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
        # Strip surrounding double quotes if present (from upstream
        # single-to-double conversion); without this, wrapping below
        # would produce '"expr"' with literal double-quote characters.
        if [[ "$marker_buf" == '"'*'"' ]]; then
          marker_buf="${marker_buf#\"}"
          marker_buf="${marker_buf%\"}"
        fi
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
        # Drop stray backslash tokens silently
        elif [[ "$word" == "\\" ]]; then
          :
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
    # Strip surrounding double quotes (see mid-stream flush comment)
    if [[ "$marker_buf" == '"'*'"' ]]; then
      marker_buf="${marker_buf#\"}"
      marker_buf="${marker_buf%\"}"
    fi
    if [[ "$marker_buf" == *" "* || "$marker_buf" == *"("* ]]; then
      output+="'${marker_buf}'"
    else
      output+="${marker_buf}"
    fi
  fi

  echo "${output% }"
}

###############################################################################
# Main
###############################################################################

# --- GPU initialization ---
echo "--- ROCm info"
rocminfo

# --- Docker status ---
report_docker_usage

# --- Pull test image ---
echo "--- Pulling container"
image_name="rocm/vllm-ci:${BUILDKITE_COMMIT}"
container_name="rocm_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"
docker pull "${image_name}"

remove_docker_container() {
  # docker run uses --rm, so the container is normally already gone when the
  # EXIT trap runs. Cleanup is best-effort and must not affect the test result.
  docker rm -f "${container_name}" >/dev/null 2>&1 || true
}

on_exit() {
  local exit_code=$?
  remove_docker_container
  exit "$exit_code"
}
trap on_exit EXIT

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
  commands_source="env"
  echo "Commands sourced from VLLM_TEST_COMMANDS (quoting preserved)"
else
  commands="$*"
  commands_source="argv"
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

# Only try to repair stripped pytest -m/-k quoting in legacy argv mode.
# VLLM_TEST_COMMANDS preserves inner quoting already, and re-quoting that path
# can corrupt embedded echo strings or otherwise well-formed shell fragments.
if [[ "$commands_source" == "argv" ]]; then
  commands=$(re_quote_pytest_markers "$commands")
  echo "After re-quoting: $commands"
else
  echo "Skipping re-quoting for VLLM_TEST_COMMANDS input"
fi

echo "Final commands: $commands"

MYPYTHONPATH="/vllm-workspace"

container_job_id="${BUILDKITE_JOB_ID:-${BUILDKITE_PARALLEL_JOB:-0}}"
container_job_id="${container_job_id//[^A-Za-z0-9_.-]/_}"
container_job_id_short="${container_job_id:0:8}"
# Keep TMPDIR short because Ray creates AF_UNIX sockets under it and Linux
# limits socket paths to 107 bytes.
CONTAINER_TMPDIR="/tmp/vllm-${container_job_id_short}"
CONTAINER_CACHE_ROOT="/tmp/vllm-buildkite-${container_job_id}/cache"
CONTAINER_PREFLIGHT="mkdir -p \"\$TMPDIR\" \"\$TORCHINDUCTOR_CACHE_DIR\" \"\$TRITON_CACHE_DIR\" \"\$VLLM_CACHE_ROOT\" \"\$XDG_CACHE_HOME\" && python -c \"import encodings, importlib.metadata as im, importlib.util as iu; [im.version(d) for d in ('transformers', 'torch', 'ray', 'sympy', 'markupsafe', 'vllm')]; missing=[m for m in ('torch.utils.model_zoo', 'transformers.models.nomic_bert', 'ray.dag', 'sympy.physics', 'markupsafe._speedups') if iu.find_spec(m) is None]; assert not missing, missing\""

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
    exit_code=$?
    cleanup_network
    handle_pytest_exit "$exit_code"
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
    -e BUILDKITE_PARALLEL_JOB \
    -e BUILDKITE_PARALLEL_JOB_COUNT \
    -v "${HF_CACHE}:${HF_MOUNT}" \
    -e "HF_HOME=${HF_MOUNT}" \
    -e "PYTHONPATH=${MYPYTHONPATH}" \
    -e "TMPDIR=${CONTAINER_TMPDIR}/tmp" \
    -e "TORCHINDUCTOR_CACHE_DIR=${CONTAINER_CACHE_ROOT}/torchinductor" \
    -e "TRITON_CACHE_DIR=${CONTAINER_CACHE_ROOT}/triton" \
    -e "VLLM_CACHE_ROOT=${CONTAINER_CACHE_ROOT}/vllm" \
    -e "XDG_CACHE_HOME=${CONTAINER_CACHE_ROOT}/xdg" \
    -e "PYTORCH_ROCM_ARCH=" \
    --name "${container_name}" \
    "${image_name}" \
    /bin/bash -c "${CONTAINER_PREFLIGHT} && ${commands}"

  exit_code=$?
  handle_pytest_exit "$exit_code"
fi
