#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Preserve the runner's original device constraint. The selected device is
# exported through ASCEND_RT_VISIBLE_DEVICES, so reading that variable on a
# later probe would otherwise prevent the gate from moving to another card.
ASCEND_E2E_CANDIDATE_DEVICES=${ASCEND_E2E_CANDIDATE_DEVICES-${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-}}}
NPU_MEMORY_EXIT_CODE=${NPU_MEMORY_EXIT_CODE:-87}

ascend_memory_status_line() {
  local status=$1
  local device_id=$2
  local free_memory_mb=$3
  local total_memory_mb=$4
  local required_free_memory_mb=$5
  local utilization=$6

  if [[ ! "$device_id" =~ ^[0-9]+$ || ! "$free_memory_mb" =~ ^[0-9]+$ || \
    ! "$total_memory_mb" =~ ^[0-9]+$ || ! "$required_free_memory_mb" =~ ^[0-9]+$ ]]; then
    return 0
  fi

  awk -v status="$status" -v device="$device_id" \
    -v free="$free_memory_mb" -v total="$total_memory_mb" \
    -v required="$required_free_memory_mb" -v utilization="$utilization" \
    'BEGIN {
      printf "ASCEND_NPU_MEMORY_STATUS=%s device=npu:%s free_gib=%.2f total_gib=%.2f required_gib=%.2f utilization=%.4f\n", status, device, free / 1024, total / 1024, required / 1024, utilization
    }'
}

select_ascend_e2e_device() {
  local workload_name=${1:-Ascend E2E test}
  local selector=${ASCEND_DEVICE_SELECTOR:?ASCEND_DEVICE_SELECTOR must be set}
  local min_free_ratio=${GPU_MEMORY_UTILIZATION:-0.92}
  local candidate_devices=${ASCEND_E2E_CANDIDATE_DEVICES:-}
  local selection_output=""
  local selector_status=0
  local status=""
  local device_id=""
  local source=""
  local free_memory_mb=""
  local total_memory_mb=""
  local required_free_memory_mb=""
  local -a selector_args=(
    --npu-smi-bin "${NPU_SMI_BIN:-npu-smi}"
    --min-free-ratio "$min_free_ratio"
    --format tsv
  )

  if [[ -n "$candidate_devices" ]]; then
    selector_args+=(--candidate-devices "$candidate_devices")
  fi

  if selection_output=$("$PYTHON_BIN" "$selector" "${selector_args[@]}"); then
    selector_status=0
  else
    selector_status=$?
  fi

  IFS=$'\t' read -r status device_id source free_memory_mb \
    total_memory_mb required_free_memory_mb _ <<<"$selection_output"

  if [[ "$selector_status" -eq 0 ]]; then
    if [[ "$status" != "selected" || ! "$device_id" =~ ^[0-9]+$ ]]; then
      echo "Ascend device selector returned an invalid success record: $selection_output" >&2
      return 1
    fi
    unset ASCEND_VISIBLE_DEVICES
    export ASCEND_RT_VISIBLE_DEVICES="$device_id"
    export ASCEND_E2E_SELECTED_DEVICE="$device_id"
    export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="npu:0"
    ascend_memory_status_line ok "$device_id" "$free_memory_mb" \
      "$total_memory_mb" "$required_free_memory_mb" "$min_free_ratio"
    echo "Selected Ascend device $device_id for $workload_name: free=${free_memory_mb}MiB total=${total_memory_mb}MiB required=${required_free_memory_mb}MiB source=$source"
    return 0
  fi

  if [[ "$selector_status" -eq 3 && "$status" == "unavailable" ]]; then
    local message="No healthy single Ascend device meets the ${min_free_ratio} free-HBM ratio required by $workload_name."
    ascend_memory_status_line insufficient "$device_id" "$free_memory_mb" \
      "$total_memory_mb" "$required_free_memory_mb" "$min_free_ratio" >&2
    echo "::error title=Ascend resource gate failed::$message" >&2
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
      printf '### Ascend resource gate\n\n- Status: failed\n- Reason: %s\n' "$message" >>"$GITHUB_STEP_SUMMARY"
    fi
    return "$NPU_MEMORY_EXIT_CODE"
  fi

  echo "Ascend device resource probe failed: ${selection_output:-no structured result}" >&2
  return 1
}

confirm_selected_ascend_e2e_device() {
  local workload_name=${1:-Ascend E2E test}
  local selector=${ASCEND_DEVICE_SELECTOR:?ASCEND_DEVICE_SELECTOR must be set}
  local selected_device=${ASCEND_E2E_SELECTED_DEVICE:?No Ascend device has been selected}
  local min_free_ratio=${GPU_MEMORY_UTILIZATION:-0.92}
  local selection_output=""
  local selector_status=0
  local status=""
  local device_id=""
  local source=""
  local free_memory_mb=""
  local total_memory_mb=""
  local required_free_memory_mb=""
  local -a selector_args=(
    --npu-smi-bin "${NPU_SMI_BIN:-npu-smi}"
    --min-free-ratio "$min_free_ratio"
    --candidate-devices "$selected_device"
    --format tsv
  )

  if selection_output=$("$PYTHON_BIN" "$selector" "${selector_args[@]}"); then
    selector_status=0
  else
    selector_status=$?
  fi

  IFS=$'\t' read -r status device_id source free_memory_mb \
    total_memory_mb required_free_memory_mb _ <<<"$selection_output"

  if [[ "$selector_status" -eq 0 && "$status" == "selected" && \
    "$device_id" == "$selected_device" ]]; then
    echo "Confirmed Ascend device $device_id for $workload_name: free=${free_memory_mb}MiB total=${total_memory_mb}MiB required=${required_free_memory_mb}MiB source=$source"
    return 0
  fi

  if [[ "$selector_status" -eq 3 && "$status" == "unavailable" ]]; then
    ascend_memory_status_line insufficient "$device_id" "$free_memory_mb" \
      "$total_memory_mb" "$required_free_memory_mb" "$min_free_ratio" >&2
    echo "Selected Ascend device $selected_device no longer meets the ${min_free_ratio} free-HBM ratio required by $workload_name." >&2
    return "$NPU_MEMORY_EXIT_CODE"
  fi

  echo "Ascend device confirmation probe failed for device $selected_device: ${selection_output:-no structured result}" >&2
  return 1
}

prepare_ascend_device_for_server() {
  local workload_name=${1:-Ascend E2E test}
  local max_attempts=${ASCEND_DEVICE_SELECTION_ATTEMPTS:-3}
  local attempt=1
  local status=0

  if [[ ! "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "ASCEND_DEVICE_SELECTION_ATTEMPTS must be a positive integer, got: $max_attempts" >&2
    return 1
  fi

  while [[ "$attempt" -le "$max_attempts" ]]; do
    if select_ascend_e2e_device "$workload_name"; then
      :
    else
      status=$?
      return "$status"
    fi

    if [[ "$ASCEND_E2E_USE_SUDO" == "1" ]]; then
      if wait_for_ascend_runtime_ready; then
        status=0
      else
        status=$?
      fi
    elif ensure_runner_npu_ready; then
      status=0
    else
      status=$?
    fi

    if [[ "$status" -ne 0 ]]; then
      return "$status"
    fi

    if confirm_selected_ascend_e2e_device "$workload_name launch"; then
      return 0
    else
      status=$?
    fi

    if [[ "$status" -ne "$NPU_MEMORY_EXIT_CODE" ]]; then
      return "$status"
    fi
    if [[ "$attempt" -eq "$max_attempts" ]]; then
      break
    fi
    echo "Ascend device availability changed during preflight; selecting again (${attempt}/${max_attempts})." >&2
    attempt=$((attempt + 1))
  done

  echo "Ascend resource gate could not keep an eligible device through preflight after ${max_attempts} attempts." >&2
  return "$NPU_MEMORY_EXIT_CODE"
}
