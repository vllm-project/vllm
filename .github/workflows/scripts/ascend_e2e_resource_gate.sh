#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Preserve the runner's original device constraint. The selected device is
# exported through ASCEND_RT_VISIBLE_DEVICES, so reading that variable on a
# later probe would otherwise prevent the gate from moving to another card.
ASCEND_E2E_CANDIDATE_DEVICES=${ASCEND_E2E_CANDIDATE_DEVICES-${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-}}}

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
    echo "Selected Ascend device $device_id for $workload_name: free=${free_memory_mb}MiB total=${total_memory_mb}MiB required=${required_free_memory_mb}MiB source=$source"
    return 0
  fi

  if [[ "$selector_status" -eq 3 && "$status" == "unavailable" ]]; then
    local message="No healthy single Ascend device meets the ${min_free_ratio} free-HBM ratio required by $workload_name."
    echo "::error title=Ascend resource gate failed::$message" >&2
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
      printf '### Ascend resource gate\n\n- Status: failed\n- Reason: %s\n' "$message" >>"$GITHUB_STEP_SUMMARY"
    fi
    return 1
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
    echo "Selected Ascend device $selected_device no longer meets the ${min_free_ratio} free-HBM ratio required by $workload_name." >&2
    return 1
  fi

  echo "Ascend device confirmation probe failed for device $selected_device: ${selection_output:-no structured result}" >&2
  return 1
}
