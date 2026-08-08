#!/usr/bin/env bash

# Source-only helpers for selecting and applying the AMD Hugging Face client
# cache mode. This does not provide network isolation.

vllm_amd_hf_resolve_mode() {
  if [[ $# -ne 4 ]]; then
    echo "vllm_amd_hf_resolve_mode requires ENABLED, RETRY_COUNT," \
      "DISABLED, and INITIAL_ONLINE" >&2
    return 2
  fi

  local enabled=$1
  local retry_count=$2
  local disabled=$3
  local initial_online=$4

  if [[ "${enabled}" != "0" && "${enabled}" != "1" ]]; then
    echo "VLLM_CI_HF_OFFLINE_RETRY must be 0 or 1" >&2
    return 2
  fi
  if [[ ! "${retry_count}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "BUILDKITE_RETRY_COUNT must be a nonnegative integer" >&2
    return 2
  fi
  if [[ "${disabled}" != "0" && "${disabled}" != "1" ]]; then
    echo "VLLM_CI_DISABLE_HF_OFFLINE_RETRY must be 0 or 1" >&2
    return 2
  fi
  if [[ "${initial_online}" != "0" && "${initial_online}" != "1" ]]; then
    echo "AMD Hugging Face initial-online mode must be 0 or 1" >&2
    return 2
  fi

  if [[ "${enabled}" == "0" || "${disabled}" == "1" ]]; then
    printf '%s\n' disabled
  elif [[ "${retry_count}" != "0" || "${initial_online}" == "1" ]]; then
    printf '%s\n' online
  else
    printf '%s\n' cache-only
  fi
}

vllm_amd_hf_apply_mode() {
  if [[ $# -ne 1 ]]; then
    echo "vllm_amd_hf_apply_mode requires MODE" >&2
    return 2
  fi

  case "$1" in
    disabled)
      ;;
    cache-only)
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
      ;;
    online)
      unset HF_HUB_OFFLINE
      unset TRANSFORMERS_OFFLINE
      ;;
    *)
      echo "Unknown AMD Hugging Face client mode: $1" >&2
      return 2
      ;;
  esac
}

vllm_amd_hf_container_offline_value() {
  if [[ $# -ne 1 ]]; then
    echo "vllm_amd_hf_container_offline_value requires MODE" >&2
    return 2
  fi

  case "$1" in
    disabled)
      printf '%s\n' inherit
      ;;
    cache-only)
      printf '%s\n' 1
      ;;
    online)
      printf '%s\n' 0
      ;;
    *)
      echo "Unknown AMD Hugging Face client mode: $1" >&2
      return 2
      ;;
  esac
}
