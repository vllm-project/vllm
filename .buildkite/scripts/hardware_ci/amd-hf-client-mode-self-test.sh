#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=amd-hf-client-mode.sh
source "${script_dir}/amd-hf-client-mode.sh"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

assert_equal() {
  local expected=$1
  local actual=$2
  local description=$3

  if [[ "${actual}" != "${expected}" ]]; then
    printf 'FAIL: %s\n  expected: %q\n  actual:   %q\n' \
      "${description}" "${expected}" "${actual}" >&2
    exit 1
  fi
}

assert_mode() {
  local expected=$1
  local enabled=$2
  local retry_count=$3
  local disabled=$4
  local initial_online=$5
  local actual
  local context="enabled=${enabled}, retry_count=${retry_count}, disabled=${disabled}, initial_online=${initial_online}"

  actual=$(
    vllm_amd_hf_resolve_mode \
      "${enabled}" "${retry_count}" "${disabled}" "${initial_online}"
  )
  assert_equal "${expected}" "${actual}" "${context}"
}

assert_resolve_error() {
  local enabled=$1
  local retry_count=$2
  local disabled=$3
  local initial_online=$4
  local status
  local context="invalid enabled=${enabled}, retry_count=${retry_count}, disabled=${disabled}, initial_online=${initial_online}"

  set +e
  vllm_amd_hf_resolve_mode \
    "${enabled}" "${retry_count}" "${disabled}" \
    "${initial_online}" >/dev/null 2>&1
  status=$?
  set -e
  assert_equal 2 "${status}" "${context}"
}

environment_snapshot() {
  bash -c 'printf "%s|%s|%s\n" \
    "${HF_HUB_OFFLINE-unset}" \
    "${TRANSFORMERS_OFFLINE-unset}" \
    "${HF_DATASETS_OFFLINE-unset}"'
}

assert_mode disabled 0 0 0 0
assert_mode disabled 0 3 0 1
assert_mode cache-only 1 0 0 0
assert_mode online 1 0 0 1
assert_mode online 1 1 0 0
assert_mode online 1 1 0 1
assert_mode online 1 12 0 0
assert_mode disabled 1 0 1 0
assert_mode disabled 1 3 1 1

assert_equal inherit \
  "$(vllm_amd_hf_container_offline_value disabled)" \
  "disabled containers inherit their baseline"
assert_equal 1 \
  "$(vllm_amd_hf_container_offline_value cache-only)" \
  "initial containers force cache-only clients"
assert_equal 0 \
  "$(vllm_amd_hf_container_offline_value online)" \
  "online containers force online clients"

assert_resolve_error "" 0 0 0
assert_resolve_error 2 0 0 0
assert_resolve_error true 0 0 0
assert_resolve_error 1 "" 0 0
assert_resolve_error 1 -1 0 0
assert_resolve_error 1 01 0 0
assert_resolve_error 1 +1 0 0
assert_resolve_error 1 1.5 0 0
assert_resolve_error 1 0 "" 0
assert_resolve_error 1 0 2 0
assert_resolve_error 1 0 true 0
assert_resolve_error 1 0 0 ""
assert_resolve_error 1 0 0 2
assert_resolve_error 1 0 0 true

export HF_HUB_OFFLINE=inherited-hub
export TRANSFORMERS_OFFLINE=inherited-transformers
export HF_DATASETS_OFFLINE=inherited-datasets
vllm_amd_hf_apply_mode disabled
assert_equal "inherited-hub|inherited-transformers|inherited-datasets" \
  "$(environment_snapshot)" "disabled mode preserves inherited environment"

vllm_amd_hf_apply_mode cache-only
assert_equal "1|1|inherited-datasets" "$(environment_snapshot)" \
  "cache-only mode changes only owned client variables"

vllm_amd_hf_apply_mode online
assert_equal "unset|unset|inherited-datasets" "$(environment_snapshot)" \
  "online mode clears only owned client variables"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
vllm_amd_hf_apply_mode disabled
assert_equal "unset|unset|unset" "$(environment_snapshot)" \
  "disabled mode preserves an unset environment"

if vllm_amd_hf_apply_mode invalid >/dev/null 2>&1; then
  fail "invalid apply mode succeeded"
fi
if vllm_amd_hf_container_offline_value invalid >/dev/null 2>&1; then
  fail "invalid container mode succeeded"
fi

echo "PASS: AMD Hugging Face client mode"
