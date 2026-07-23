#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
helper="${script_dir}/hf-offline-retry.sh"
test_root=$(mktemp -d "${TMPDIR:-/tmp}/vllm-hf-offline-retry-test.XXXXXX")

cleanup() {
  rm -rf -- "${test_root}"
}
trap cleanup EXIT

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

status_command_file="${test_root}/status-command.sh"
cat >"${status_command_file}" <<'VLLM_TEST_COMMAND'
if [[ "${HF_HUB_OFFLINE-}" == "1" &&
      "${TRANSFORMERS_OFFLINE-}" == "1" &&
      "${HF_DATASETS_OFFLINE-}" == "1" ]]; then
  printf '%s\n' offline >>"${ATTEMPTS_FILE}"
  exit "${OFFLINE_STATUS}"
fi

if [[ -z "${HF_HUB_OFFLINE+x}" &&
      -z "${TRANSFORMERS_OFFLINE+x}" &&
      -z "${HF_DATASETS_OFFLINE+x}" ]]; then
  printf '%s\n' online >>"${ATTEMPTS_FILE}"
  exit "${ONLINE_STATUS}"
fi

exit 91
VLLM_TEST_COMMAND

run_status_case() {
  local name=$1
  local offline_status=$2
  local online_status=$3
  local expected_status=$4
  local expected_attempts=$5
  local attempts_file="${test_root}/${name}.attempts"
  local actual_status
  local actual_attempts

  set +e
  ATTEMPTS_FILE="${attempts_file}" \
    OFFLINE_STATUS="${offline_status}" \
    ONLINE_STATUS="${online_status}" \
    bash "${helper}" "${status_command_file}"
  actual_status=$?
  set -e

  actual_attempts=$(<"${attempts_file}")
  assert_equal "${expected_status}" "${actual_status}" "${name} exit status"
  assert_equal "${expected_attempts}" "${actual_attempts}" "${name} attempts"
}

run_status_case offline_success 0 99 0 "offline"
run_status_case retry_status_1 1 0 0 $'offline\nonline'
run_status_case retry_status_2 2 0 0 $'offline\nonline'
run_status_case retry_status_123 123 0 0 $'offline\nonline'
run_status_case no_retry_status_42 42 0 42 "offline"
run_status_case online_failure 1 7 7 $'offline\nonline'

transport_file="${test_root}/transport.out"
transport_command_file="${test_root}/transport-command.sh"
cat >"${transport_command_file}" <<'VLLM_TRANSPORT_COMMAND'
cat >"${OUTPUT_FILE}" <<'INNER_PAYLOAD'
literal:$HOME
quotes:'single' "double"
two command-file lines
INNER_PAYLOAD
printf 'expanded:%s\n' "${TRANSPORT_VALUE}" >>"${OUTPUT_FILE}"
VLLM_TRANSPORT_COMMAND

OUTPUT_FILE="${transport_file}" \
  TRANSPORT_VALUE='value with spaces, $dollars, and "quotes"' \
  bash "${helper}" "${transport_command_file}"

transport_output=$(<"${transport_file}")
assert_equal \
  $'literal:$HOME\nquotes:\'single\' "double"\ntwo command-file lines\nexpanded:value with spaces, $dollars, and "quotes"' \
  "${transport_output}" \
  "quotes, dollars, and newlines"

stdin_file="${test_root}/stdin.out"
stdin_payload="stdin payload with \$dollars and \"quotes\""
stdin_command_file="${test_root}/stdin-command.sh"
cat >"${stdin_command_file}" <<'VLLM_STDIN_COMMAND'
IFS= read -r input
printf '%s\n' "${input}" >"${OUTPUT_FILE}"
VLLM_STDIN_COMMAND

printf '%s\n' "${stdin_payload}" |
  OUTPUT_FILE="${stdin_file}" bash "${helper}" "${stdin_command_file}"

stdin_output=$(<"${stdin_file}")
assert_equal \
  "${stdin_payload}" \
  "${stdin_output}" \
  "stdin preservation"

set +e
bash "${helper}" >/dev/null 2>&1
usage_status=$?
bash "${helper}" "${test_root}/missing-command" >/dev/null 2>&1
missing_status=$?
set -e
assert_equal "64" "${usage_status}" "missing argument exit status"
assert_equal "66" "${missing_status}" "unreadable command file exit status"

echo "PASS: hf-offline-retry"
