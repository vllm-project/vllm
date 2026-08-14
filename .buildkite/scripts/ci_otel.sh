#!/usr/bin/env bash

_VLLM_CI_OTEL_DIR="${VLLM_CI_OTEL_DIR:-/vllm-workspace/.buildkite/scripts}"
export PYTHONPATH="${_VLLM_CI_OTEL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p ci_pytest_otel"

ci_otel_run() {
  local command_index="$1"
  local encoded_label="$2"
  local encoded_command="$3"
  local command_label
  local command_text
  local context
  local trace_id
  local span_id
  local parent_span_id
  local start_ns
  local end_ns
  local command_status

  if ! command_label="$(printf '%s' "${encoded_label}" | base64 --decode)" ||
    ! command_text="$(printf '%s' "${encoded_command}" | base64 --decode)"; then
    echo "vLLM CI OTel: could not decode generated command metadata" >&2
    return 1
  fi

  if ! context="$(python3 "${_VLLM_CI_OTEL_DIR}/ci_otel.py" new-context)"; then
    eval "${command_text}"
    return $?
  fi
  read -r trace_id span_id parent_span_id <<<"${context}"
  [[ "${parent_span_id}" == "-" ]] && parent_span_id=""
  start_ns="$(date +%s%N)"

  local VLLM_CI_TRACE_ID="${trace_id}"
  local VLLM_CI_COMMAND_SPAN_ID="${span_id}"
  export VLLM_CI_TRACE_ID VLLM_CI_COMMAND_SPAN_ID
  if eval "${command_text}"; then
    command_status=0
  else
    command_status=$?
  fi
  end_ns="$(date +%s%N)"

  python3 "${_VLLM_CI_OTEL_DIR}/ci_otel.py" command \
    --trace-id "${trace_id}" \
    --span-id "${span_id}" \
    --parent-span-id "${parent_span_id}" \
    --start-ns "${start_ns}" \
    --end-ns "${end_ns}" \
    --index "${command_index}" \
    --label "${command_label}" \
    --exit-code "${command_status}" || true
  return "${command_status}"
}
