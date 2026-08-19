#!/bin/sh

_CI_INFRA_OTEL_DIR="${CI_INFRA_OTEL_DIR:?CI_INFRA_OTEL_DIR is required}"
CI_INFRA_OTEL_RUNTIME_DIR="${CI_INFRA_OTEL_RUNTIME_DIR:-$(mktemp -d 2>/dev/null || :)}"
CI_INFRA_OTEL_SPOOL_DIR="${CI_INFRA_OTEL_SPOOL_DIR:-${CI_INFRA_OTEL_RUNTIME_DIR}/spans}"
export CI_INFRA_OTEL_RUNTIME_DIR CI_INFRA_OTEL_SPOOL_DIR

if [ -z "${CI_INFRA_OTEL_RUNTIME_DIR}" ] ||
  ! command -v python3 >/dev/null 2>&1 ||
  ! mkdir -p "${CI_INFRA_OTEL_SPOOL_DIR}" ||
  ! PYTHONPATH="${_CI_INFRA_OTEL_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -c "import ci_otel" >/dev/null 2>&1; then
  echo "vLLM CI OTel: tracing unavailable; test command will run normally" >&2 || :
  return 0
fi

_CI_INFRA_OTEL_REAL_PYTEST=""
_CI_INFRA_OTEL_OLD_IFS="${IFS}"
IFS=:
for _CI_INFRA_OTEL_PATH_DIR in ${PATH}; do
  if [ -x "${_CI_INFRA_OTEL_PATH_DIR:-.}/pytest" ]; then
    _CI_INFRA_OTEL_REAL_PYTEST="${_CI_INFRA_OTEL_PATH_DIR:-.}/pytest"
    break
  fi
done
IFS="${_CI_INFRA_OTEL_OLD_IFS}"
if [ -x "${_CI_INFRA_OTEL_REAL_PYTEST}" ] &&
  mkdir -p "${CI_INFRA_OTEL_RUNTIME_DIR}/bin" &&
  ln -s "${_CI_INFRA_OTEL_DIR}/ci_pytest.sh" \
    "${CI_INFRA_OTEL_RUNTIME_DIR}/bin/pytest"; then
  CI_INFRA_OTEL_REAL_PYTEST="${_CI_INFRA_OTEL_REAL_PYTEST}"
  PATH="${CI_INFRA_OTEL_RUNTIME_DIR}/bin:${PATH}"
  export CI_INFRA_OTEL_REAL_PYTEST PATH
  pytest() {
    "${CI_INFRA_OTEL_RUNTIME_DIR}/bin/pytest" "$@"
  }
fi

_ci_otel_python() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 2s python3 "$@"
  else
    python3 "$@"
  fi
}

ci_otel_start() {
  _CI_INFRA_OTEL_COMMAND_INDEX="$1"
  _CI_INFRA_OTEL_COMMAND_LABEL="$(printf '%s' "$2" | base64 --decode 2>/dev/null)" ||
    _CI_INFRA_OTEL_COMMAND_LABEL="command ${_CI_INFRA_OTEL_COMMAND_INDEX}"
  _CI_INFRA_OTEL_CONTEXT="$(_ci_otel_python "${_CI_INFRA_OTEL_DIR}/ci_otel.py" new-context)" ||
    return 0
  set -- ${_CI_INFRA_OTEL_CONTEXT}
  [ "$#" -eq 4 ] || return 0

  CI_INFRA_TRACE_ID="$1"
  CI_INFRA_COMMAND_SPAN_ID="$2"
  export CI_INFRA_TRACE_ID CI_INFRA_COMMAND_SPAN_ID
  _CI_INFRA_OTEL_ACTIVE=1
  _CI_INFRA_OTEL_PARENT_SPAN_ID="$3"
  _CI_INFRA_OTEL_START_NS="$4"
}

ci_otel_finish() {
  _CI_INFRA_OTEL_COMMAND_STATUS="${1:-0}"
  [ "${_CI_INFRA_OTEL_ACTIVE:-0}" = "1" ] || return 0
  _CI_INFRA_OTEL_ACTIVE=0
  _ci_otel_python "${_CI_INFRA_OTEL_DIR}/ci_otel.py" record-command \
    "${CI_INFRA_TRACE_ID}" "${CI_INFRA_COMMAND_SPAN_ID}" \
    "${_CI_INFRA_OTEL_PARENT_SPAN_ID}" "${_CI_INFRA_OTEL_START_NS}" \
    "${_CI_INFRA_OTEL_COMMAND_INDEX}" "${_CI_INFRA_OTEL_COMMAND_STATUS}" \
    "${_CI_INFRA_OTEL_COMMAND_LABEL}" || :
  CI_INFRA_TRACE_ID=""
  CI_INFRA_COMMAND_SPAN_ID=""
  export CI_INFRA_TRACE_ID CI_INFRA_COMMAND_SPAN_ID
}

_ci_otel_on_exit() {
  _CI_INFRA_OTEL_EXIT_STATUS=$?
  trap - 0
  ci_otel_finish "${_CI_INFRA_OTEL_EXIT_STATUS}" || :
  if command -v timeout >/dev/null 2>&1; then
    timeout 4s python3 "${_CI_INFRA_OTEL_DIR}/ci_otel.py" flush || :
  else
    python3 "${_CI_INFRA_OTEL_DIR}/ci_otel.py" flush || :
  fi
  exit "${_CI_INFRA_OTEL_EXIT_STATUS}"
}

trap _ci_otel_on_exit 0
:
