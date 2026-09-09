#!/bin/sh

[ "${_CI_INFRA_OTEL_INITIALIZED:-0}" = "1" ] && return 0

# No-op fallbacks keep every generated wrapper safe when setup is unavailable.
ci_otel_start() { :; }
ci_otel_finish() { :; }
ci_otel_run() { shift 2; env "$@"; return $?; }

if [ -z "${CI_INFRA_OTEL_DIR:-}" ]; then
  echo "vLLM CI OTel: helper directory is unset; tracing disabled" >&2 || :
  return 0
fi
export CI_INFRA_OTEL_DIR

_CI_INFRA_OTEL_OWNS_RUNTIME=0
if [ -z "${CI_INFRA_OTEL_RUNTIME_DIR:-}" ]; then
  CI_INFRA_OTEL_RUNTIME_DIR="$(mktemp -d 2>/dev/null || :)"
  [ -n "${CI_INFRA_OTEL_RUNTIME_DIR}" ] && _CI_INFRA_OTEL_OWNS_RUNTIME=1
fi
CI_INFRA_OTEL_SPOOL_DIR="${CI_INFRA_OTEL_SPOOL_DIR:-${CI_INFRA_OTEL_RUNTIME_DIR}/spans}"
export CI_INFRA_OTEL_RUNTIME_DIR CI_INFRA_OTEL_SPOOL_DIR

_CI_INFRA_OTEL_PYTHON="$(command -v python3 2>/dev/null || :)"
if [ -z "${CI_INFRA_OTEL_RUNTIME_DIR}" ] ||
  [ -z "${_CI_INFRA_OTEL_PYTHON}" ] ||
  [ ! -f "${CI_INFRA_OTEL_DIR}/ci_otel.py" ] ||
  ! mkdir -p "${CI_INFRA_OTEL_SPOOL_DIR}" ||
  ! PYTHONPATH="${CI_INFRA_OTEL_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${_CI_INFRA_OTEL_PYTHON}" -c "import ci_otel" >/dev/null 2>&1; then
  echo "vLLM CI OTel: tracing unavailable; test command will run normally" >&2 || :
  if [ "${_CI_INFRA_OTEL_OWNS_RUNTIME}" = "1" ]; then
    rm -rf -- "${CI_INFRA_OTEL_RUNTIME_DIR}" || :
  fi
  return 0
fi

_CI_INFRA_OTEL_SHIM_DIR="${CI_INFRA_OTEL_RUNTIME_DIR}/bin"
if mkdir -p "${_CI_INFRA_OTEL_SHIM_DIR}"; then
  _CI_INFRA_OTEL_SHIM="${_CI_INFRA_OTEL_SHIM_DIR}/pytest"
  if [ ! -e "${_CI_INFRA_OTEL_SHIM}" ] && [ ! -L "${_CI_INFRA_OTEL_SHIM}" ]; then
    ln -s "${CI_INFRA_OTEL_DIR}/ci_pytest.sh" "${_CI_INFRA_OTEL_SHIM}" || :
  fi
  if [ -L "${_CI_INFRA_OTEL_SHIM}" ] &&
    [ "$(readlink "${_CI_INFRA_OTEL_SHIM}" 2>/dev/null || :)" = \
      "${CI_INFRA_OTEL_DIR}/ci_pytest.sh" ]; then
    case ":${CI_INFRA_OTEL_SHIM_PATHS:-}:" in
      *":${_CI_INFRA_OTEL_SHIM_DIR}:"*) ;;
      *)
        CI_INFRA_OTEL_SHIM_PATHS="${CI_INFRA_OTEL_SHIM_PATHS:+${CI_INFRA_OTEL_SHIM_PATHS}:}${_CI_INFRA_OTEL_SHIM_DIR}"
        PATH="${_CI_INFRA_OTEL_SHIM_DIR}:${PATH}"
        ;;
    esac
    export CI_INFRA_OTEL_SHIM_PATHS PATH
    hash -r 2>/dev/null || :
  fi
fi

_ci_otel_python() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 2s "${_CI_INFRA_OTEL_PYTHON}" "$@"
  else
    "${_CI_INFRA_OTEL_PYTHON}" "$@"
  fi
}

ci_otel_start() {
  _CI_INFRA_OTEL_COMMAND_INDEX="$1"
  _CI_INFRA_OTEL_COMMAND_LABEL="${2:-command ${_CI_INFRA_OTEL_COMMAND_INDEX}}"
  _CI_INFRA_OTEL_CONTEXT="$(_ci_otel_python "${CI_INFRA_OTEL_DIR}/ci_otel.py" new-context)" ||
    return 0
  # The helper prints four whitespace-free fields.
  # shellcheck disable=SC2086
  set -- ${_CI_INFRA_OTEL_CONTEXT}
  [ "$#" -eq 4 ] || return 0

  _CI_INFRA_OTEL_TRACE_ID="$1"
  _CI_INFRA_OTEL_COMMAND_SPAN_ID="$2"
  _CI_INFRA_OTEL_PARENT_SPAN_ID="$3"
  _CI_INFRA_OTEL_START_NS="$4"
  CI_INFRA_TRACE_ID="${_CI_INFRA_OTEL_TRACE_ID}"
  CI_INFRA_COMMAND_SPAN_ID="${_CI_INFRA_OTEL_COMMAND_SPAN_ID}"
  export CI_INFRA_TRACE_ID CI_INFRA_COMMAND_SPAN_ID
  _CI_INFRA_OTEL_ACTIVE=1
}

ci_otel_finish() {
  _CI_INFRA_OTEL_COMMAND_STATUS="${1:-0}"
  [ "${_CI_INFRA_OTEL_ACTIVE:-0}" = "1" ] || return 0
  _CI_INFRA_OTEL_ACTIVE=0
  _ci_otel_python "${CI_INFRA_OTEL_DIR}/ci_otel.py" record-command \
    "${_CI_INFRA_OTEL_TRACE_ID}" "${_CI_INFRA_OTEL_COMMAND_SPAN_ID}" \
    "${_CI_INFRA_OTEL_PARENT_SPAN_ID}" "${_CI_INFRA_OTEL_START_NS}" \
    "${_CI_INFRA_OTEL_COMMAND_INDEX}" "${_CI_INFRA_OTEL_COMMAND_STATUS}" \
    "${_CI_INFRA_OTEL_COMMAND_LABEL}" || :
  CI_INFRA_TRACE_ID=""
  CI_INFRA_COMMAND_SPAN_ID=""
  export CI_INFRA_TRACE_ID CI_INFRA_COMMAND_SPAN_ID
}

# Run a simple command with tracing. Only for commands that do not modify
# shell state (export, cd, etc.) — those need the explicit start/finish pair.
# Assignment-prefixed arguments (VAR=value) are routed through env so the
# shell does not try to execute the assignment as a program.
ci_otel_run() {
  _CI_INFRA_OTEL_RUN_INDEX="$1"
  _CI_INFRA_OTEL_RUN_LABEL="$2"
  shift 2
  ci_otel_start "${_CI_INFRA_OTEL_RUN_INDEX}" "${_CI_INFRA_OTEL_RUN_LABEL}" || :
  env "$@"
  _CI_INFRA_OTEL_RUN_STATUS=$?
  ci_otel_finish "${_CI_INFRA_OTEL_RUN_STATUS}" || :
  return "${_CI_INFRA_OTEL_RUN_STATUS}"
}

_ci_otel_on_exit() {
  _CI_INFRA_OTEL_EXIT_STATUS=$?
  trap - 0
  ci_otel_finish "${_CI_INFRA_OTEL_EXIT_STATUS}" || :
  if command -v timeout >/dev/null 2>&1; then
    timeout 4s "${_CI_INFRA_OTEL_PYTHON}" "${CI_INFRA_OTEL_DIR}/ci_otel.py" flush || :
  else
    "${_CI_INFRA_OTEL_PYTHON}" "${CI_INFRA_OTEL_DIR}/ci_otel.py" flush || :
  fi
  if [ "${_CI_INFRA_OTEL_OWNS_RUNTIME}" = "1" ]; then
    rm -rf -- "${CI_INFRA_OTEL_RUNTIME_DIR}" || :
  fi
  exit "${_CI_INFRA_OTEL_EXIT_STATUS}"
}

trap _ci_otel_on_exit 0
_CI_INFRA_OTEL_INITIALIZED=1
:
