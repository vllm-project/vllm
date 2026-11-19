#!/bin/sh

real_pytest=""
old_ifs="${IFS}"
IFS=:
for path_dir in ${PATH}; do
  path_dir="${path_dir:-.}"
  case ":${CI_INFRA_OTEL_SHIM_PATHS:-}:" in
    *":${path_dir}:"*) continue ;;
  esac
  if [ -x "${path_dir}/pytest" ]; then
    real_pytest="${path_dir}/pytest"
    break
  fi
done
IFS="${old_ifs}"

if [ -z "${real_pytest}" ]; then
  echo "vLLM CI OTel: pytest executable not found" >&2 || :
  exit 127
fi

helper_dir="${CI_INFRA_OTEL_DIR:-}"
otel_pythonpath="${helper_dir}${PYTHONPATH:+:${PYTHONPATH}}"
otel_available() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 2s env PYTHONPATH="${otel_pythonpath}" python3 -c "import ci_otel" \
      >/dev/null 2>&1
  else
    PYTHONPATH="${otel_pythonpath}" python3 -c "import ci_otel" >/dev/null 2>&1
  fi
}

if [ -n "${helper_dir}" ] &&
  [ -n "${CI_INFRA_TRACE_ID:-}" ] &&
  [ -n "${CI_INFRA_COMMAND_SPAN_ID:-}" ] &&
  otel_available; then
  PYTHONPATH="${otel_pythonpath}" exec "${real_pytest}" -p ci_otel "$@"
fi

echo "vLLM CI OTel: pytest tracing skipped; running pytest normally" >&2 || :
exec "${real_pytest}" "$@"
