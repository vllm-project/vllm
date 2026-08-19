#!/bin/sh

real_pytest="${CI_INFRA_OTEL_REAL_PYTEST}"
otel_pythonpath="${CI_INFRA_OTEL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -n "${CI_INFRA_TRACE_ID:-}" ] &&
  [ -n "${CI_INFRA_COMMAND_SPAN_ID:-}" ] &&
  PYTHONPATH="${otel_pythonpath}" python3 -c "import ci_otel" >/dev/null 2>&1; then
  PYTHONPATH="${otel_pythonpath}" exec "${real_pytest}" -p ci_otel "$@"
fi

echo "vLLM CI OTel: pytest tracing skipped; running pytest normally" >&2 || :
exec "${real_pytest}" "$@"
