#!/bin/sh

# Transparently add the CI timing plugin to direct `pytest` invocations. The
# original pytest executable is always the fallback: tracing must never decide
# whether a test command succeeds.
real_pytest="${CI_INFRA_OTEL_REAL_PYTEST:-}"
helper_dir="${CI_INFRA_OTEL_DIR:-}"

if [ -z "${real_pytest}" ] || [ ! -x "${real_pytest}" ] ||
  [ -z "${helper_dir}" ] || [ "${CI_INFRA_OTEL_READY:-0}" != "1" ] ||
  [ -z "${CI_INFRA_TRACE_ID:-}" ] ||
  [ -z "${CI_INFRA_COMMAND_SPAN_ID:-}" ]; then
  exec "${real_pytest}" "$@"
fi

otel_pythonpath="${helper_dir}${PYTHONPATH:+:${PYTHONPATH}}"

# Setup already validated plugin registration with this pytest executable.
# Recheck that the helper modules remain importable under the command's
# effective PYTHONPATH. This catches deleted or hidden helpers without paying
# for a second pytest startup on every invocation.
if command -v timeout >/dev/null 2>&1; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="${otel_pythonpath}" \
    timeout 2s python3 -c "import ci_otel, ci_pytest_otel" >/dev/null 2>&1
else
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="${otel_pythonpath}" \
    python3 -c "import ci_otel, ci_pytest_otel" >/dev/null 2>&1
fi
preflight_status=$?

if [ "${preflight_status}" -ne 0 ]; then
  echo "vLLM CI OTel: pytest tracing skipped; running pytest normally" >&2 || :
  exec "${real_pytest}" "$@"
fi

PYTHONPATH="${otel_pythonpath}" exec "${real_pytest}" -p ci_pytest_otel "$@"
