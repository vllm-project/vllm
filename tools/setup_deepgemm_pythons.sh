#!/usr/bin/env bash
# Provision bare Python interpreters for the DeepGEMM `_C` per-Python build
# and print a colon-separated list of their paths to stdout.
#
# Each target Python only needs a working interpreter — torch is not
# installed since `tools/build_deepgemm_C.py` runs from the build interpreter.
# uv re-uses any matching system Python and downloads a managed build
# otherwise.
#
# Usage:
#   export DEEPGEMM_PYTHON_INTERPRETERS=$(tools/setup_deepgemm_pythons.sh \
#       3.10 3.11 3.12 3.13 3.14)
#   python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
#
# Skip this script if you don't have uv: set DEEPGEMM_PYTHON_INTERPRETERS
# directly to existing interpreter paths. Editable / single-Python builds
# don't need the env var at all (cmake falls back to the build interpreter).
#
# Optional: DEEPGEMM_VENV_PREFIX (default: /tmp/dgenv).
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <python_version> [<python_version>...]" >&2
  exit 1
fi

prefix="${DEEPGEMM_VENV_PREFIX:-/tmp/dgenv}"
mkdir -p "$prefix"

paths=""
for V in "$@"; do
  venv="$prefix/$V"
  [ -x "$venv/bin/python" ] || uv venv --python "$V" "$venv" --seed >/dev/null
  paths="$paths:$venv/bin/python"
done
echo "${paths#:}"
