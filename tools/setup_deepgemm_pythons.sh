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
#   export DEEPGEMM_PYTHON_INTERPRETERS=$(tools/setup_deepgemm_pythons.sh)
#   python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
#
# With no args, expands to every CPython covered by `requires-python` in
# pyproject.toml. Pass explicit versions (e.g. `3.10 3.11`) to override.
#
# Skip this script if you don't have uv: set DEEPGEMM_PYTHON_INTERPRETERS
# directly to existing interpreter paths. Editable / single-Python builds
# don't need the env var at all (cmake falls back to the build interpreter).
#
# Optional: DEEPGEMM_VENV_PREFIX (default: /tmp/dgenv).
set -euo pipefail

if [ "$#" -eq 0 ]; then
  # Derive the matrix from `requires-python = ">=3.X,<3.Y"` in pyproject.toml.
  pyproject="$(dirname "$0")/../pyproject.toml"
  spec=$(grep -E '^requires-python' "$pyproject" \
         | grep -oE '>=3\.[0-9]+,<3\.[0-9]+')
  lo=${spec#>=3.}; lo=${lo%%,*}
  hi=${spec##*<3.}
  set -- $(seq "$lo" $((hi - 1)) | sed 's/^/3./')
fi

prefix="${DEEPGEMM_VENV_PREFIX:-/tmp/dgenv}"
mkdir -p "$prefix"

paths=""
for V in "$@"; do
  venv="$prefix/$V"
  # Force a managed (uv-downloaded) Python so dev headers are bundled.
  # System Pythons on the build base may lack headers (manylinux's
  # /opt/python/cpXY-cpXY are off PATH; an apt-installed python3.X often
  # has no -dev), and the per-Python build needs Python.h.
  [ -x "$venv/bin/python" ] || \
    uv venv --python "$V" "$venv" --python-preference only-managed --seed \
      >/dev/null
  paths="$paths:$venv/bin/python"
done
echo "${paths#:}"
