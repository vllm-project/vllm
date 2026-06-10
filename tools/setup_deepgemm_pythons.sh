#!/usr/bin/env bash
# Provision one bare Python per `requires-python` entry (or per argument) and
# print their paths as ":"-separated DEEPGEMM_PYTHON_INTERPRETERS. Skip this
# entirely if you already have interpreter paths.
#
# Usage:
#   export DEEPGEMM_PYTHON_INTERPRETERS=$(tools/setup_deepgemm_pythons.sh)
#   python setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
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
  # uv-managed Python ensures Python.h is present; system 3.X-dev packages
  # on the manylinux / Ubuntu build bases are not always installed.
  [ -x "$venv/bin/python" ] || \
    uv venv --python "$V" "$venv" --python-preference only-managed --seed \
      >/dev/null
  paths="$paths:$venv/bin/python"
done
echo "${paths#:}"
