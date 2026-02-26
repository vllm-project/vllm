#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRE_COMMIT_DIR="${PRE_COMMIT_HOME:-${REPO_ROOT}/.cache/pre-commit}"
XDG_CACHE_DIR="${XDG_CACHE_HOME:-${REPO_ROOT}/.cache}"

export PRE_COMMIT_HOME="$PRE_COMMIT_DIR"
export XDG_CACHE_HOME="$XDG_CACHE_DIR"

mkdir -p "${PRE_COMMIT_HOME}" "${XDG_CACHE_HOME}"

if ! command -v pre-commit > /dev/null 2>&1; then
  echo "pre-commit not found. Install it (for example, with: uv pip install pre-commit)." >&2
  exit 1
fi

pre-commit "$@"
