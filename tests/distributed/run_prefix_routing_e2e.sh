#!/usr/bin/env bash
set -euo pipefail

workspace_root=$(git rev-parse --show-toplevel)
cd "$workspace_root"

expected_git_sha=${EXPECTED_GIT_SHA:-}
if [[ ! "$expected_git_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "EXPECTED_GIT_SHA must be the exact 40-character commit SHA" >&2
  exit 2
fi

actual_git_sha=$(git rev-parse HEAD)
if [[ "$actual_git_sha" != "$expected_git_sha" ]]; then
  echo "Expected Git SHA $expected_git_sha, found $actual_git_sha" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to record fixed-SHA evidence from a dirty worktree" >&2
  exit 2
fi

python_bin=${PYTHON_BIN:-$workspace_root/.venv/bin/python}
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found or not executable: $python_bin" >&2
  exit 2
fi

export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

echo "prefix_routing_e2e_git_sha=$actual_git_sha"
echo "prefix_routing_e2e_pythonhashseed=$PYTHONHASHSEED"
echo "prefix_routing_e2e_started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$python_bin" -m pytest \
  tests/distributed/test_prefix_scheduler.py \
  tests/distributed/test_prefix_routing_e2e.py \
  -v

echo "prefix_routing_e2e_completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
