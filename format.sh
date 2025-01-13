#!/usr/bin/env bash
# YAPF formatter, adapted from ray and skypilot.
#
# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❓❓$1 is not installed, please run \`pip install -r requirements-lint.txt\`"
        exit 1
    fi
}

# TODO: run pre-commit here
check_command mypy

MYPY_VERSION=$(mypy --version | awk '{print $2}')

# # params: tool name, tool version, required version
tool_version_check() {
    expected=$(grep "$1" requirements-lint.txt | cut -d'=' -f3)
    if [[ "$2" != "$expected" ]]; then
        echo "❓❓Wrong $1 version installed: $expected is required, not $2."
        exit 1
    fi
}

tool_version_check "mypy" "$MYPY_VERSION"

# Run mypy
echo 'vLLM mypy:'
tools/mypy.sh
echo 'vLLM mypy: Done'

echo 'vLLM shellcheck:'
tools/shellcheck.sh
echo 'vLLM shellcheck: Done'

echo 'excalidraw png check:'
tools/png-lint.sh
echo 'excalidraw png check: Done'

if ! git diff --quiet &>/dev/null; then
    echo 
    echo "🔍🔍There are files changed by the format checker or by you that are not added and committed:"
    git --no-pager diff --name-only
    echo "🔍🔍Format checker passed, but please add, commit and push all the files above to include changes made by the format checker."

    exit 1
else
    echo "✨🎉 Format check passed! Congratulations! 🎉✨"
fi
