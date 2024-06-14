#!/bin/bash

set -euo pipefail

if [[ "${BUILDKITE_BRANCH}" == "main" ]]; then
    echo "Run full tests on main"
    exit 0
fi

get_diff() {
    echo $(git diff --name-only --diff-filter=ACM $(git merge-base origin/main HEAD))
}

diff=$(get_diff)

patterns=(
    ".buildkite/"
    ".github/"
    "cmake/"
    "benchmarks/"
    "csrc/"
    "tests/"
    "vllm/"
    "Dockerfile"
    "format.sh"
    "pyproject.toml"
    "requirements*"
    "setup.py"
)

for file in $diff; do
    for pattern in "${patterns[@]}"; do
        if [[ $file == $pattern* ]] || [[ $file == $pattern ]]; then
            echo "Matched pattern: $pattern"
            exit 0
        fi
    done
done

echo "No relevant changes found to trigger tests. Skipping."
exit 2
