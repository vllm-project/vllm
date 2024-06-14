#!/bin/bash

set -euo pipefail

upload_pipeline() {
    echo "Uploading pipeline..."
    exit 0
}

get_diff() {
    echo $(git diff --name-only --diff-filter=ACM $(git merge-base origin/main HEAD))
}

if [[ "${BUILDKITE_BRANCH}" == "main" ]]; then
    echo "Run full tests on main"
    upload_pipeline
fi


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
        if [[ $file == *$pattern* ]] || [[ $file == $pattern ]]; then
            TRIGGER=1
            echo "Found relevant changes: $file"
            upload_pipeline
        fi
    done
done

echo "No relevant changes found to trigger tests."
exit 0
