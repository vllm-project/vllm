#!/bin/bash
set -e

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

# TODO - fix shellcheck warnings in excluded directories/files below
find . -path "./.git" -prune -o -name "*.sh" \
    -not -path "./.buildkite/*" \
    -not -path "./benchmarks/*" \
    -not -path "./examples/*" \
    -not -path "./tests/v1/ec_connector/integration/*" \
    -not -path "./tests/v1/kv_connector/nixl_integration/*" \
    -not -path "./tests/standalone_tests/python_only_compile.sh" \
    -not -path "./tools/install_deepgemm.sh" \
    -not -path "./tools/flashinfer-build.sh" \
    -not -path "./tools/vllm-tpu/build.sh" \
    -not -path "./tools/ep_kernels/*" \
    -print0 | xargs -0 -I {} sh -c 'git check-ignore -q "{}" || shellcheck -s bash "{}"'
