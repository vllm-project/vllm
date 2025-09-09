#!/bin/bash

CI=${1:-0}
PYTHON_VERSION=${2:-local}

if [ "$CI" -eq 1 ]; then
    set -e
fi

if [ $PYTHON_VERSION == "local" ]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

run_mypy() {
    echo "Running mypy on $1"
    if [ "$CI" -eq 1 ] && [ -z "$1" ]; then
        mypy --python-version "${PYTHON_VERSION}" "$@"
        return
    fi
    mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
}

run_mypy # Note that this is less strict than CI
run_mypy tests
run_mypy vllm/attention
run_mypy vllm/compilation
run_mypy vllm/distributed
run_mypy vllm/engine
run_mypy vllm/executor
run_mypy vllm/inputs
run_mypy vllm/lora
run_mypy --exclude 'vllm/model_executor/layers/fla/ops' vllm/model_executor
run_mypy vllm/plugins
run_mypy vllm/worker
run_mypy vllm/v1
