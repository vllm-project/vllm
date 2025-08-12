#!/bin/bash

CI=${1:-0}

run_mypy() {
    echo "Running mypy on $1"
    if [ $CI -eq 1 ] && [ -z "$1" ]; then
        mypy "$@"
        return
    fi
    mypy --follow-imports skip "$@"
}

run_mypy # Note that this is less strict than CI
run_mypy tests
run_mypy vllm/attention
run_mypy vllm/compilation
run_mypy vllm/distributed
run_mypy vllm/engine
run_mypy vllm/executor
run_mypy vllm/lora
run_mypy vllm/model_executor
run_mypy vllm/plugins
run_mypy vllm/prompt_adapter
run_mypy vllm/spec_decode
run_mypy vllm/worker
