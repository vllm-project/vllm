#!/bin/bash

set -o pipefail
set -x

check_gpus() {
    # check the number of GPUs and GPU type.
    declare -g gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [[ $gpu_count -gt 0 ]]; then
        echo "GPU found."
    else
        echo "Need at least 1 GPU to run benchmarking."
        exit 1
    fi
    declare -g gpu_type=$(echo $(nvidia-smi --query-gpu=name --format=csv,noheader) | awk '{print $2}')
    echo "GPU type is $gpu_type"
}

check_hf_token() {
    # check if HF_TOKEN is available and valid
    if [[ -z "$HF_TOKEN" ]]; then
        echo "Error: HF_TOKEN is not set."
        exit 1
    elif [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
        echo "Error: HF_TOKEN does not start with 'hf_'."
        exit 1
    else
        echo "HF_TOKEN is set and valid."
    fi
}

get_current_llm_serving_engine() {

    # run lmdeploy
    if which lmdeploy >/dev/null; then
        echo "Container: lmdeploy"
        export CURRENT_LLM_SERVING_ENGINE=lmdeploy
        return
    fi

    # run tgi
    if [ -e /tgi-entrypoint.sh ]; then
        echo "Container: tgi"
        export CURRENT_LLM_SERVING_ENGINE=tgi
        return
    fi

    # run trt
    if which trtllm-build >/dev/null; then
        echo "Container: trt"
        export CURRENT_LLM_SERVING_ENGINE=trt
        return
    fi

    # run sgl
    if [ -e /sgl-workspace ]; then
        echo "Container: sgl"
        export CURRENT_LLM_SERVING_ENGINE=sgl
        return
    fi

    # run vllm
    if [ -e /vllm-workspace ]; then
        echo "Container: vllm"
        export CURRENT_LLM_SERVING_ENGINE=vllm
        return
    fi
}

ensure_installed () {
    # Ensure that the given command is installed by apt-get
    local cmd=$1
    if ! which $cmd >/dev/null; then
        apt-get update && apt-get install -y $cmd
    fi
}

main() {

    check_gpus
    check_hf_token
    get_current_llm_serving_engine

    # check storage
    df -h

    ensure_installed wget
    ensure_installed curl
    ensure_installed jq

    cd $VLLM_SOURCE_CODE_LOC/benchmarks
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    declare -g RESULTS_FOLDER=results/
    mkdir -p $RESULTS_FOLDER
    BENCHMARK_ROOT=../.buildkite/nightly-benchmarks/

    run_serving_tests $BENCHMARK_ROOT/tests/nightly-tests.json
    python3 -m pip install tabulate pandas
    python3 $BENCHMARK_ROOT/scripts/summary-nightly-results.py
    upload_to_buildkite

}

main "$@"