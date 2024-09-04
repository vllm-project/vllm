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

main() {

    check_gpus
    check_hf_token

    df -h

    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)

    cd $VLLM_SOURCE_CODE_LOC/benchmarks
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
    

    # run lmdeploy
    if which lmdeploy >/dev/null; then
        echo "lmdeploy is available, redirect to run-lmdeploy-nightly.sh"
        bash ../.buildkite/nightly-benchmarks/scripts/run-lmdeploy-nightly.sh
        exit 0
    fi

    # run tgi
    if [ -e /tgi-entrypoint.sh ]; then
        echo "tgi is available, redirect to run-tgi-nightly.sh"
        bash ../.buildkite/nightly-benchmarks/scripts/run-tgi-nightly.sh
        exit 0
    fi

    # run trt
    if which trtllm-build >/dev/null; then
        echo "trtllm is available, redirect to run-trt-nightly.sh"
        bash ../.buildkite/nightly-benchmarks/scripts/run-trt-nightly.sh
        exit 0
    fi

    # run vllm
    if [ -e /vllm-workspace ]; then
        echo "vllm is available, redirect to run-vllm-nightly.sh"
        bash ../.buildkite/nightly-benchmarks/scripts/run-vllm-nightly.sh
        exit 0
    fi

}

main "$@"