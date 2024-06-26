#!/bin/bash

set -ex
set -o pipefail

main() {

    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    cd /workspace
    ls
    cd ./vllm
    ls
    exit 0
    cd /
    git clone https://github.com/KuntaiDu/vllm.git
    cd vllm
    git checkout kuntai-benchmark-dev

    if [ ! -f /workspace/buildkite-agent ]; then
        echo "buildkite-agent binary not found. Skip uploading the results."
        return 0
    else
        /workspace/buildkite-agent annotate --style "info" --context "benchmark-results" < /vllm/.buildkite/nightly-benchmarks/nightly-descriptions.md
    fi
    
}

main "$@"