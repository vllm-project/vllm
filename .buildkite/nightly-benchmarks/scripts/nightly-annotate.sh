#!/bin/bash

set -ex
set -o pipefail


main() {

    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    
    df -h

    if [ ! -f /workspace/buildkite-agent ]; then
        echo "buildkite-agent binary not found. Skip uploading the results."
        exit 0
    fi

    # initial annotation
    description="$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/nightly-descriptions.md"

    # download results
    cd $VLLM_SOURCE_CODE_LOC/benchmarks
    mkdir -p results/
    /workspace/buildkite-agent artifact download 'results/*nightly_results.json' results

    # generate figures
    python3 -m pip install tabulate pandas
    python3 $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/summary-nightly-results.py \
        --results-folder results \
        --description $description

    
    
}

main "$@"