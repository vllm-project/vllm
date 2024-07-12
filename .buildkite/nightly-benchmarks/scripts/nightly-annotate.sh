#!/bin/bash

set -ex
set -o pipefail


main() {

    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)

    if [ ! -f /workspace/buildkite-agent ]; then
        echo "buildkite-agent binary not found. Skip plotting the results."
        exit 0
    fi

    # initial annotation
    description="$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/nightly-descriptions.md"

    # download results
    cd $VLLM_SOURCE_CODE_LOC/benchmarks
    mkdir -p results/
    /workspace/buildkite-agent artifact download 'results/*nightly_results.json' results/
    ls
    ls results/

    # generate figures
    python3 -m pip install tabulate pandas matplotlib
    python3 $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/plot-nightly-results.py \
        --description $description \
        --results-folder results/
    
    # upload results and figures
    /workspace/buildkite-agent artifact upload "nightly_results.png"
    /workspace/buildkite-agent artifact upload $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/nightly-pipeline.yaml
    /workspace/buildkite-agent artifact upload $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/tests/nightly-tests.json
    /workspace/buildkite-agent annotate --style "success" --context "nightly-benchmarks-results" --append < nightly_results.md
}

main "$@"