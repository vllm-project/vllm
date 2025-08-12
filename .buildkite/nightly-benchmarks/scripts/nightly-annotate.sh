#!/bin/bash

set -ex
set -o pipefail


main() {

    (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
    (which jq) || (apt-get update && apt-get -y install jq)
    (which zip) || (apt-get install -y zip)

    if [ ! -f /workspace/buildkite-agent ]; then
        echo "buildkite-agent binary not found. Skip plotting the results."
        exit 0
    fi

    # initial annotation
    #description="$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/nightly-descriptions.md"

    # download results
    cd "$VLLM_SOURCE_CODE_LOC/benchmarks"
    mkdir -p results/
    /workspace/buildkite-agent artifact download 'results/*nightly_results.json' results/
    ls
    ls results/

    # upload benchmark results
    zip -r results.zip results/
    /workspace/buildkite-agent artifact upload "results.zip"

    # upload benchmarking scripts
    cd "$VLLM_SOURCE_CODE_LOC/"
    zip -r nightly-benchmarks.zip .buildkite/ benchmarks/
    /workspace/buildkite-agent artifact upload "nightly-benchmarks.zip"

    cd "$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/"
    # upload benchmarking pipeline
    /workspace/buildkite-agent artifact upload "nightly-pipeline.yaml"

    cd "$VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/"
    /workspace/buildkite-agent annotate --style "success" --context "nightly-benchmarks-results" --append < nightly-annotation.md
    


    # The figures should be genereated by a separate process outside the CI/CD pipeline

    # # generate figures
    # python3 -m pip install tabulate pandas matplotlib

    # python3 $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/generate-nightly-markdown.py \
    #     --description $description \
    #     --results-folder results/ 


    # python3 $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/plot-nightly-results.py \
    #     --description $description \
    #     --results-folder results/ \
    #     --dataset sharegpt

    # python3 $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/plot-nightly-results.py \
    #     --description $description \
    #     --results-folder results/ \
    #     --dataset sonnet_2048_128

    # python3 $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/scripts/plot-nightly-results.py \
    #     --description $description \
    #     --results-folder results/ \
    #     --dataset sonnet_128_2048
    
    # # upload results and figures
    # /workspace/buildkite-agent artifact upload "nightly_results*.png"
    # /workspace/buildkite-agent artifact upload $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/nightly-pipeline.yaml
    # /workspace/buildkite-agent artifact upload $VLLM_SOURCE_CODE_LOC/.buildkite/nightly-benchmarks/tests/nightly-tests.json
    # /workspace/buildkite-agent annotate --style "success" --context "nightly-benchmarks-results" --append < nightly_results.md
}

main "$@"
