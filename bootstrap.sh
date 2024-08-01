#!/bin/bash

set -euo pipefail

upload_pipeline() {
    echo "Uploading pipeline..."
    # ls .buildkite || buildkite-agent annotate --style error 'Please merge upstream main branch for buildkite CI'
    # curl -sSfL https://github.com/mitsuhiko/minijinja/releases/latest/download/minijinja-cli-installer.sh | sh
    # source /var/lib/buildkite-agent/.cargo/env
    # if [ $BUILDKITE_PIPELINE_SLUG == "fastcheck" ]; then
    #     curl -o .buildkite/test-template-fastcheck.j2 https://raw.githubusercontent.com/vllm-project/buildkite-ci/main/scripts/test-template-fastcheck.j2
    #     cd .buildkite && minijinja-cli test-template-fastcheck.j2 test-pipeline.yaml > pipeline.yml
    #     cat pipeline.yml
    #     buildkite-agent pipeline upload pipeline.yml
    #     exit 0
    # fi
    # if [ ! -e ".buildkite/test-template.j2" ]; then
    #     curl -o .buildkite/test-template.j2 https://raw.githubusercontent.com/vllm-project/buildkite-ci/main/scripts/test-template-aws.j2
    # fi
    cd .buildkite
    echo $list_file_diff
    minijinja-cli test-template-aws.j2 test-pipeline.yaml -D list_file_diff="$list_file_diff" > pipeline.yml
    # buildkite-agent pipeline upload pipeline.yml
    exit 0
}

get_diff() {
    $(git add .)
    echo $(git diff --name-only --diff-filter=ACMDR $(git merge-base upstream/main HEAD))
}

list_file_diff=$(get_diff | tr ' ' '|')
upload_pipeline
