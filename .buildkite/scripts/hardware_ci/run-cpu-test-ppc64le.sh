#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Setup cleanup
remove_docker_container() {
  if [[ -n "$container_id" ]]; then
      podman stop --all -t0
      podman rm -f "$container_id" || true
  fi
  podman system prune -f
}
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
podman build -t cpu-test-ubi9-ppc -f docker/Dockerfile.ppc64le .

# Run the image
container_id=$(podman run -itd --entrypoint /bin/bash -v /tmp/:/root/.cache/huggingface --privileged=true --network host -e HF_TOKEN cpu-test-ubi9-ppc)

function cpu_tests() {

  # offline inference
  podman exec -it "$container_id" bash -c "
    set -xve
    python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m" >> $HOME/test_basic.log

  # Run basic model test
  podman exec -it "$container_id" bash -c "
    set -evx
    pip install pytest pytest-asyncio einops peft Pillow soundfile transformers_stream_generator matplotlib
    pip install sentence-transformers datamodel_code_generator

    # Note: disable Bart until supports V1
    # pytest -v -s tests/models/language/generation/test_bart.py -m cpu_model
    pytest -v -s tests/models/language/generation/test_common.py::test_models[False-5-32-openai-community/gpt2]
    pytest -v -s tests/models/language/generation/test_common.py::test_models[False-5-32-facebook/opt-125m]
    pytest -v -s tests/models/language/generation/test_common.py::test_models[False-5-32-google/gemma-1.1-2b-it]
    pytest -v -s tests/models/language/pooling/test_classification.py::test_models[float-jason9693/Qwen2.5-1.5B-apeach]
    # TODO: Below test case tests/models/language/pooling/test_embedding.py::test_models[True-ssmits/Qwen2-7B-Instruct-embed-base] fails on ppc64le. Disabling it for time being.
    # pytest -v -s tests/models/language/pooling/test_embedding.py -m cpu_model" >> $HOME/test_rest.log
}

# All of CPU tests are expected to be finished less than 40 mins.

export container_id
export -f cpu_tests
timeout 120m bash -c cpu_tests

