#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Setup cleanup
remove_docker_container() {
  if [[ -n "$container_id" ]]; then
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
    set -e
    python3 examples/offline_inference/basic/generate.py --model facebook/opt-125m"

  # Run basic model test
  podman exec -it "$container_id" bash -c "
    set -e
    pip install pytest pytest-asyncio einops peft Pillow soundfile transformers_stream_generator matplotlib
    pip install sentence-transformers datamodel_code_generator
    pytest -v -s tests/models/embedding/language/test_cls_models.py::test_classification_models[float-jason9693/Qwen2.5-1.5B-apeach]
    pytest -v -s tests/models/embedding/language/test_embedding.py::test_models[half-BAAI/bge-base-en-v1.5]
    pytest -v -s tests/models/encoder_decoder/language -m cpu_model"
}

# All of CPU tests are expected to be finished less than 40 mins.

export container_id
export -f cpu_tests
timeout 40m bash -c cpu_tests

