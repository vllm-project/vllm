# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t cpu-test -f Dockerfile.cpu .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run -itd -v ~/.cache/huggingface:/root/.cache/huggingface --cpuset-cpus=48-95 --cpuset-mems=1 --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --name cpu-test cpu-test

# offline inference
docker exec cpu-test bash -c "python3 examples/offline_inference.py"

# Run basic model test
docker exec cpu-test bash -c "cd tests;
  pip install pytest Pillow protobuf
  bash ../.buildkite/download-images.sh
  cd ../
  pytest -v -s tests/models --ignore=tests/models/test_llava.py --ignore=tests/models/test_embedding.py --ignore=tests/models/test_registry.py"
