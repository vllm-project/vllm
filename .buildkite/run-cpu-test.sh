# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t cpu-test -f Dockerfile.cpu .
docker build --build-arg VLLM_CPU_DISABLE_AVX512="true" -t cpu-test-avx2 -f Dockerfile.cpu .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test cpu-test-avx2 || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --cpuset-cpus=48-95 \
  --cpuset-mems=1 --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --name cpu-test cpu-test
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --cpuset-cpus=48-95 \
  --cpuset-mems=1 --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --name cpu-test-avx2 cpu-test-avx2

# offline inference
docker exec cpu-test bash -c "python3 examples/offline_inference.py"
docker exec cpu-test-avx2 bash -c "python3 examples/offline_inference.py"

# Run basic model test
docker exec cpu-test bash -c "cd tests;
  pip install pytest Pillow protobuf
  cd ../
  pytest -v -s tests/models -m \"not vlm\" --ignore=tests/models/test_embedding.py --ignore=tests/models/test_registry.py --ignore=tests/models/test_jamba.py" # Mamba on CPU is not supported
