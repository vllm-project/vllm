# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_engine || true; \
                            docker rm -f rocm_test_tokenization || true; \
                            docker rm -f rocm_test_sequence || true; \
                            docker rm -f rocm_test_config || true;}
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_engine \
        rocm python3 -m pytest -v -s vllm/tests/engine
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_tokenization \
        -e HF_TOKEN rocm python3 -m pytest -v -s vllm/tests/tokenization
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_sequence \
        rocm python3 -m pytest -v -s vllm/tests/test_sequence.py
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_config \
        rocm python3 -m pytest -v -s vllm/tests/test_config.py

