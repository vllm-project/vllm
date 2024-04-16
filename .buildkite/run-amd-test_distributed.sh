# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_distributed || true; \
						docker rm -f rocm_test_basic_distributed_correctness_opt || true; \
						docker rm -f rocm_test_basic_distributed_correctness_opt || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_distributed \
       	rocm  python3 -m pytest -v -s vllm/tests/distributed/test_pynccl.py

docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_basic_distributed_correctness_opt \
	        rocm  /bin/bash -c "TEST_DIST_MODEL=facebook/opt-125m python3 -m pytest \
		-v -s vllm/tests/distributed/test_basic_distributed_correctness.py"

docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_basic_distributed_correctness_llama \
	        -e HF_TOKEN rocm  /bin/bash -c "TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf python3 -m pytest \
		-v -s vllm/tests/distributed/test_basic_distributed_correctness.py"

