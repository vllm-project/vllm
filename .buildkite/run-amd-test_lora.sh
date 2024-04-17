# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_lora || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --shm-size=10.24gb --network host --name rocm_test_lora \
	rocm /bin/bash -c "pip install peft; python3 -m pytest -v -s vllm/tests/lora" #\
#	--shard-id=$BUILDKITE_PARALLEL_JOB --num-shards=$BUILDKITE_PARALLEL_JOB_COUNT

