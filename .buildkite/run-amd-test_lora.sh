# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "Node's GPUs state is \"clean\""
                break
        fi
done

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_lora || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --shm-size=10.24gb --network host --name rocm_test_lora \
	rocm /bin/bash -c "pip install peft; export VLLM_INSTALL_PUNICA_KERNELS=1; \
	python3 -m pytest -v -s vllm/tests/lora" #\
#	--shard-id=$BUILDKITE_PARALLEL_JOB --num-shards=$BUILDKITE_PARALLEL_JOB_COUNT

