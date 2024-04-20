# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() {     docker rm -f rocm_test_kernels0 || true; \
                                docker rm -f rocm_test_kernels1 || true; \
                                docker rm -f rocm_test_kernels2 || true; \
                                docker rm -f rocm_test_kernels3 || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image

#PROTOTYPE
#docker run --device /dev/kfd --device /dev/dri --network host -e HIP_VISIBLE_DEVICES \
#        --name rocm_test_kernels rocm python3 -m pytest -v -s vllm/tests/kernels \
#        --shard-id=0 --num-shards=4

export HIP_VISIBLE_DEVICES=1
docker run --device /dev/kfd --device /dev/dri --network host -e HIP_VISIBLE_DEVICES \
        --name rocm_test_kernels0 rocm python3 -m pytest -v -s vllm/tests/kernels \
        --shard-id=0 --num-shards=4 &

export HIP_VISIBLE_DEVICES=2
docker run --device /dev/kfd --device /dev/dri --network host -e HIP_VISIBLE_DEVICES \
        --name rocm_test_kernels1 rocm python3 -m pytest -v -s vllm/tests/kernels \
        --shard-id=1 --num-shards=4 &

export HIP_VISIBLE_DEVICES=3
docker run --device /dev/kfd --device /dev/dri --network host -e HIP_VISIBLE_DEVICES \
        --name rocm_test_kernels2 rocm python3 -m pytest -v -s vllm/tests/kernels \
        --shard-id=2 --num-shards=4 &

export HIP_VISIBLE_DEVICES=4
docker run --device /dev/kfd --device /dev/dri --network host -e HIP_VISIBLE_DEVICES \
        --name rocm_test_kernels3 rocm python3 -m pytest -v -s vllm/tests/kernels \
        --shard-id=3 --num-shards=4 