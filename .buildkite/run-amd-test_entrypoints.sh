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
remove_docker_container() {     docker rm -f rocm_test_entrypoints1 || true; \
                                docker rm -f rocm_test_entrypoints2 || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
export HIP_VISIBLE_DEVICES=1
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_entrypoints1 \
         --shm-size=10.24gb rocm /bin/bash -c "pip install openai; \
         python3 -m pytest -v -s vllm/tests/entrypoints --ignore=vllm/tests/entrypoints/test_server_oot_registration.py" &

export HIP_VISIBLE_DEVICES=2
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_entrypoints2 \
         --shm-size=10.24gb rocm /bin/bash -c "pip install openai; \
         python3 -m pytest -v -s vllm/tests/entrypoints/test_server_oot_registration.py"

