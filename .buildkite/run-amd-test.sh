# This script build the ROCm docker image and runs test inside it.
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

sha=$(git rev-parse --short HEAD)
container_name=rocm_${sha}
docker build -t ${container_name} -f Dockerfile.rocm .

docker run \
        --device /dev/kfd --device /dev/dri \
        --network host \
        --rm \
        -e HF_TOKEN \
        ${container_name} \
        /bin/bash -c "$1"
