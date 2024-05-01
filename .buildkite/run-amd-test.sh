# This script build the ROCm docker image and runs test inside it.
set -ex

# Print ROCm version
echo "--- ROCm info"
rocminfo

echo "--- Resetting GPUs"

echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

echo "--- Building container"
sha=$(git rev-parse --short HEAD)
container_name=rocm_${sha}
docker build \
        -t ${container_name} \
        -f Dockerfile.rocm \
        --progress plain \
        .

remove_docker_container() {
   docker rm -f ${container_name} || docker image rm -f ${container_name} || true
}
trap remove_docker_container EXIT

echo "--- Running container"

docker run \
        --device /dev/kfd --device /dev/dri \
        --network host \
        --rm \
        -e HF_TOKEN \
        --name ${container_name} \
        ${container_name} \
        /bin/bash -c $(echo $1 | sed "s/^'//" | sed "s/'$//")

