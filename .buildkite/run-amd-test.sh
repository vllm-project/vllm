# This script runs test inside the corresponding ROCm docker container.
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
image_name=rocm_${sha}
container_name=rocm_${sha}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)
docker build \
        -t ${image_name} \
        -f Dockerfile.rocm \
        --progress plain \
        .

remove_docker_container() {
   docker rm -f ${container_name} || docker image rm -f ${image_name} || true
}
trap remove_docker_container EXIT

echo "--- Running container"

docker run \
        --device /dev/kfd --device /dev/dri \
        --network host \
        --rm \
        -e HF_TOKEN \
        --name ${container_name} \
        ${image_name} \
        /bin/bash -c "${@}"

