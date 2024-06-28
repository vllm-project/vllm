# This script runs test inside the corresponding ROCm docker container.
set -ex

# Print ROCm version
echo "--- ROCm info"
rocminfo

# cleanup older docker images
cleanup_docker() {
  # Get Docker's root directory
  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory."
    exit 1
  fi
  echo "Docker root directory: $docker_root"
  # Check disk usage of the filesystem where Docker's root directory is located
  disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
  # Define the threshold
  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
    # Remove dangling images (those that are not tagged and not used by any container)
    docker image prune -f
    # Remove unused volumes
    docker volume prune -f
    echo "Docker images and volumes cleanup completed."
  else
    echo "Disk usage is below $threshold%. No cleanup needed."
  fi
}

# Call the cleanup docker function
cleanup_docker

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

