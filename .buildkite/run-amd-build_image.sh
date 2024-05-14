# This script builds the ROCm docker image
set -ex

# Print ROCm version
echo "--- ROCm info"
rocminfo

echo "--- Building container"
sha=$(git rev-parse --short HEAD)
image_name=rocm_${sha}
docker build \
        -t ${image_name} \
        -f Dockerfile.rocm \
        --progress plain \
        .

