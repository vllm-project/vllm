#!/bin/bash
#
# This script adds Pascal GPU support to the VLLM OpenAI Docker image.
# It updates the CMakeLists.txt and Dockerfile files to include 6.0, 6.1 and 6.2.
#

# Ask user for confirmation
read -p "This script will add Pascal GPU support to vLLM. Continue? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting..."
    exit 1
fi
echo
echo "Adding Pascal GPU support..."

# Update CMakeLists.txt and Dockerfile
echo " - Updating CMakeLists.txt"
cuda_supported_archs="6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0"
sed -i.orig "s/set(CUDA_SUPPORTED_ARCHS \"7.0;7.5;8.0;8.6;8.9;9.0\")/set(CUDA_SUPPORTED_ARCHS \"$cuda_supported_archs\")/g" CMakeLists.txt

echo " - Updating Dockerfile"
torch_cuda_arch_list="6.0 6.1 6.2 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
sed -i.orig "s/ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'/ARG torch_cuda_arch_list='$torch_cuda_arch_list'/g" Dockerfile

cat <<EOF

You can now build from source with Pascal GPU support:
    pip install -e .

Or build the Docker image with:
    DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai

EOF
