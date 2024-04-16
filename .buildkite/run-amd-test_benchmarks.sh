# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_benchmarks || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host --name rocm_test_benchmarks \
	-e HF_TOKEN rocm /bin/bash -c "/bin/bash vllm/.buildkite/run-benchmarks.sh"

