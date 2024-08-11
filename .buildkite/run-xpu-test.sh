# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t xpu-test -f Dockerfile.xpu .

# Setup cleanup
remove_docker_container() { docker rm -f xpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and launch offline inference
docker run --network host --name xpu-test --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path xpu-test python3 examples/offline_inference.py
