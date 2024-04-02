# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t cpu-test -f Dockerfile.cpu .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and launch offline inference
docker run --network host --env VLLM_CPU_KVCACHE_SPACE=1 --name cpu-test cpu-test python3 examples/offline_inference.py
