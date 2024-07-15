set -e

# Build the docker image.
docker build -f Dockerfile.tpu -t vllm-tpu .

# Set up cleanup.
remove_docker_container() { docker rm -f tpu-test || true; }
trap remove_docker_container EXIT
# Remove the container that might not be cleaned up in the previous run.
remove_docker_container

# For HF_TOKEN.
source /etc/environment
# Run a simple end-to-end example.
docker run --privileged --net host --shm-size=16G -it -e HF_TOKEN=$HF_TOKEN --name tpu-test vllm-tpu \
    python3 /workspace/vllm/examples/offline_inference_tpu.py
