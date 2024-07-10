set -ex

echo $USER
groups $USER

# Build the docker image.
docker build -f Dockerfile.tpu -t vllm-tpu .