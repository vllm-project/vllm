set -ex

echo $USER
echo groups $USER

# Build the docker image.
docker build -f Dockerfile.tpu -t vllm-tpu .