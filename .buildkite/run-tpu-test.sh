set -ex

echo $USER
groups $USER
newgrp docker

# Build the docker image.
docker build -f Dockerfile.tpu -t vllm-tpu .