set -ex

echo $USER
groups $USER
sudo usermod -aG docker buildkite-agent
sudo systemctl restart docker



# Build the docker image.
docker build -f Dockerfile.tpu -t vllm-tpu .