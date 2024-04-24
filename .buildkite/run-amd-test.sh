# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host --name rocm rocm python3 -m vllm.entrypoints.api_server &

# Wait for the server to start
wait_for_server_to_start() {
    timeout=300
    counter=0

    while [ "$(curl -s -o /dev/null -w ''%{http_code}'' localhost:8000/health)" != "200" ]; do
        sleep 1
        counter=$((counter + 1))
        if [ $counter -ge $timeout ]; then
            echo "Timeout after $timeout seconds"
            break
        fi
    done
}
wait_for_server_to_start

# Test a simple prompt
curl -X POST -H "Content-Type: application/json" \
    localhost:8000/generate \
    -d '{"prompt": "San Francisco is a"}'
