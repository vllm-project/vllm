# This script build the ROCm docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Print ROCm version
rocminfo

echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

# Try building the docker image
docker build -t rocm -f Dockerfile.rocm .

# Setup cleanup
remove_docker_container() { docker rm -f rocm_test_offline_inference || true; \
			    docker rm -f rocm_test_offline_inference_with_prefix || true; \
			    docker rm -f rocm_test_llm_engine_example || true; \
			    docker rm -f rocm_test_llava_example || true;}
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run --device /dev/kfd --device /dev/dri --network host \
	--name rocm_test_offline_inference rocm python3 vllm/examples/offline_inference.py
docker run --device /dev/kfd --device /dev/dri --network host \
	--name rocm_test_offline_inference_with_prefix rocm python3 vllm/examples/offline_inference_with_prefix.py
docker run --device /dev/kfd --device /dev/dri --network host \
	--name rocm_test_llm_engine_example rocm python3 vllm/examples/llm_engine_example.py
docker run --device /dev/kfd --device /dev/dri --network host \
	--name rocm_test_llava_example rocm /bin/bash -c "pip install awscli; python3 vllm/examples/llava_example.py"

