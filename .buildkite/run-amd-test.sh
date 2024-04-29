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

docker build -t rocm_${1} -f Dockerfile.rocm .


case ${2} in
    1)
    # AMD Default Test
    ##################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_default || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_default rocm_${1} python3 -m vllm.entrypoints.api_server &

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
    curl -X POST -H "Content-Type: application/json" \
            localhost:8000/generate \
            -d '{"prompt": "San Francisco is a"}'
    ;;
    2)
    # AMD Regression Test
    #####################

    ;;
    3)
    # AMD AsyncEngine Test
    ######################

    ;;
    4)
    # AMD Basic Correctness Test
    ############################

    ;;
    5)
    # AMD Core Test
    ###############

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_core || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_core \
        rocm_${1} python3 -m pytest -v -s vllm/tests/core
    ;;
    6)
    # AMD Distributed Comm Ops Test
    ###############################

    ;;
    7)
    # AMD Distributed Test
    ######################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_distributed || true; \
                            docker rm -f rocm_${1}_test_basic_distributed_correctness_opt || true; \
                            docker rm -f rocm_${1}_test_basic_distributed_correctness_llama || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    #docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_distributed \
    #       	rocm  python3 -m pytest -v -s vllm/tests/distributed/test_pynccl.py

    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_basic_distributed_correctness_opt \
                rocm_${1}  /bin/bash -c "TEST_DIST_MODEL=facebook/opt-125m python3 -m pytest \
            -v -s vllm/tests/distributed/test_basic_distributed_correctness.py"

    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_basic_distributed_correctness_llama \
	        -e HF_TOKEN rocm_${1}  /bin/bash -c "TEST_DIST_MODEL=meta-llama/Llama-2-7b-hf python3 -m pytest \
		-v -s vllm/tests/distributed/test_basic_distributed_correctness.py"
    ;;
    8)
    # AMD Engine Test
    #################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_engine || true; \
                                docker rm -f rocm_${1}_test_tokenization || true; \
                                docker rm -f rocm_${1}_test_sequence || true; \
                                docker rm -f rocm_${1}_test_config || true;}
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    #docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_engine \
    #        rocm_${1} python3 -m pytest -v -s vllm/tests/engine
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_tokenization \
            -e HF_TOKEN rocm_${1} python3 -m pytest -v -s vllm/tests/tokenization
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_sequence \
            rocm_${1} python3 -m pytest -v -s vllm/tests/test_sequence.py
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_config \
            -e HF_TOKEN rocm_${1} python3 -m pytest -v -s vllm/tests/test_config.py
    ;;
    9)
    # AMD Entrypoints Test
    ######################

    ;;
    10)
    # AMD Example Test
    ##################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_offline_inference || true; \
                    docker rm -f rocm_${1}_test_offline_inference_with_prefix || true; \
                    docker rm -f rocm_${1}_test_llm_engine_example || true; \
                    docker rm -f rocm_${1}_test_llava_example || true;}
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host \
        --name rocm_${1}_test_offline_inference rocm_${1} python3 vllm/examples/offline_inference.py
    docker run --device /dev/kfd --device /dev/dri --network host \
        --name rocm_${1}_test_offline_inference_with_prefix rocm_${1} python3 vllm/examples/offline_inference_with_prefix.py
    docker run --device /dev/kfd --device /dev/dri --network host \
        --name rocm_${1}_test_llm_engine_example rocm_${1} python3 vllm/examples/llm_engine_example.py
    docker run --device /dev/kfd --device /dev/dri --network host \
        --name rocm_${1}_test_llava_example rocm_${1} /bin/bash -c "pip install awscli; python3 vllm/examples/llava_example.py"
    ;;
    11)
    # AMD Kernels Test
    ##################

    ;;
    12)
    # AMD Models Test
    #################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_models || true; \
                docker rm -f rocm_${1}_test_oot_registration || true; \
                docker rm -f rocm_${1}_test_models_py || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    #docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_models \
    #    -e HF_TOKEN rocm_${1} /bin/bash -c "cd vllm/tests; /bin/bash ../.buildkite/download-images.sh; \
    #    python3 -m pytest -v -s models --ignore=models/test_llava.py --ignore=models/test_mistral.py"

    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_oot_registration \
                rocm_${1} python3 -m pytest -v -s vllm/tests/models/test_oot_registration.py

    #docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_models_py \
    #                    -e HF_TOKEN rocm_${1} python3 -m pytest -v -s vllm/tests/models/test_models.py
    ;;
    13)
    # AMD Llava Test
    ################

    ;;
    14)
    # AMD Prefix Caching Test
    #########################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_prefix_caching || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_prefix_caching \
            rocm_${1} python3 -m pytest -v -s vllm/tests/prefix_caching 
    ;;
    15)
    # AMD Samplers Test
    ###################

    ;;
    16)
    # AMD LogitsProcessor Test
    ##########################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_logits_processor || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_logits_processor \
            rocm_${1} python3 -m pytest -v -s vllm/tests/test_logits_processor.py

    ;;

    17)
    # AMD Worker Test
    #################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_worker || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_worker \
            rocm_${1} python3 -m pytest -v -s vllm/tests/worker
    ;;
    18)
    # AMD Speculative Decoding Test
    ###############################
    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_speculative_decoding || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_speculative_decoding \
            -e HF_TOKEN rocm_${1} python3 -m pytest -v -s vllm/tests/spec_decode
    ;;
    19)
    # AMD LoRA Test
    ###############

    ;;
    20)
    # AMD Tensorizer Test
    #####################

    ;;
    21)
    # AMD Quantization Test
    #######################

    ;;
    22)
    # AMD Metrics Test
    ##################

    ;;
    23)
    # AMD Benchmarks Test
    #####################

    # Setup cleanup
    remove_docker_container() { docker rm -f rocm_${1}_test_benchmarks || true; }
    trap "remove_docker_container ${1}" EXIT
    remove_docker_container ${1}

    # Run the image
    docker run --device /dev/kfd --device /dev/dri --network host --name rocm_${1}_test_benchmarks \
        -e HF_TOKEN rocm_${1} /bin/bash -c "cd /vllm-workspace/.buildkite; /bin/bash run-benchmarks.sh"
    ;;
esac
