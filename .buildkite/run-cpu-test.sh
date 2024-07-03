# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

docker image prune -f

# Try building the docker image
docker build -t cpu-test -f Dockerfile.cpu .
docker build --build-arg VLLM_CPU_DISABLE_AVX512="true" -t cpu-test-avx2 -f Dockerfile.cpu .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test cpu-test-avx2 || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --cpuset-cpus=48-95 \
  --cpuset-mems=1 --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --name cpu-test cpu-test
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --cpuset-cpus=48-95 \
  --cpuset-mems=1 --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --name cpu-test-avx2 cpu-test-avx2

# offline inference
docker exec cpu-test bash -c "python3 examples/offline_inference.py"
docker exec cpu-test-avx2 bash -c "python3 examples/offline_inference.py"

# Run basic model test
docker exec cpu-test bash -c "cd tests;
  pip install pytest Pillow protobuf
  cd ../
  pytest -v -s tests/models -m \"not vlm\" --ignore=tests/models/test_embedding.py --ignore=tests/models/test_registry.py --ignore=tests/models/test_jamba.py" # Mamba on CPU is not supported

# online inference
docker exec cpu-test bash -c "
  export VLLM_CPU_KVCACHE_SPACE=10 
  export VLLM_CPU_OMP_THREADS_BIND=96-142 
  python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m & server_pid=$! 
  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json 
  timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
  python3 benchmarks/benchmark_serving.py \
    --backend vllm \
    --dataset-name sharegpt \
    --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json \
    --model facebook/opt-125m \
    --num-prompts 20 \
    --endpoint /v1/completions \
    --tokenizer facebook/opt-125m
  echo $server_pid && kill $server_pid"
