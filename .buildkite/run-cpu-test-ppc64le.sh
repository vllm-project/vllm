# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t cpu-test -f Dockerfile.ppc64le .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image, setting --shm-size=4g for tensor parallel.
source /etc/environment
#docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true --network host -e HF_TOKEN --env VLLM_CPU_KVCACHE_SPACE=4 --shm-size=4g --name cpu-test cpu-test
docker run -itd --entrypoint /bin/bash -v ~/.cache/huggingface:/root/.cache/huggingface --privileged=true --network host -e HF_TOKEN=$HF_TOKEN --name cpu-test cpu-test

# Run basic model test
docker exec cpu-test bash -c "
  pip install pytest matplotlib einops transformers_stream_generator datamodel_code_generator
  pip install Pillow librosa
  pip install torchvision --index-url https://download.pytorch.org/whl/cpu
  pytest -v -s tests/models/embedding/language
  pytest -v -s tests/models/encoder_decoder/language
  pytest -v -s tests/models/decoder_only/language/test_models.py
  pytest -v -s tests/models/decoder_only/audio_language -m core_model
  pytest -v -s tests/models/decoder_only/vision_language -m core_model"

# online inference
docker exec cpu-test bash -c "
  python3 -m vllm.entrypoints.openai.api_server --model facebook/opt-125m & 
  timeout 600 bash -c 'until curl localhost:8000/v1/models; do sleep 1; done' || exit 1
  python3 benchmarks/benchmark_serving.py \
    --backend vllm \
    --dataset-name random \
    --model facebook/opt-125m \
    --num-prompts 20 \
    --endpoint /v1/completions \
    --tokenizer facebook/opt-125m"
