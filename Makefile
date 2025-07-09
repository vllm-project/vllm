HF_TOKEN := $(shell cat token)
UID := $(shell id -u)
GID := $(shell id -g)
DOCKER_PROGRESS    ?= auto
USER_ID            ?= $(shell id --user)
USER_NAME          ?= $(shell id --user --name)
GROUP_ID           ?= $(shell id --group)
GROUP_NAME         ?= $(shell id --group --name)
IMAGE_TAG_SUFFIX   ?= -$(USER_NAME)

NSYS_PROFILE ?= 0
ifeq ($(NSYS_PROFILE), 1)
	NSYS_PROFILE_CMD := nsys profile -o vllm-sample-profile.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node --force-overwrite=true
else
	NSYS_PROFILE_CMD :=
endif

define add_local_user
	docker build \
		--progress $(DOCKER_PROGRESS) \
		--build-arg BASE_IMAGE_WITH_TAG=$(1) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg GROUP_NAME=$(GROUP_NAME) \
		--file Dockerfile \
		--tag myimage$(IMAGE_TAG_SUFFIX) \
		..
endef

build-vllm-image:
	$(call add_local_user,flashinfer_vllm_dev:7204195724929729558)

vllm-setup:
	git clone hypdeb/vllm
	git checkout dan_branch
	VLLM_USE_PRECOMPILED=1 pip install --editable .
	pip install flashinfer-python --index-url https://gitlab-master.nvidia.com/api/v4/projects/179694/packages/pypi/simple

force-reinstall-tke:
	pip install "trtllm-kernel-export @ git+ssh://git@gitlab.com:nvidia/tensorrt-llm/private/tensorrt-llm-kernel-export.git" --force-reinstall --no-input
run-vllm:
	docker run -it --gpus all \
		-v $(shell pwd):$(shell pwd) \
		-v $(shell pwd)/vllm/utils/docker/:/dockercmd:ro \
		-v $(shell pwd)/tmp/hf_cache:/llm_cache/ \
		-v $(SSH_AUTH_SOCK):/ssh-agent \
		-e SSH_AUTH_SOCK=/ssh-agent \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e HF_HOME=$(shell pwd)/tmp/hf_cache \
		--ipc=host \
		-w $(shell pwd) \
		--entrypoint /bin/bash \
		myimage-dblanaru 

build-flashinfer-wheel:
	export FLASHINFER_ENABLE_AOT=1; \
	export TORCH_CUDA_ARCH_LIST='9.0+PTX'; \
	cd 3rdparty; \
	rm -rf flashinfer; \
	git clone https://github.com/flashinfer-ai/flashinfer.git --recursive; \
	cd flashinfer; \
	git checkout v0.2.6.post1 --recurse-submodules; \
	pip install --no-build-isolation --verbose .; \
	pip install build ;\
	python -m flashinfer.aot ;\
	python -m build --no-isolation --wheel

push-flashinfer-wheel:
	# pip install twine
	TWINE_PASSWORD=$(shell cat gitlab_token) \
	TWINE_USERNAME=scout \
	python3 -m twine upload --repository-url https://gitlab-master.nvidia.com/api/v4/projects/179694/packages/pypi 3rdparty/flashinfer/dist/* --verbose

vllm-sample-flashinfer:
	VLLM_ATTENTION_BACKEND=FLASHINFER python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Meta-Llama-3.1-8B --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt

vllm-sample-flashattn:
	VLLM_ATTENTION_BACKEND=FLASH_ATTN python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt

vllm-sample:
	VLLM_ATTENTION_BACKEND=TKE python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt 

vllm-sample-70b:
	VLLM_ATTENTION_BACKEND=TKE python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-70B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt 

vllm-sample-70b-tp4:
	VLLM_ATTENTION_BACKEND=TKE python vllm_sample.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-70B-Instruct-FP8 --enforce-eager --batch-size 3 --output-len 10 --num-iters 1 --num-iters-warmup 0 --prompts-file z_hacky_layer_test/sample_prompts.txt --tp 4

build-model8b-edgar4:
	python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=meta-llama/Llama-3.1-8B token-norm-dist --num-requests=30 --input-mean=2048 --output-mean=128 --input-stdev=0 --output-stdev=0  > ./tmp/synthetic_2048_128.txt
	trtllm-bench --workspace=./tmp --model meta-llama/Llama-3.1-8B build  --dataset ./tmp/synthetic_2048_128.txt

run-trtllm:
	make -C docker dan-vllm_run DOCKER_RUN_ARGS="-e HF_TOKEN=$(HF_TOKEN) -e HF_HOME=/code/tensorrt_llm/tmp/hf_cache" LOCAL_USER=1

run-base-vllm:
	docker rm -f vllm_flashinfer
	docker run --name vllm_flashinfer --gpus all --ipc host --shm-size 1g -e VLLM_ATTENTION_BACKEND=FLASHINFER -e HF_TOKEN=$(HF_TOKEN) -v $(shell pwd):$(shell pwd) -e HF_HOME=$(shell pwd)/tmp/hf_cache -p 8000:8000 vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B --dtype float16 --chat-template $(shell pwd)/examples/tool_chat_template_llama3.1_json.jinja
	docker logs -f vllm_flashinfer

trt-llm-setup:
	python scripts/build_wheel.py --use_ccache -p -a native

benchmark-latency: TKE_BACKEND := TKE
benchmark-latency: FLASH_BACKEND := FLASH_ATTN
benchmark-latency: MODEL_PATH := /scratch/usr/quantized_model
benchmark-latency: TP_SIZE := 4
benchmark-latency: INPUT_LEN := 80000
benchmark-latency: MAX_MODEL_LEN := 131072
benchmark-latency: NUM_ITERS_WARMUP := 1
benchmark-latency: BATCH_SIZE := 2
benchmark-latency: NUM_ITERS := 25

# Note: we need VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and to specify the max model length explicitly because vLLM reads the max model length from the tokenizer for some reason.
# It is unclear why the tokenizer would need to know the max model length at all, even less clear why this would be the source of truth for model length.
# I guess there is some explanation for this, but I am clueless at this time.
benchmark-latency:
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 LLM_ATTENTION_BACKEND=$(TKE_BACKEND) python benchmarks/benchmark_latency.py \
		--model $(MODEL_PATH) \
		--tensor-parallel-size $(TP_SIZE) \
		--quantization modelopt \
		--input-len $(INPUT_LEN) \
		--max-model-len $(MAX_MODEL_LEN) \
		--num-iters-warmup $(NUM_ITERS_WARMUP) \
		--batch-size $(BATCH_SIZE) \
		--num-iters $(NUM_ITERS) \
		--kv-cache-dtype fp8 \
		--enforce-eager
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 LLM_ATTENTION_BACKEND=$(FLASH_BACKEND) python benchmarks/benchmark_latency.py \
		--model $(MODEL_PATH) \
		--tensor-parallel-size $(TP_SIZE) \
		--quantization modelopt \
		--input-len $(INPUT_LEN) \
		--max-model-len $(MAX_MODEL_LEN) \
		--num-iters-warmup $(NUM_ITERS_WARMUP) \
		--batch-size $(BATCH_SIZE) \
		--num-iters $(NUM_ITERS) \
		--kv-cache-dtype fp8 \
		--enforce-eager

delete-vllm-cache:
	rm -rf ~/.cache/vllm

vllm-sample-flashinfer-v1: delete-vllm-cache
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=FLASHINFER_VLLM_V1 $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 1 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--tensor-parallel-size 4 > flashinfer.txt 2>&1

vllm-sample-tke: delete-vllm-cache
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=TKE $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 1 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--tensor-parallel-size 4 > tke.txt 2>&1

vllm-sample-flash-attn: delete-vllm-cache
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1 $(NSYS_PROFILE_CMD) python vllm_sample.py \
	--model /scratch/usr/quantized_model/ \
	--batch-size 1 \
	--prompts-file sample_prompts.txt \
	--num-iters 1 \
	--num-iters-warmup 0 \
	--kv-cache-dtype fp8 \
	--tensor-parallel-size 4 > flash_attn.txt 2>&1

all-samples: vllm-sample-flash-attn vllm-sample-tke vllm-sample-flashinfer-v1

install-lm-eval:
	git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
	cd lm-evaluation-harness
	pip install -e .

lm-eval-hellaswag:
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=TKE lm_eval \
		--model vllm \
		--tasks hellaswag \
		--log_samples \
		--output_path /scratch/usr/lm-eval-results-hellaswag.json \
		--model_args "pretrained=/scratch/usr/quantized_model,tensor_parallel_size=4,quantization=modelopt"

lm-eval-tiny-hellaswag-tke:
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=TKE lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--log_samples \
		--batch_size 4 \
		--output_path /scratch/usr/lm-eval-results-tiny-hellaswag.json \
		--model_args "pretrained=/scratch/usr/quantized_model,tensor_parallel_size=4,quantization=modelopt,gpu_memory_utilization=0.95,kv_cache_dtype=fp8,block_size=32,enforce_eager=True"

lm-eval-tiny-hellaswag-flash-attn:
	VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN lm_eval \
		--model vllm \
		--tasks tinyHellaswag \
		--log_samples \
		--batch_size 1 \
		--output_path /scratch/usr/lm-eval-results-tiny-hellaswag.json \
		--model_args "pretrained=/scratch/usr/llama-8b-fp8,tensor_parallel_size=1,quantization=modelopt,gpu_memory_utilization=0.95,enforce_eager=True"

lm-eval-gsm8k:
	CUDA_LAUNCH_BLOCKING=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ATTENTION_BACKEND=TKE lm_eval \
		--model vllm \
		--tasks gsm8k \
		--log_samples \
		--output_path /scratch/usr/lm-eval-results-gsm8k.json \
		--model_args "pretrained=/scratch/usr/quantized_model,tensor_parallel_size=4,quantization=modelopt"

make send-requests:
	bash -c '
	curl -s -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d "{\"model\":\"meta-llama/Llama-3.1-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a short poem about the ocean.\"}],\"max_tokens\":10}" 

	curl -s -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d "{\"model\":\"meta-llama/Llama-3.1-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain quantum entanglement in simple terms.\"}],\"max_tokens\":10}" 

	curl -s -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d "{\"model\":\"meta-llama/Llama-3.1-8B\",\"messages\":[{\"role\":\"user\",\"content\":\"List three benefits of exercise.\"}],\"max_tokens\":10}" 

	wait
	echo "Request 1:"; cat req1.json; echo
	echo "Request 2:"; cat req2.json; echo
	echo "Request 3:"; cat req3.json
	';