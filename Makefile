HF_TOKEN := $(shell cat token)

run-trtllm:
	make -C docker dan-vllm_run DOCKER_RUN_ARGS="-e HF_TOKEN=$(HF_TOKEN) -e HF_HOME=/code/tensorrt_llm/tmp/hf_cache" LOCAL_USER=1

trt-llm-setup:

	python scripts/build_wheel.py --use_ccache -p -a native
 
UID := $(shell id -u)
GID := $(shell id -g)
run-vllm:
	docker run -it --gpus all \
		-v $(shell pwd):/workspace \
		-v $(shell pwd)/vllm/utils/docker/:/dockercmd:ro \
		-v $(shell pwd)/tmp/hf_cache:/llm_cache/ \
		-v $(SSH_AUTH_SOCK):/ssh-agent \
		-e SSH_AUTH_SOCK=/ssh-agent \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e HF_HOME=/workspace/tmp/hf_cache \
		--ipc=host \
		-w /workspace \
		--entrypoint /bin/bash \
		vllm/vllm-openai:v0.8.4-dblanaru 

vllm-setup:
	VLLM_USE_PRECOMPILED=1 pip install --editable .

DOCKER_PROGRESS    ?= auto
USER_ID            ?= $(shell id --user)
USER_NAME          ?= $(shell id --user --name)
GROUP_ID           ?= $(shell id --group)
GROUP_NAME         ?= $(shell id --group --name)
IMAGE_TAG_SUFFIX   ?= -$(USER_NAME)
define add_local_user
	docker build \
		--progress $(DOCKER_PROGRESS) \
		--build-arg BASE_IMAGE_WITH_TAG=$(1) \
		--build-arg USER_ID=$(USER_ID) \
		--build-arg USER_NAME=$(USER_NAME) \
		--build-arg GROUP_ID=$(GROUP_ID) \
		--build-arg GROUP_NAME=$(GROUP_NAME) \
		--file Dockerfile \
		--tag $(1)$(IMAGE_TAG_SUFFIX) \
		..
endef

build-vllm:
	$(call add_local_user,vllm/vllm-openai:v0.8.4)
	
vllm-stuff:
	PYTHONPATH=. python3 benchmarks/benchmark_latency.py --model /trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct --enforce_eager --batch_size 1 --output_len 2 --num_iters 1 --num_iters_warmup 0
 

build-model8b-edgar4:
	python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=meta-llama/Llama-3.1-8B token-norm-dist --num-requests=30 --input-mean=2048 --output-mean=128 --input-stdev=0 --output-stdev=0  > ./tmp/synthetic_2048_128.txt
	trtllm-bench --workspace=./tmp --model meta-llama/Llama-3.1-8B build  --dataset ./tmp/synthetic_2048_128.txt