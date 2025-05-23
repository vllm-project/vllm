HF_TOKEN := $(shell cat token)
UID := $(shell id -u)
GID := $(shell id -g)
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
		--tag myimage$(IMAGE_TAG_SUFFIX) \
		..
endef

build-vllm-image:
	$(call add_local_user,flashinfer_vllm_dev:7204195724929729558)

vllm-setup:
	VLLM_USE_PRECOMPILED=1 pip install --editable .
	pip install "bok @ git+ssh://git@gitlab-master.nvidia.com:12051/jdebache/bok.git"
	pip install flashinfer-python --index-url https://gitlab-master.nvidia.com/api/v4/projects/179694/packages/pypi/simple

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
		myimage-dblanaru 

build-flashinfer-wheel:
	export FLASHINFER_ENABLE_AOT=1
	export TORCH_CUDA_ARCH_LIST='9.0+PTX'
	cd 3rdparty
	git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
	cd flashinfer
	git checkout v0.2.5 --recurse-submodules
	rm -rf build
	python3 setup.py bdist_wheel --dist-dir=dist --verbose

push-flashinfer-wheel:
	# pip install twine
	TWINE_PASSWORD=$(shell cat gitlab_token) \
	TWINE_USERNAME=scout \
	python3 -m twine upload --repository-url https://gitlab-master.nvidia.com/api/v4/projects/179694/packages/pypi 3rdparty/flashinfer/dist/* --verbose

vllm-sample:
	VLLM_ATTENTION_BACKEND=FLASHINFER python vllm_sample.py --model meta-llama/Llama-3.1-8B --enforce-eager --batch-size 3 --output-len 2 --num-iters 1 --num-iters-warmup 0 --prompts-file sample_prompts.txt
 

build-model8b-edgar4:
	python benchmarks/cpp/prepare_dataset.py --stdout --tokenizer=meta-llama/Llama-3.1-8B token-norm-dist --num-requests=30 --input-mean=2048 --output-mean=128 --input-stdev=0 --output-stdev=0  > ./tmp/synthetic_2048_128.txt
	trtllm-bench --workspace=./tmp --model meta-llama/Llama-3.1-8B build  --dataset ./tmp/synthetic_2048_128.txt

run-trtllm:
	make -C docker dan-vllm_run DOCKER_RUN_ARGS="-e HF_TOKEN=$(HF_TOKEN) -e HF_HOME=/code/tensorrt_llm/tmp/hf_cache" LOCAL_USER=1

trt-llm-setup:
	python scripts/build_wheel.py --use_ccache -p -a native
 