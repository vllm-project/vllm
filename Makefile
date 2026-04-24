IMAGE_NAME ?= us-central1-docker.pkg.dev/cohere-artifacts/cohere/vllm
COMMIT_SHA ?= $(shell git rev-parse HEAD)
PUSH_TAG ?=
DEPOT_PROJECT ?=
TORCH_CUDA_ARCH_LIST ?=

# the final image is always tagged with sha, which is used to check if image already built.
NVIDIA_TAGS := -t ${IMAGE_NAME}-nvidia:${COMMIT_SHA}
AMD_TAGS := -t ${IMAGE_NAME}-rocm:${COMMIT_SHA}
CPU_TAGS := -t ${IMAGE_NAME}-cpu:${COMMIT_SHA}

# base images are tagged with base-<sha> in the same repo as final images
NVIDIA_BASE_IMAGE ?= ${IMAGE_NAME}-nvidia:base-${COMMIT_SHA}
AMD_BASE_IMAGE ?= ${IMAGE_NAME}-rocm:base-${COMMIT_SHA}

# ROCm base image built from Dockerfile.rocm_base, tagged by content hash.
# Only rebuilt when Dockerfile.rocm_base changes (e.g. PyTorch/Triton pin bumps).
ROCM_BASE_HASH := $(shell sha256sum docker/Dockerfile.rocm_base | cut -c1-12)
ROCM_BASE_IMAGE := ${IMAGE_NAME}-rocm-base:${ROCM_BASE_HASH}

# on push, we also tag with PUSH_TAG (release version)
ifdef PUSH_TAG
NVIDIA_TAGS += -t ${IMAGE_NAME}-nvidia:${PUSH_TAG}
AMD_TAGS += -t ${IMAGE_NAME}-rocm:${PUSH_TAG}
CPU_TAGS += -t ${IMAGE_NAME}-cpu:${PUSH_TAG}
endif

NVIDIA_CUSTOM_TAG :=
AMD_CUSTOM_TAG :=
CPU_CUSTOM_TAG :=
ifdef CUSTOM_TAG
NVIDIA_CUSTOM_TAG = -t ${IMAGE_NAME}-nvidia:${CUSTOM_TAG}
AMD_CUSTOM_TAG = -t ${IMAGE_NAME}-rocm:${CUSTOM_TAG}
CPU_CUSTOM_TAG = -t ${IMAGE_NAME}-cpu:${CUSTOM_TAG}
endif

NVIDIA_TORCH_CUDA_ARCH_LIST_ARG :=
ifdef TORCH_CUDA_ARCH_LIST
NVIDIA_TORCH_CUDA_ARCH_LIST_ARG = --build-arg torch_cuda_arch_list="${TORCH_CUDA_ARCH_LIST}"
endif

# build vllm in the container with `python3 setup_cython.py build_ext --inplace`
# in order for development of vllm-rocm, in addition to the above stemp you would require run `python setup.py develop`
dev-amd: build
	@docker run --rm -it --name vllm-rocm \
		--device=/dev/kfd \
		--device=/dev/dri\
		--group-add=video\
		--cap-add=SYS_PTRACE \
		--security-opt seccomp=unconfined \
		--network=host \
		-v /mnt/storage:/mnt/storage \
		-v ${PWD}:/workspace \
		-w /workspace \
		${IMAGE_NAME}:${COMMIT_SHA} \
		bash

# build vllm with https://docs.vllm.ai/en/stable/getting_started/installation.html#troubleshooting
dev:
	@docker run --rm -it --name vllm \
		--gpus all \
		--network=host \
		--shm-size=1G \
		-v /mnt/storage:/mnt/storage \
		-v ${PWD}:/workspace \
		-w /workspace \
		-e MAX_JOBS=64 \
		-e NVCC_THREADS=16 \
		nvcr.io/nvidia/pytorch:23.10-py3 \
		bash

# TODO: doesn't seem easy to avoid the first --push param
build-vllm-nvidia:
	@depot build \
		--project=$(DEPOT_PROJECT) \
		--build-arg USE_SCCACHE=${USE_SCCACHE} \
		--build-arg SCCACHE_WEBDAV_ENDPOINT=${SCCACHE_WEBDAV_ENDPOINT} \
		--build-arg SCCACHE_WEBDAV_TOKEN=${SCCACHE_WEBDAV_TOKEN} \
		--build-arg SCCACHE_WEBDAV_KEY_PREFIX=${SCCACHE_WEBDAV_KEY_PREFIX} \
		--build-arg PYTHON_VERSION=3.12 \
		--build-arg CCACHE_DIR=/root/.cache/ccache \
		--build-arg RUN_WHEEL_CHECK=false \
		--build-arg max_jobs=64 \
		--build-arg nvcc_threads=16 \
		--build-arg VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED} \
		${NVIDIA_TORCH_CUDA_ARCH_LIST_ARG} \
		-t ${IMAGE_NAME}-nvidia:base-${COMMIT_SHA} \
		--target vllm-base \
		--platform linux/amd64,linux/arm64 \
		-f docker/Dockerfile \
		--push \
		.

# docker/Dockerfile.cohere always reinstalls vLLM in editable mode.
build-cohere-nvidia:
	@depot build \
		--project=$(DEPOT_PROJECT) \
		--build-arg "BASE_IMAGE=${NVIDIA_BASE_IMAGE}" \
		${NVIDIA_TAGS} ${NVIDIA_CUSTOM_TAG} \
		--platform linux/amd64,linux/arm64 \
		-f docker/Dockerfile.cohere \
		--push \
		.

build-and-push-nvidia: build-vllm-nvidia build-cohere-nvidia

build-rocm-base:
	@if docker manifest inspect ${ROCM_BASE_IMAGE} > /dev/null 2>&1; then \
		echo "ROCm base image already exists: ${ROCM_BASE_IMAGE}"; \
	else \
		echo "Building ROCm base image: ${ROCM_BASE_IMAGE}" \
		&& depot build \
			--project=${DEPOT_PROJECT} \
			-t ${ROCM_BASE_IMAGE} \
			-f docker/Dockerfile.rocm_base \
			--push \
			.; \
	fi

build-vllm-amd: build-rocm-base
	@depot build \
		--project=${DEPOT_PROJECT} \
		--build-arg "BASE_IMAGE=${ROCM_BASE_IMAGE}" \
		--build-arg "ARG_PYTORCH_ROCM_ARCH=gfx942" \
		--build-arg "USE_CYTHON=0" \
		-t ${IMAGE_NAME}-rocm:base-${COMMIT_SHA} \
		-f docker/Dockerfile.rocm \
		--push \
		.

# docker/Dockerfile.cohere always reinstalls vLLM in editable mode.
build-cohere-amd:
	@depot build \
		--project=${DEPOT_PROJECT} \
		--build-arg "BASE_IMAGE=${AMD_BASE_IMAGE}" \
		${AMD_TAGS} ${AMD_CUSTOM_TAG} \
		-f docker/Dockerfile.cohere \
		--push \
		.

build-and-push-amd: build-vllm-amd build-cohere-amd

build-and-push-cpu:
	@depot build \
		--project=$(DEPOT_PROJECT) \
		${CPU_TAGS} ${CPU_CUSTOM_TAG} \
		--build-arg VLLM_CPU_DISABLE_AVX512=true \
		--target vllm-test \
		-f docker/Dockerfile.cpu \
		--push \
		.
