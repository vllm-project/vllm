# --8<-- [start:installation]

vLLM supports AMD GPUs with ROCm 6.3 or above. Pre-built wheels are available for ROCm 7.0.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU: MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201), Ryzen AI MAX / AI 300 Series (gfx1151/1150)
- ROCm 6.3 or above
    - MI350 requires ROCm 7.0 or above
    - Ryzen AI MAX / AI 300 Series requires ROCm 7.0.2 or above

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

The vLLM wheel bundles PyTorch and all required dependencies, and you should use the included PyTorch for compatibility. Because vLLM compiles many ROCm kernels to ensure a validated, high‑performance stack, the resulting binaries may not be compatible with other ROCm or PyTorch builds.
If you need a different ROCm version or want to use an existing PyTorch installation, you’ll need to build vLLM from source.  See [below](#build-wheel-from-source) for more details.

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

To install the latest version of vLLM for Python 3.12, ROCm 7.0 and `glibc >= 2.35`.

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

!!! tip
    You can find out about which ROCm version the latest vLLM supports by checking the index in extra-index-url [https://wheels.vllm.ai/rocm/](https://wheels.vllm.ai/rocm/) .

To install a specific version and ROCm variant of vLLM wheel.

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.15.0/rocm700
```

!!! warning "Caveats for using `pip`" 

    We recommend leveraging `uv` to install vLLM wheel. Using `pip` to install from custom indices is cumbersome, because `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install wheel from custom index if exact versions of all packages are specified exactly. In contrast, `uv` gives the extra index [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

    If you insist on using `pip`, you have to specify the exact vLLM version and full URL of the wheel path `https://wheels.vllm.ai/rocm/<version>/<rocm-variant>` (which can be obtained from the web page).

    ```bash
    pip install vllm==0.15.0+rocm700 --extra-index-url https://wheels.vllm.ai/rocm/0.15.0/rocm700
    ```

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

!!! tip
    - If you found that the following installation step does not work for you, please refer to [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base). Dockerfile is a form of installation steps.

0. Install prerequisites (skip if you are already in an environment/docker with the following installed):

    - [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
    - [PyTorch](https://pytorch.org/)

    For installing PyTorch, you can start from a fresh docker image, e.g, `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0`, `rocm/pytorch-nightly`. If you are using docker image, you can skip to Step 3.

    Alternatively, you can install PyTorch using PyTorch wheels. You can check PyTorch installation guide in PyTorch [Getting Started](https://pytorch.org/get-started/locally/). Example:

    ```bash
    # Install PyTorch
    pip uninstall torch -y
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
    ```

1. Install [Triton for ROCm](https://github.com/ROCm/triton.git)

    Install ROCm's Triton following the instructions from [ROCm/triton](https://github.com/ROCm/triton.git)

    ```bash
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton.git
    cd triton
    # git checkout $TRITON_BRANCH
    git checkout f9e5bf54
    if [ ! -f setup.py ]; then cd python; fi
    python3 setup.py install
    cd ../..
    ```

    !!! note
        - The validated `$TRITON_BRANCH` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).
        - If you see HTTP issue related to downloading packages during building triton, please try again as the HTTP error is intermittent.

2. Optionally, if you choose to use CK flash attention, you can install [flash attention for ROCm](https://github.com/Dao-AILab/flash-attention.git)

    Install ROCm's flash attention (v2.8.0) following the instructions from [ROCm/flash-attention](https://github.com/Dao-AILab/flash-attention#amd-rocm-support)

    For example, for ROCm 7.0, suppose your gfx arch is `gfx942`. To get your gfx architecture, run `rocminfo |grep gfx`.

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    # git checkout $FA_BRANCH
    git checkout 0e60e394
    git submodule update --init
    GPU_ARCHS="gfx942" python3 setup.py install
    cd ..
    ```

    !!! note
        - The validated `$FA_BRANCH` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).


3. Optionally, if you choose to build AITER yourself to use a certain branch or commit, you can build AITER using the following steps:

    ```bash
    python3 -m pip uninstall -y aiter
    git clone --recursive https://github.com/ROCm/aiter.git
    cd aiter
    git checkout $AITER_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    python3 setup.py develop
    ```

    !!! note
        - You will need to config the `$AITER_BRANCH_OR_COMMIT` for your purpose.
        - The validated `$AITER_BRANCH_OR_COMMIT` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).


4. Optionally, if you want to use MORI for EP or PD disaggregation, you can install [MORI](https://github.com/ROCm/mori) using the following steps:

    ```bash
    git clone https://github.com/ROCm/mori.git
    cd mori
    git checkout $MORI_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    MORI_GPU_ARCHS="gfx942;gfx950" python3 setup.py install
    ```

    !!! note
        - You will need to config the `$MORI_BRANCH_OR_COMMIT` for your purpose.
        - The validated `$MORI_BRANCH_OR_COMMIT` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).


5. Build vLLM. For example, vLLM on ROCM 7.0 can be built with the following steps:

    ???+ console "Commands"

        ```bash
        pip install --upgrade pip

        # Build & install AMD SMI
        pip install /opt/rocm/share/amd_smi

        # Install dependencies
        pip install --upgrade numba \
            scipy \
            huggingface-hub[cli,hf_transfer] \
            setuptools_scm
        pip install -r requirements/rocm.txt

        # To build for a single architecture (e.g., MI300) for faster installation (recommended):
        export PYTORCH_ROCM_ARCH="gfx942"

        # To build vLLM for multiple arch MI210/MI250/MI300, use this instead
        # export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

        python3 setup.py develop
        ```

    This may take 5-10 minutes. Currently, `pip install .` does not work for ROCm when installing vLLM from source.

    !!! tip
        - The ROCm version of PyTorch, ideally, should match the ROCm driver version.

!!! tip
    - For MI300x (gfx942) users, to achieve optimal performance, please refer to [MI300x tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html) for performance optimization and tuning tips on system and workflow level.
      For vLLM, please refer to [vLLM performance optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html).

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

vLLM offers an official Docker image for deployment.
The image can be used to run OpenAI compatible server and is available on Docker Hub as [vllm/vllm-openai-rocm](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags).

???+ console "Commands"
    ```bash
    docker run --rm \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        -p 8000:8000 \
        --ipc=host \
        vllm/vllm-openai-rocm:latest \
        --model Qwen/Qwen3-0.6B
    ```

# --8<-- [end:pre-built-images]
# --8<-- [start:use-custom-docker-image]

To run vLLM with the custom-built Docker image:

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm <args...>
```

The argument `vllm/vllm-openai-rocm` specifies the image to run, and should be replaced with the name of the custom-built image (the `-t` tag from the build command).

To use the docker image as base for development, you can launch it in interactive session through overriding the entrypoint.

???+ console "Commands"
    ```bash
    docker run --rm -it \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --network=host \
        --ipc=host \
        --entrypoint bash \
        vllm/vllm-openai-rocm
    ```

# --8<-- [end:use-custom-docker-image]
# --8<-- [start:amd-prebuilt-images]

Prior to January 20th, 2026 when the official docker images are available on [upstream vllm docker hub](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/), the [AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) offers a prebuilt, optimized
docker image designed for validating inference performance on the AMD Instinct MI300X™ accelerator.
AMD also offers nightly prebuilt docker image from [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev), which has vLLM and all its dependencies installed. The entrypoint of this docker image is `/bin/bash` (different from the vLLM's Official Docker Image).

???+ console "Commands"
    ```bash
    docker pull rocm/vllm-dev:nightly # to get the latest image
    docker run -it --rm \
    --network=host \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v <path/to/your/models>:/app/models \
    -e HF_HOME="/app/models" \
    rocm/vllm-dev:nightly
    ```

!!! tip
    Please check [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)
    for instructions on how to use this prebuilt docker image.

# --8<-- [end:amd-prebuilt-images]
# --8<-- [start:build-image-from-source]

You can build and run vLLM from source via the provided [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm).

??? info "(Optional) Build an image with ROCm software stack"

    Build a docker image from [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) which setup ROCm software stack needed by the vLLM.
    **This step is optional as this rocm_base image is usually prebuilt and store at [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) under tag `rocm/vllm-dev:base` to speed up user experience.**
    If you choose to build this rocm_base image yourself, the steps are as follows.

    It is important that the user kicks off the docker build using buildkit. Either the user put `DOCKER_BUILDKIT=1` as environment variable when calling docker build command, or the user needs to set up buildkit in the docker daemon configuration `/etc/docker/daemon.json` as follows and restart the daemon:

    ```json
    {
        "features": {
            "buildkit": true
        }
    }
    ```

    To build vllm on ROCm 7.0 for MI200 and MI300 series, you can use the default:

    ```bash
    DOCKER_BUILDKIT=1 docker build \
        -f docker/Dockerfile.rocm_base \
        -t rocm/vllm-dev:base .
    ```

First, build a docker image from [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) and launch a docker container from the image.
It is important that the user kicks off the docker build using buildkit. Either the user put `DOCKER_BUILDKIT=1` as environment variable when calling docker build command, or the user needs to set up buildkit in the docker daemon configuration /etc/docker/daemon.json as follows and restart the daemon:

```json
{
    "features": {
        "buildkit": true
    }
}
```

[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) uses ROCm 7.0 by default, but also supports ROCm 5.7, 6.0, 6.1, 6.2, 6.3, and 6.4, in older vLLM branches.
It provides flexibility to customize the build of docker image using the following arguments:

- `BASE_IMAGE`: specifies the base image used when running `docker build`. The default value `rocm/vllm-dev:base` is an image published and maintained by AMD. It is being built using [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base)
- `ARG_PYTORCH_ROCM_ARCH`: Allows to override the gfx architecture values from the base docker image

Their values can be passed in when running `docker build` with `--build-arg` options.

To build vllm on ROCm 7.0 for MI200 and MI300 series, you can use the default (which build a docker image with `vllm serve` as entrypoint):


```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vllm-openai-rocm .
```

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

See [Feature x Hardware](../../features/README.md#feature-x-hardware) compatibility matrix for feature support information.

# --8<-- [end:supported-features]
