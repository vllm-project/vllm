# --8<-- [start:installation]

vLLM supports AMD GPUs with ROCm 6.3 or above, and torch 2.8.0 and above.

!!! tip
    [Docker](#set-up-using-docker) is the recommended way to use vLLM on ROCm.

!!! warning
    There are no pre-built wheels for this device, so you must either use the pre-built Docker image or build vLLM from source.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU: MI210/MI250 (gfx90a), MI300/MI325 (gfx942), MI350/MI355 (gfx950), Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201)
- ROCm 6.0 or above
    - MI350/MI355 requires ROCm 7.0 or above

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

There is no extra information on creating a new Python environment for this device.

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Currently, there are no pre-built ROCm wheels.

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

!!! tip
    - If you find that the following installation steps do not work for you, please refer to [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) for the latest installation steps used to build the nightly/release images for ROCm.

1. Install prerequisites (skip if you are already in an environment/docker with the following installed):

    - [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
    - [PyTorch](https://pytorch.org/)

    For installing PyTorch and Triton, you can start from a fresh docker image, e.g, `rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0`, `rocm/pytorch-nightly`. If you are using docker image, you can skip to Step 3.

    Alternatively, you can install PyTorch using PyTorch wheels. You can check PyTorch installation guide in PyTorch [Getting Started](https://pytorch.org/get-started/locally/). Example:

    ```bash
    # Install PyTorch
    pip uninstall torch -y
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
    ```

    !!! note
        For ROCm 7.0 wheels (with gfx950 support), please use the PyTorch nightly build:
        ```bash
        pip install --no-cache-dir --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0
        ```

2. Install [Triton](https://github.com/triton-lang/triton)

    Install Triton on ROCm following the instructions from [ROCm/triton](https://github.com/ROCm/triton):

    ```bash
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton
    cd triton
    # git checkout TRITON_BRANCH
    git checkout 57c693b6
    pip install -r python/requirements.txt
    pip install .
    if [ -d python/triton_kernels ]; then
        cd python/triton_kernels
        pip install .
        cd ../../
    fi
    cd ../
    ```

    !!! note
        - The validated `$TRITON_BRANCH` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).
        - If you see HTTP issue related to downloading packages during building triton, please try again as the HTTP error is intermittent.

3. Optionally, install [flash attention for ROCm](https://github.com/Dao-AILab/flash-attention)

    Install ROCm's latest flash attention (v2.8.3) following the instructions from [ROCm/flash-attention](https://github.com/Dao-AILab/flash-attention#amd-rocm-support):

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    # git checkout $FA_BRANCH
    git checkout 0e60e394
    git submodule update --init
    GPU_ARCHS="gfx942" python3 setup.py install
    cd ..
    ```

    To get your gfx architecture, run `rocminfo | grep gfx`. Alternatively, wheels intended for vLLM use can be accessed under the releases.

    !!! note
        - The validated `$FA_BRANCH` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).


4. Optionally, to use AITER kernels, you can install [AITER](https://github.com/ROCm/aiter); see [Dockerfile.rocm_base](../../../docker/Dockerfile.rocm_base) for the most recent commit:

    ```bash
    python3 -m pip uninstall -y aiter
    git clone --recursive https://github.com/ROCm/aiter.git
    cd aiter
    # git checkout AITER_BRANCH
    git checkout 9716b1b8
    git submodule sync; git submodule update --init --recursive
    python3 setup.py develop
    ```

    !!! note
        - The validated `$AITER_BRANCH` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).
        

4. Build vLLM. For example, vLLM on ROCM 7.0 can be built with the following steps:

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

        # To build for a single architecture (e.g., MI300+MI325) for faster installation (recommended):
        export PYTORCH_ROCM_ARCH="gfx942"

        # To build vLLM for multiple arch MI210/MI250/MI300/MI325/MI350/MI355, use this instead
        # export PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950"

        python3 setup.py develop
        ```

    This may take 5-10 minutes. Currently, `pip install .` does not work for ROCm installation.

    !!! tip
        - The ROCm version of PyTorch, ideally, should match the ROCm driver version.

!!! tip
    - For MI300x (gfx942) users, to achieve optimal performance, please refer to [MI300x tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html) for performance optimization and tuning tips on system and workflow level.
      For vLLM, please refer to [vLLM performance optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html).

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

```bash
docker pull rocm/vllm:latest
```

The [rocm/vllm](https://hub.docker.com/r/rocm/vllm/tags) Dockerhub repository offers a prebuilt, optimized
docker image designed for validating inference performance on the AMD Instinct™ MI300X, MI325X, MMI350X, and MI355X accelerators.

AMD also offers nightly prebuilt docker images at [rocm/vllm-dev](https://hub.docker.com/r/rocm/vllm-dev), which have vLLM and all its dependencies installed.

```bash
docker pull rocm/vllm-dev:nightly
```

!!! tip
    Please check [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)
    for instructions on how to use this prebuilt docker image.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

#### (Optional) Build a base image with the ROCm software stack

    Build a docker image from [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) which sets up ROCm software stack needed by the vLLM.
    **This step is optional as this rocm_base image is usually prebuilt and stored at [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) under the tag `rocm/vllm-dev:base` to speed up the user experience.**
    If you choose to build this rocm_base image yourself, the steps are as follows.

    It is important that the user kicks off the docker build using buildkit. Either the user sets DOCKER_BUILDKIT=1 as environment variable when calling docker build command, or the user needs to set up buildkit in the docker daemon configuration /etc/docker/daemon.json as follows and restart the daemon:

    ```json
    {
        "features": {
            "buildkit": true
        }
    }
    ```

To build vllm on ROCm 7.0, you can use the default:

    ```bash
    DOCKER_BUILDKIT=1 docker build \
        -f docker/Dockerfile.rocm_base \
        -t rocm/vllm-dev:base .
    ```

#### Build an image with vLLM

Build a docker image from [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) and launch a docker container from the image.
It is important that the user kicks off the docker build using buildkit. Either the user put `DOCKER_BUILDKIT=1` as environment variable when calling docker build command, or the user needs to set up buildkit in the docker daemon configuration /etc/docker/daemon.json as follows and restart the daemon:

```bash
{
    "features": {
        "buildkit": true
    }
}
```

[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) uses ROCm 7.0 by default, but also supports ROCm 5.7, 6.0, 6.1, 6.2, 6.3, and 6.4, in older vLLM branches.
It provides flexibility to customize the build of docker image using the following arguments:

- `BASE_IMAGE`: specifies the base image used when running `docker build`. The default value `rocm/vllm-dev:base` is an image published and maintained by AMD. It is built using [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base)
- `ARG_PYTORCH_ROCM_ARCH`: Allows to override the gfx architecture values from the base docker image

Their values can be passed in when running `docker build` with `--build-arg` options.

To build vllm on ROCm 7.0, you can use the default:

???+ console "Commands"
    ```bash
    DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-rocm .
    ```

To run the above docker image `vllm-rocm`, use the below command:

???+ console "Commands"
    ```bash
    docker run -it \
    --network=host \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v <path/to/model>:/app/model \
    vllm-rocm
    ```

Where the `<path/to/model>` is the location where the model is stored, for example, the weights for llama2 or llama3 models.

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

See [Feature x Hardware](../../features/README.md#feature-x-hardware) compatibility matrix for feature support information.

# --8<-- [end:supported-features]
