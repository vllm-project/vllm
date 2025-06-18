# --8<-- [start:installation]

vLLM supports AMD GPUs with ROCm 6.3.

!!! warning
    There are no pre-built wheels for this device, so you must either use the pre-built Docker image or build vLLM from source.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU: MI200s (gfx90a), MI300 (gfx942), Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201)
- ROCm 6.3

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Currently, there are no pre-built ROCm wheels.

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

0. Install prerequisites (skip if you are already in an environment/docker with the following installed):

    - [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
    - [PyTorch](https://pytorch.org/)

    For installing PyTorch, you can start from a fresh docker image, e.g, `rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0`, `rocm/pytorch-nightly`. If you are using docker image, you can skip to Step 3.

    Alternatively, you can install PyTorch using PyTorch wheels. You can check PyTorch installation guide in PyTorch [Getting Started](https://pytorch.org/get-started/locally/). Example:

    ```console
    # Install PyTorch
    $ pip uninstall torch -y
    $ pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3
    ```

1. Install [Triton flash attention for ROCm](https://github.com/ROCm/triton)

    Install ROCm's Triton flash attention (the default triton-mlir branch) following the instructions from [ROCm/triton](https://github.com/ROCm/triton/blob/triton-mlir/README.md)

    ```console
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/OpenAI/triton.git
    cd triton
    git checkout e5be006
    cd python
    pip3 install .
    cd ../..
    ```

    !!! note
        If you see HTTP issue related to downloading packages during building triton, please try again as the HTTP error is intermittent.

2. Optionally, if you choose to use CK flash attention, you can install [flash attention for ROCm](https://github.com/ROCm/flash-attention)

    Install ROCm's flash attention (v2.7.2) following the instructions from [ROCm/flash-attention](https://github.com/ROCm/flash-attention#amd-rocm-support)
    Alternatively, wheels intended for vLLM use can be accessed under the releases.

    For example, for ROCm 6.3, suppose your gfx arch is `gfx90a`. To get your gfx architecture, run `rocminfo |grep gfx`.

    ```console
    git clone https://github.com/ROCm/flash-attention.git
    cd flash-attention
    git checkout b7d29fb
    git submodule update --init
    GPU_ARCHS="gfx90a" python3 setup.py install
    cd ..
    ```

    !!! note
        You might need to downgrade the "ninja" version to 1.10 as it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)

3. If you choose to build AITER yourself to use a certain branch or commit, you can build AITER using the following steps:

    ```console
    python3 -m pip uninstall -y aiter
    git clone --recursive https://github.com/ROCm/aiter.git
    cd aiter
    git checkout $AITER_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    python3 setup.py develop
    ```

    !!! note
        You will need to config the `$AITER_BRANCH_OR_COMMIT` for your purpose.

4. Build vLLM. For example, vLLM on ROCM 6.3 can be built with the following steps:

    ```bash
    pip install --upgrade pip

    # Build & install AMD SMI
    pip install /opt/rocm/share/amd_smi

    # Install dependencies
    pip install --upgrade numba \
        scipy \
        huggingface-hub[cli,hf_transfer] \
        setuptools_scm
    pip install "numpy<2"
    pip install -r requirements/rocm.txt

    # Build vLLM for MI210/MI250/MI300.
    export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
    python3 setup.py develop
    ```

    This may take 5-10 minutes. Currently, `pip install .` does not work for ROCm installation.

    !!! tip
        - Triton flash attention is used by default. For benchmarking purposes, it is recommended to run a warm up step before collecting perf numbers.
        - Triton flash attention does not currently support sliding window attention. If using half precision, please use CK flash-attention for sliding window support.
        - To use CK flash-attention or PyTorch naive attention, please use this flag `export VLLM_USE_TRITON_FLASH_ATTN=0` to turn off triton flash attention.
        - The ROCm version of PyTorch, ideally, should match the ROCm driver version.

!!! tip
    - For MI300x (gfx942) users, to achieve optimal performance, please refer to [MI300x tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html) for performance optimization and tuning tips on system and workflow level.
      For vLLM, please refer to [vLLM performance optimization](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html#vllm-performance-optimization).

## Set up using Docker (Recommended)

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

The [AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) offers a prebuilt, optimized
docker image designed for validating inference performance on the AMD Instinct™ MI300X accelerator.

!!! tip
    Please check [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)
    for instructions on how to use this prebuilt docker image.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

Building the Docker image from source is the recommended way to use vLLM with ROCm.

#### (Optional) Build an image with ROCm software stack

Build a docker image from <gh-file:docker/Dockerfile.rocm_base> which setup ROCm software stack needed by the vLLM.
**This step is optional as this rocm_base image is usually prebuilt and store at [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) under tag `rocm/vllm-dev:base` to speed up user experience.**
If you choose to build this rocm_base image yourself, the steps are as follows.

It is important that the user kicks off the docker build using buildkit. Either the user put DOCKER_BUILDKIT=1 as environment variable when calling docker build command, or the user needs to setup buildkit in the docker daemon configuration /etc/docker/daemon.json as follows and restart the daemon:

```console
{
    "features": {
        "buildkit": true
    }
}
```

To build vllm on ROCm 6.3 for MI200 and MI300 series, you can use the default:

```console
DOCKER_BUILDKIT=1 docker build \
    -f docker/Dockerfile.rocm_base \
    -t rocm/vllm-dev:base .
```

#### Build an image with vLLM

First, build a docker image from <gh-file:docker/Dockerfile.rocm> and launch a docker container from the image.
It is important that the user kicks off the docker build using buildkit. Either the user put `DOCKER_BUILDKIT=1` as environment variable when calling docker build command, or the user needs to setup buildkit in the docker daemon configuration /etc/docker/daemon.json as follows and restart the daemon:

```console
{
    "features": {
        "buildkit": true
    }
}
```

<gh-file:docker/Dockerfile.rocm> uses ROCm 6.3 by default, but also supports ROCm 5.7, 6.0, 6.1, and 6.2, in older vLLM branches.
It provides flexibility to customize the build of docker image using the following arguments:

- `BASE_IMAGE`: specifies the base image used when running `docker build`. The default value `rocm/vllm-dev:base` is an image published and maintained by AMD. It is being built using <gh-file:docker/Dockerfile.rocm_base>
- `ARG_PYTORCH_ROCM_ARCH`: Allows to override the gfx architecture values from the base docker image

Their values can be passed in when running `docker build` with `--build-arg` options.

To build vllm on ROCm 6.3 for MI200 and MI300 series, you can use the default:

```console
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-rocm .
```

To build vllm on ROCm 6.3 for Radeon RX7900 series (gfx1100), you should pick the alternative base image:

```console
DOCKER_BUILDKIT=1 docker build \
    --build-arg BASE_IMAGE="rocm/vllm-dev:navi_base" \
    -f docker/Dockerfile.rocm \
    -t vllm-rocm \
    .
```

To run the above docker image `vllm-rocm`, use the below command:

```console
docker run -it \
   --network=host \
   --group-add=video \
   --ipc=host \
   --cap-add=SYS_PTRACE \
   --security-opt seccomp=unconfined \
   --device /dev/kfd \
   --device /dev/dri \
   -v <path/to/model>:/app/model \
   vllm-rocm \
   bash
```

Where the `<path/to/model>` is the location where the model is stored, for example, the weights for llama2 or llama3 models.

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

See [feature-x-hardware][feature-x-hardware] compatibility matrix for feature support information.

# --8<-- [end:supported-features]
# --8<-- [end:extra-information]
