# Installation

vLLM supports AMD GPUs with ROCm 6.2.

## Requirements

- GPU: MI200s (gfx90a), MI300 (gfx942), Radeon RX 7900 series (gfx1100)
- ROCm 6.2

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built ROCm wheels.

### Build wheel from source

0. Install prerequisites (skip if you are already in an environment/docker with the following installed):

- [ROCm](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
- [PyTorch](https://pytorch.org/)

    For installing PyTorch, you can start from a fresh docker image, e.g, `rocm/pytorch:rocm6.2_ubuntu20.04_py3.9_pytorch_release_2.3.0`, `rocm/pytorch-nightly`.

    Alternatively, you can install PyTorch using PyTorch wheels. You can check PyTorch installation guide in PyTorch [Getting Started](https://pytorch.org/get-started/locally/)

1. Install [Triton flash attention for ROCm](https://github.com/ROCm/triton)

    Install ROCm's Triton flash attention (the default triton-mlir branch) following the instructions from [ROCm/triton](https://github.com/ROCm/triton/blob/triton-mlir/README.md)

    ```console
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/OpenAI/triton.git
    cd triton
    git checkout e192dba
    cd python
    pip3 install .
    cd ../..
    ```

    ```{note}
    - If you see HTTP issue related to downloading packages during building triton, please try again as the HTTP error is intermittent.
    ```

2. Optionally, if you choose to use CK flash attention, you can install [flash attention for ROCm](https://github.com/ROCm/flash-attention/tree/ck_tile)

    Install ROCm's flash attention (v2.5.9.post1) following the instructions from [ROCm/flash-attention](https://github.com/ROCm/flash-attention/tree/ck_tile#amd-gpurocm-support)
    Alternatively, wheels intended for vLLM use can be accessed under the releases.

    For example, for ROCm 6.2, suppose your gfx arch is `gfx90a`. To get your gfx architecture, run `rocminfo |grep gfx`.

    ```console
    git clone https://github.com/ROCm/flash-attention.git
    cd flash-attention
    git checkout 3cea2fb
    git submodule update --init
    GPU_ARCHS="gfx90a" python3 setup.py install
    cd ..
    ```

    ```{note}
    - You might need to downgrade the "ninja" version to 1.10 it is not used when compiling flash-attention-2 (e.g. `pip install ninja==1.10.2.4`)
    ```

3. Build vLLM. For example, vLLM on ROCM 6.2 can be built with the following steps:

    ```bash
    $ pip install --upgrade pip

    # Install PyTorch
    $ pip uninstall torch -y
    $ pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/rocm6.2

    # Build & install AMD SMI
    $ pip install /opt/rocm/share/amd_smi

    # Install dependencies
    $ pip install --upgrade numba scipy huggingface-hub[cli]
    $ pip install "numpy<2"
    $ pip install -r requirements-rocm.txt

    # Build vLLM for MI210/MI250/MI300.
    $ export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
    $ python3 setup.py develop
    ```

    This may take 5-10 minutes. Currently, `pip install .` does not work for ROCm installation.

    ```{tip}
    - Triton flash attention is used by default. For benchmarking purposes, it is recommended to run a warm up step before collecting perf numbers.
    - Triton flash attention does not currently support sliding window attention. If using half precision, please use CK flash-attention for sliding window support.
    - To use CK flash-attention or PyTorch naive attention, please use this flag `export VLLM_USE_TRITON_FLASH_ATTN=0` to turn off triton flash attention.
    - The ROCm version of PyTorch, ideally, should match the ROCm driver version.
    ```

```{tip}
- For MI300x (gfx942) users, to achieve optimal performance, please refer to [MI300x tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html) for performance optimization and tuning tips on system and workflow level.
  For vLLM, please refer to [vLLM performance optimization](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html#vllm-performance-optimization).
```

## Set up using Docker

### Pre-built images

Currently, there are no pre-built ROCm images.

### Build image from source

Building the Docker image from source is the recommended way to use vLLM with ROCm.

First, build a docker image from <gh-file:Dockerfile.rocm> and launch a docker container from the image.
It is important that the user kicks off the docker build using buildkit. Either the user put DOCKER_BUILDKIT=1 as environment variable when calling docker build command, or the user needs to setup buildkit in the docker daemon configuration /etc/docker/daemon.json as follows and restart the daemon:

```console
{
    "features": {
        "buildkit": true
    }
}
```

<gh-file:Dockerfile.rocm> uses ROCm 6.2 by default, but also supports ROCm 5.7, 6.0 and 6.1 in older vLLM branches.
It provides flexibility to customize the build of docker image using the following arguments:

- `BASE_IMAGE`: specifies the base image used when running `docker build`, specifically the PyTorch on ROCm base image.
- `BUILD_FA`: specifies whether to build CK flash-attention. The default is 1. For [Radeon RX 7900 series (gfx1100)](https://rocm.docs.amd.com/projects/radeon/en/latest/index.html), this should be set to 0 before flash-attention supports this target.
- `FX_GFX_ARCHS`: specifies the GFX architecture that is used to build CK flash-attention, for example, `gfx90a;gfx942` for MI200 and MI300. The default is `gfx90a;gfx942`
- `FA_BRANCH`: specifies the branch used to build the CK flash-attention in [ROCm's flash-attention repo](https://github.com/ROCmSoftwarePlatform/flash-attention). The default is `ae7928c`
- `BUILD_TRITON`: specifies whether to build triton flash-attention. The default value is 1.

Their values can be passed in when running `docker build` with `--build-arg` options.

To build vllm on ROCm 6.2 for MI200 and MI300 series, you can use the default:

```console
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm -t vllm-rocm .
```

To build vllm on ROCm 6.2 for Radeon RX7900 series (gfx1100), you should specify `BUILD_FA` as below:

```console
DOCKER_BUILDKIT=1 docker build --build-arg BUILD_FA="0" -f Dockerfile.rocm -t vllm-rocm .
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

## Supported features

See <project:#feature-x-hardware> compatibility matrix for feature support information.
