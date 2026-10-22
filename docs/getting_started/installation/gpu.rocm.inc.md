<!-- markdownlint-disable MD041 MD051 -->
--8<-- [start:installation]

vLLM supports AMD GPUs with ROCm 6.3 or above. Pre-built wheels are available for ROCm 10.0 (the default) and ROCm 7.2.

#### Prebuilt Wheels

| ROCm Variant | Python Version | ROCm Version | glibc Requirement | Supported Versions |
| ------------ | -------------- | ------------ | ----------------- | ------------------ |
| `rocm700` | 3.12 | 7.0 | >= 2.35 | `0.14.0` to `0.18.0` |
| `rocm721` | 3.12 | 7.2.1 | >= 2.35 | Nightly releases after commit `171775f306a333a9cf105bfd533bf3e113d401d9` |
| `rocm72` | 3.12 | 7.2 | >= 2.39 | Nightly releases on the legacy ROCm 7.2 stack |
| `rocm100` | 3.12 | 10.0 | >= 2.35 | Nightly releases (default) |

--8<-- [end:installation]
--8<-- [start:requirements]

- GPU: MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 series (gfx1100/1101), Radeon RX 9000 series (gfx1200/1201), Ryzen AI MAX / AI 300 Series (gfx1151/1150)
- ROCm 6.3 or above
    - MI350 requires ROCm 7.0 or above
    - Ryzen AI MAX / AI 300 Series requires ROCm 7.0.2 or above

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

The vLLM wheel bundles PyTorch and all required dependencies, and you should use the included PyTorch for compatibility. Because vLLM compiles many ROCm kernels to ensure a validated, high‑performance stack, the resulting binaries may not be compatible with other ROCm or PyTorch builds.
If you need a different ROCm version or want to use an existing PyTorch installation, you’ll need to build vLLM from source.  See [below](#build-wheel-from-source) for more details.

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

!!! warning "Python 3.12 required for ROCm wheels"

    ROCm pre-built wheels are only available for **Python 3.12**. If you are using a different Python version (e.g. 3.11 or 3.13), the installer **will silently fall back** to the CUDA wheel from PyPI, which will fail on AMD GPUs with errors like `libcudart.so: cannot open shared object file`.

    To check your Python version: `python3 --version`

    If you need Python 3.12, you can create an isolated environment with `uv`:

    ```bash
    uv venv --python 3.12 --seed --managed-python
    source .venv/bin/activate
    ```

To install the latest version of vLLM for Python 3.12, ROCm 10.0 and `glibc >= 2.35`:

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade
```

!!! tip
    The ROCm 10.0 wheels install the ROCm SDK from pip ([TheRock](https://github.com/ROCm/TheRock)), including the GPU kernels for every supported architecture.

    Wheels for the legacy ROCm 7.2 stack, which use a system ROCm installation, remain available at `https://wheels.vllm.ai/rocm/${VLLM_VERSION}/rocm72`.

!!! tip
    You can find out about which ROCm version the latest vLLM supports by checking the `vllm` package in index in extra-index-url <https://wheels.vllm.ai/rocm/> at [https://wheels.vllm.ai/rocm/vllm](https://wheels.vllm.ai/rocm/vllm) .

    Another approach is that you can use this following commands to automatically extract the wheel variants:

    ```bash
    # automatically extract the available rocm variant
    export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/vllm | grep -oP 'rocm\d+' | head -1)

    # automatically extract the vLLM version
    export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/vllm | grep -oP 'vllm-\K[0-9.]+' | head -1)

    # inspect if the ROCm version is compatible with your environment
    echo $VLLM_ROCM_VARIANT
    echo $VLLM_VERSION
    ```

To install a specific version and ROCm variant of vLLM wheel.

```bash
# version without the `v`
uv pip install vllm==${VLLM_VERSION} --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}

# Example
uv pip install vllm==0.18.0 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700
```

!!! warning "Caveats for using `pip`"

    We recommend leveraging `uv` to install the vLLM wheel. Using `pip` to install from custom indices is cumbersome because `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version. This makes it difficult to install a wheel from a custom index unless exact versions of all packages are specified. In contrast, `uv` gives the extra index [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

    If you insist on using `pip`, you need to specify the exact vLLM version in the package name and provide the custom index URL `https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}` via `--extra-index-url`.

    ```bash
    pip install vllm==0.18.0+rocm700 --extra-index-url https://wheels.vllm.ai/rocm/0.18.0/rocm700
    ```

#### Install the latest code

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides wheels for every commit since commit `171775f306a333a9cf105bfd533bf3e113d401d9` on <https://wheels.vllm.ai/rocm/nightly/>. The custom index to be used is `https://wheels.vllm.ai/rocm/nightly/${VLLM_ROCM_VARIANT}`

**NOTE:** The first ROCm Variant that supports nightly wheel is ROCm 7.2.1

To install from latest nightly index, run:

```bash
# automatically extract the available rocm variant
export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/nightly | \
    grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

# inspect if the ROCm version is compatible with your environment
echo $VLLM_ROCM_VARIANT

uv pip install --pre vllm \
    --extra-index-url https://wheels.vllm.ai/rocm/nightly/${VLLM_ROCM_VARIANT} \
    --index-strategy unsafe-best-match
```

##### Install specific revisions

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), you can specify the commit hash in the URL, example:

```bash
export VLLM_COMMIT=5b8c30d62b754b575e043ce2fc0dcbf8a64f6306

export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT} | \
    grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

# Extract the version from the wheel URL
export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/ | \
    grep -oP 'vllm-\K[^-]+' | head -1  | sed 's/%2B/+/g')

# inspect the version if it is compatible with the ROCm version of your environment
echo $VLLM_ROCM_VARIANT
echo $VLLM_VERSION

uv pip install vllm==${VLLM_VERSION} \
  --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT} \
  --index-strategy unsafe-best-match
```

!!! warning "`pip` caveat"

    Using `pip` to install from nightly indices is _not supported_, because `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a development version prior to the released version. In contrast, `uv` gives the extra index [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

    If you insist on using `pip`, you need to specify the exact vLLM version in the package name and provide the custom index URL (which can be obtained from the web page).

    ```bash
    export VLLM_COMMIT=5b8c30d62b754b575e043ce2fc0dcbf8a64f6306

    export VLLM_ROCM_VARIANT=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT} | \
        grep -oP 'rocm\d+' | head -1  | sed 's/%2B/+/g')

    # Extract the version from the wheel URL
    export VLLM_VERSION=$(curl -s https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}/vllm/ | \
        grep -oP 'vllm-\K[^-]+' | head -1  | sed 's/%2B/+/g')

    # inspect the version if it is compatible with the ROCm version of your environment
    echo $VLLM_ROCM_VARIANT
    echo $VLLM_VERSION

    pip install vllm==${VLLM_VERSION} \
    --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_COMMIT}/${VLLM_ROCM_VARIANT}
    ```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

#### Set up using Python-only build (without compilation) {#python-only-build}

If you only need to change Python code, you can build and install vLLM without
compilation. Changes you make to the code will be reflected when you run vLLM:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 python3 setup.py develop
```

This command will do the following:

1. Look for the current branch in your vLLM clone.
1. Identify the corresponding base commit in the main branch.
1. Detect the ROCm version in your environment and select the matching wheel
   variant.
1. Download the pre-built wheel of the base commit.
1. Use its compiled libraries and `vllm-rs` binary in the installation.

!!! note
    1. If you change C++, HIP, or kernel code, you cannot use Python-only build;
       otherwise you may see an import error about a library not being found or
       an undefined symbol.
    2. If you rebase your development branch, it is recommended to uninstall
       vLLM and re-run the above command to make sure your libraries are up to
       date.

!!! tip "Rebuilding the Rust frontend"
If you need to recompile the `vllm-rs` Rust frontend binary, you can rebuild and
install it without re-running the full installation:

    ```bash
    ./tools/build_rust.sh          # release build
    ./tools/build_rust.sh --debug  # faster build for development
    ```

    This will install the required Rust toolchain if needed, build the binary,
    and place it in `vllm/vllm-rs`.

If you see an error about a wheel not being found, the wheel for your base
commit and ROCm version might not be available. Check the available
variants under `https://wheels.vllm.ai/rocm/<commit>/`. Variants are named by ROCm
major and minor version, e.g. ROCm 7.2.3 uses the `rocm72` variant.

There are more environment variables to control the behavior of Python-only
build:

- `VLLM_PRECOMPILED_WHEEL_LOCATION`: specify the exact wheel URL or local file
  path of a pre-compiled wheel to use. All other logic to find the wheel will be
  skipped.
- `VLLM_PRECOMPILED_WHEEL_COMMIT`: override the full commit hash used to
  download the pre-compiled wheel.
- `VLLM_PRECOMPILED_WHEEL_VARIANT`: specify the ROCm variant subdirectory, e.g.,
  `rocm72` or `rocm100`. If not specified, the variant is auto-detected based
  on your system's ROCm version. An explicitly specified variant must match the
  detected environment.

You can find more information about vLLM's wheels in
[Install the latest code](#install-the-latest-code).

!!! note
    There is a possibility that your source code may have a different commit ID
    compared to the vLLM wheel, which could potentially lead to unknown errors.
    It is recommended to use the same commit ID for the source code as the vLLM
    wheel you have installed. Please refer to
    [Install the latest code](#install-the-latest-code) for instructions on how
    to install a specified wheel.

#### Full build (with compilation) {#full-build}

!!! tip
    - The steps below follow [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base), which pins the validated version of every component. If a step does not work for you, refer to it; a Dockerfile is a form of installation steps.
    - The legacy ROCm 7.2 stack uses a system ROCm installation instead; its steps are in [docker/Dockerfile.rocm_72_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_72_base).

0. Install the ROCm SDK and PyTorch (skip if you are already in an environment or docker image with them installed, e.g. `vllm/vllm-openai-rocm:base-nightly`).

    ROCm 10.0 is installed from pip ([TheRock](https://github.com/ROCm/TheRock)), together with PyTorch and the GPU kernels for your GPU's architecture. In a Python 3.12 environment:

    ```bash
    # Your GPU's architecture, e.g. gfx942 for MI300 or gfx950 for MI350
    export GPU_ARCH=$(uvx --from rocm-bootstrap rocm-bootstrap-detect --unique)
    export ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/

    pip install --index-url ${ROCM_INDEX} \
        "rocm[libraries,devel,device-${GPU_ARCH}]==10.0.0" \
        "torch[device-${GPU_ARCH}]==2.12.0+rocm10.0.0" \
        "torchvision[device-${GPU_ARCH}]==0.27.0+rocm10.0.0"
    rocm-sdk init

    # Point builds at the pip-installed ROCm SDK
    export ROCM_PATH=$(rocm-sdk path --root)
    export ROCM_HOME=${ROCM_PATH} HIP_PATH=${ROCM_PATH}
    export PATH=$(rocm-sdk path --bin):${PATH}
    ```

    !!! note
        - The validated `ROCM_SDK_VERSION`, `TORCH_VERSION` and `TORCHVISION_VERSION` can be found in [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).
        - To support several GPU architectures, add a `device-<arch>` extra for each of them.

1. Install [Triton for ROCm](https://github.com/ROCm/triton.git)

    Install ROCm's Triton following the instructions from [ROCm/triton](https://github.com/ROCm/triton.git)

    ```bash
    python3 -m pip install ninja cmake wheel pybind11
    pip uninstall -y triton
    git clone https://github.com/ROCm/triton.git
    cd triton
    git checkout $TRITON_BRANCH
    if [ ! -f setup.py ]; then cd python; fi
    python3 setup.py install
    cd ../..
    ```

    !!! note
        - The validated `$TRITON_BRANCH` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).
        - If you see HTTP issue related to downloading packages during building triton, please try again as the HTTP error is intermittent.

2. Optionally, if you choose to use CK flash attention, you can install [flash attention for ROCm](https://github.com/Dao-AILab/flash-attention.git)

    Install ROCm's flash attention following the instructions from [ROCm/flash-attention](https://github.com/Dao-AILab/flash-attention#amd-rocm-support)

    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    git checkout $FA_BRANCH
    git submodule update --init
    GPU_ARCHS="${GPU_ARCH}" python3 setup.py install
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
    # Keep the Triton installed in step 1
    AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop
    ```

    !!! note
        - You will need to config the `$AITER_BRANCH_OR_COMMIT` for your purpose.
        - The validated `$AITER_BRANCH_OR_COMMIT` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).

4. Optionally, if you want to use MORI for EP or PD disaggregation, you can install [MORI](https://github.com/ROCm/mori) using the following steps:

    ```bash
    pip install "setuptools_scm>=6.2"
    git clone https://github.com/ROCm/mori.git
    cd mori
    git checkout $MORI_BRANCH_OR_COMMIT
    git submodule sync; git submodule update --init --recursive
    # The ROCm SDK keeps its CMake configs for system dependencies (e.g. NUMA) under rocm_sysdeps
    CMAKE_PREFIX_PATH="${ROCM_PATH}:${ROCM_PATH}/lib/rocm_sysdeps" \
        MORI_GPU_ARCHS="gfx942;gfx950" python3 setup.py install
    ```

    !!! note
        - You will need to config the `$MORI_BRANCH_OR_COMMIT` for your purpose.
        - The validated `$MORI_BRANCH_OR_COMMIT` can be found in the [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).

5. Build vLLM. For example, vLLM on ROCm 10.0 can be built with the following steps:

    ???+ console "Commands"

        ```bash
        pip install --upgrade pip

        # Install AMD SMI, which the ROCm SDK ships as sources
        SDK_CORE=$(python3 -c "import _rocm_sdk_core, os; print(os.path.dirname(_rocm_sdk_core.__file__))")
        pip install ${SDK_CORE}/share/amd_smi
        # It loads the SDK's libamd_smi relative to its own directory
        export PYTHONPATH=${SDK_CORE}/share/amd_smi

        # Install dependencies
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

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

vLLM offers official Docker images for deployment.
The images can be used to run OpenAI compatible server and are available on Docker Hub as [vllm/vllm-openai-rocm](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags).

- `vllm/vllm-openai-rocm:latest` — stable release, built on ROCm 10.0 (also tagged `latest-rocm100`)
- `vllm/vllm-openai-rocm:nightly` — preview build from the latest development branch on ROCm 10.0 (also tagged `nightly-rocm100`), use this if you want the latest features and fixes
- `vllm/vllm-openai-rocm:latest-rocm72` / `nightly-rocm72` — the same builds on the legacy ROCm 7.2 stack, kept for a transition period

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
    vllm/vllm-openai-rocm:<tag> \
    --model Qwen/Qwen3-0.6B
```

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
        --entrypoint /bin/bash \
        vllm/vllm-openai-rocm:<tag>
    ```

#### Use AMD's Docker Images (Deprecated)

!!! warning "Deprecated"
    AMD's Docker images (`rocm/vllm` and `rocm/vllm-dev`) are deprecated in favor of the official vLLM Docker images above (`vllm/vllm-openai-rocm`). Please migrate to the official images.

Prior to January 20th, 2026 when the official docker images became available on [upstream vLLM docker hub](https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/), the [AMD Infinity hub for vLLM](https://hub.docker.com/r/rocm/vllm/tags) offered a prebuilt, optimized
docker image designed for validating inference performance on the AMD Instinct MI300X™ accelerator.
AMD also offered nightly prebuilt docker image from [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev), which has vLLM and all its dependencies installed. The entrypoint of this docker image is `/bin/bash` (different from the vLLM's Official Docker Image).

!!! tip
    Please check [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)
    for instructions on how to use this prebuilt docker image.

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

You can build and run vLLM from source via the provided [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm).

??? info "(Optional) Build an image with ROCm software stack"

    Build a docker image from [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) which setup ROCm software stack needed by the vLLM.
    **This step is optional as this rocm_base image is usually prebuilt and store at [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) under tag `rocm/vllm-dev:base` to speed up user experience.** For the legacy ROCm 7.2 stack, the base image is built from [docker/Dockerfile.rocm_72_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_72_base) and published as `vllm/vllm-openai-rocm:base-nightly-rocm72`.
    If you choose to build this rocm_base image yourself, the steps are as follows.

    It is important that the user kicks off the docker build using buildkit. Either the user put `DOCKER_BUILDKIT=1` as environment variable when calling docker build command, or the user needs to set up buildkit in the docker daemon configuration `/etc/docker/daemon.json` as follows and restart the daemon:

    ```json
    {
        "features": {
            "buildkit": true
        }
    }
    ```

    To build vllm on ROCm 10.0 for all supported GPUs, you can use the default:

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

[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) uses ROCm 10.0 (installed from TheRock wheels) by default. The legacy ROCm 7.2 stack is available as [docker/Dockerfile.rocm_72](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_72) with [docker/Dockerfile.rocm_72_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_72_base); older vLLM branches support ROCm 5.7 through 7.0.
It provides flexibility to customize the build of docker image using the following arguments:

- `BASE_IMAGE`: specifies the base image used when running `docker build`. The default value `rocm/vllm-dev:base` is an image published and maintained by AMD. It is being built using [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base). For [docker/Dockerfile.rocm_72](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_72), the default is `vllm/vllm-openai-rocm:base-nightly-rocm72`
- `ARG_PYTORCH_ROCM_ARCH`: Allows to override the gfx architecture values from the base docker image

Their values can be passed in when running `docker build` with `--build-arg` options.

To build vllm on ROCm 10.0 for all supported GPUs, you can use the default (which build a docker image with `vllm serve` as entrypoint):

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm/vllm-openai-rocm .
```

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

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

See [Feature x Hardware](../../features/README.md#feature-x-hardware) compatibility matrix for feature support information.

--8<-- [end:supported-features]
