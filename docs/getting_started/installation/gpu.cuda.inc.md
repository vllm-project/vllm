# --8<-- [start:installation]

vLLM contains pre-compiled C++ and CUDA (12.8) binaries.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

!!! note
    PyTorch installed via `conda` will statically link `NCCL` library, which can cause issues when vLLM tries to use `NCCL`. See <https://github.com/vllm-project/vllm/issues/8420> for more details.

In order to be performant, vLLM has to compile many cuda kernels. The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

Therefore, it is recommended to install vLLM with a **fresh new** environment. If either you have a different CUDA version or you want to use an existing PyTorch installation, you need to build vLLM from source. See [below](#build-wheel-from-source) for more details.

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

```bash
uv pip install vllm --torch-backend=auto
```

??? console "pip"
    ```bash
    # Install vLLM with CUDA 12.9.
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
    ```

We recommend leveraging `uv` to [automatically select the appropriate PyTorch index at runtime](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection) by inspecting the installed CUDA driver version via `--torch-backend=auto` (or `UV_TORCH_BACKEND=auto`). To select a specific backend (e.g., `cu128`), set `--torch-backend=cu128` (or `UV_TORCH_BACKEND=cu128`). If this doesn't work, try running `uv self update` to update `uv` first.

!!! note
    NVIDIA Blackwell GPUs (B200, GB200) require a minimum of CUDA 12.8, so make sure you are installing PyTorch wheels with at least that version. PyTorch itself offers a [dedicated interface](https://pytorch.org/get-started/locally/) to determine the appropriate pip command to run for a given target configuration.

As of now, vLLM's binaries are compiled with CUDA 12.9 and public PyTorch release versions by default. We also provide vLLM binaries compiled with CUDA 12.8, 13.0, and public PyTorch release versions:

```bash
# Install vLLM with a specific CUDA version (e.g., 13.0).
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130 # or other
export CPU_ARCH=$(uname -m) # x86_64 or aarch64
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

#### Install the latest code

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides wheels for every commit since `v0.5.3` on <https://wheels.vllm.ai/nightly>. There are multiple indices that could be used:

* `https://wheels.vllm.ai/nightly`: the default variant (CUDA with version specified in `VLLM_MAIN_CUDA_VERSION`) built with the last commit on the `main` branch. Currently it is CUDA 12.9.
* `https://wheels.vllm.ai/nightly/<variant>`: all other variants. Now this includes `cu130`, and `cpu`. The default variant (`cu129`) also has a subdirectory to keep consistency.

To install from nightly index, run:

```bash
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly # add variant subdirectory here if needed
```

!!! warning "`pip` caveat"

    Using `pip` to install from nightly indices is _not supported_, because `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a development version prior to the released version. In contrast, `uv` gives the extra index [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

    If you insist on using `pip`, you have to specify the full URL of the wheel file (which can be obtained from the web page).

    ```bash
    pip install -U https://wheels.vllm.ai/nightly/vllm-0.11.2.dev399%2Bg3c7461c18-cp38-abi3-manylinux_2_31_x86_64.whl # current nightly build (the filename will change!)
    pip install -U https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-0.11.2.dev399%2Bg3c7461c18-cp38-abi3-manylinux_2_31_x86_64.whl # from specific commit
    ```

##### Install specific revisions

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), you can specify the commit hash in the URL:

```bash
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # use full commit hash from the main branch
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} # add variant subdirectory here if needed
```

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

#### Set up using Python-only build (without compilation) {#python-only-build}

If you only need to change Python code, you can build and install vLLM without compilation. Using `uv pip`'s [`--editable` flag](https://docs.astral.sh/uv/pip/packages/#editable-packages), changes you make to the code will be reflected when you run vLLM:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

This command will do the following:

1. Look for the current branch in your vLLM clone.
1. Identify the corresponding base commit in the main branch.
1. Download the pre-built wheel of the base commit.
1. Use its compiled libraries in the installation.

!!! note
    1. If you change C++ or kernel code, you cannot use Python-only build; otherwise you will see an import error about library not found or undefined symbol.
    2. If you rebase your dev branch, it is recommended to uninstall vllm and re-run the above command to make sure your libraries are up to date.

In case you see an error about wheel not found when running the above command, it might be because the commit you based on in the main branch was just merged and the wheel is being built. In this case, you can wait for around an hour to try again, or manually assign the previous commit in the installation using the `VLLM_PRECOMPILED_WHEEL_LOCATION` environment variable.

```bash
export VLLM_PRECOMPILED_WHEEL_COMMIT=$(git rev-parse HEAD~1) # or earlier commit on main
export VLLM_USE_PRECOMPILED=1
uv pip install --editable .
```

There are more environment variables to control the behavior of Python-only build:

* `VLLM_PRECOMPILED_WHEEL_LOCATION`: specify the exact wheel URL or local file path of a pre-compiled wheel to use. All other logic to find the wheel will be skipped.
* `VLLM_PRECOMPILED_WHEEL_COMMIT`: override the commit hash to download the pre-compiled wheel. It can be `nightly` to use the last **already built** commit on the main branch.
* `VLLM_PRECOMPILED_WHEEL_VARIANT`: specify the variant subdirectory to use on the nightly index, e.g., `cu129`, `cu130`, `cpu`. If not specified, the variant is auto-detected based on your system's CUDA version (from PyTorch or nvidia-smi). You can also set `VLLM_MAIN_CUDA_VERSION` to override auto-detection.

You can find more information about vLLM's wheels in [Install the latest code](#install-the-latest-code).

!!! note
    There is a possibility that your source code may have a different commit ID compared to the latest vLLM wheel, which could potentially lead to unknown errors.
    It is recommended to use the same commit ID for the source code as the vLLM wheel you have installed. Please refer to [Install the latest code](#install-the-latest-code) for instructions on how to install a specified wheel.

#### Full build (with compilation) {#full-build}

If you want to modify C++ or CUDA code, you'll need to build vLLM from source. This can take several minutes:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -e .
```

!!! tip
    Building from source requires a lot of compilation. If you are building from source repeatedly, it's more efficient to cache the compilation results.

    For example, you can install [ccache](https://github.com/ccache/ccache) using `conda install ccache` or `apt install ccache` .
    As long as `which ccache` command can find the `ccache` binary, it will be used automatically by the build system. After the first build, subsequent builds will be much faster.

    When using `ccache` with `pip install -e .`, you should run `CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .`. This is because `pip` creates a new folder with a random name for each build, preventing `ccache` from recognizing that the same files are being built.

    [sccache](https://github.com/mozilla/sccache) works similarly to `ccache`, but has the capability to utilize caching in remote storage environments.
    The following environment variables can be set to configure the vLLM `sccache` remote: `SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`. We also recommend setting `SCCACHE_IDLE_TIMEOUT=0`.

!!! note "Faster Kernel Development"
    For frequent C++/CUDA kernel changes, after the initial `uv pip install -e .` setup, consider using the [Incremental Compilation Workflow](../../contributing/incremental_build.md) for significantly faster rebuilds of only the modified kernel code.

##### Use an existing PyTorch installation

There are scenarios where the PyTorch dependency cannot be easily installed with `uv`, for example, when building vLLM with non-default PyTorch builds (like nightly or a custom build).

To build vLLM using an existing PyTorch installation:

```bash
# install PyTorch first, either from PyPI or from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
```

Alternatively: if you are exclusively using `uv` to create and manage virtual environments, it has [a unique mechanism](https://docs.astral.sh/uv/concepts/projects/config/#disabling-build-isolation)
for disabling build isolation for specific packages. vLLM can leverage this mechanism to specify `torch` as the package to disable build isolation for:

```bash
# install PyTorch first, either from PyPI or from source
git clone https://github.com/vllm-project/vllm.git
cd vllm
# pip install -e . does not work directly, only uv can do this
uv pip install -e .
```

##### Use the local cutlass for compilation

Currently, before starting the build process, vLLM fetches cutlass code from GitHub. However, there may be scenarios where you want to use a local version of cutlass instead.
To achieve this, you can set the environment variable VLLM_CUTLASS_SRC_DIR to point to your local cutlass directory.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass uv pip install -e .
```

##### Troubleshooting

To avoid your system being overloaded, you can limit the number of compilation jobs
to be run simultaneously, via the environment variable `MAX_JOBS`. For example:

```bash
export MAX_JOBS=6
uv pip install -e .
```

This is especially useful when you are building on less powerful machines. For example, when you use WSL it only [assigns 50% of the total memory by default](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings), so using `export MAX_JOBS=1` can avoid compiling multiple files simultaneously and running out of memory.
A side effect is a much slower build process.

Additionally, if you have trouble building vLLM, we recommend using the NVIDIA PyTorch Docker image.

```bash
# Use `--ipc=host` to make sure the shared memory is large enough.
docker run \
    --gpus all \
    -it \
    --rm \
    --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```

If you don't want to use docker, it is recommended to have a full installation of CUDA Toolkit. You can download and install it from [the official website](https://developer.nvidia.com/cuda-toolkit-archive). After installation, set the environment variable `CUDA_HOME` to the installation path of CUDA Toolkit, and make sure that the `nvcc` compiler is in your `PATH`, e.g.:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

Here is a sanity check to verify that the CUDA Toolkit is correctly installed:

```bash
nvcc --version # verify that nvcc is in your PATH
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME
```

#### Unsupported OS build

vLLM can fully run only on Linux but for development purposes, you can still build it on other systems (for example, macOS), allowing for imports and a more convenient development environment. The binaries will not be compiled and won't work on non-Linux systems.

Simply disable the `VLLM_TARGET_DEVICE` environment variable before installing:

```bash
export VLLM_TARGET_DEVICE=empty
uv pip install -e .
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

vLLM offers an official Docker image for deployment.
The image can be used to run OpenAI compatible server and is available on Docker Hub as [vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags).

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B
```

This image can also be used with other container engines such as [Podman](https://podman.io/).

```bash
podman run --device nvidia.com/gpu=all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
--env "HF_TOKEN=$HF_TOKEN" \
-p 8000:8000 \
--ipc=host \
docker.io/vllm/vllm-openai:latest \
--model Qwen/Qwen3-0.6B
```

You can add any other [engine-args](https://docs.vllm.ai/en/latest/configuration/engine_args/) you need after the image tag (`vllm/vllm-openai:latest`).

!!! note
    You can either use the `ipc=host` flag or `--shm-size` flag to allow the
    container to access the host's shared memory. vLLM uses PyTorch, which uses shared
    memory to share data between processes under the hood, particularly for tensor parallel inference.

!!! note
    Optional dependencies are not included in order to avoid licensing issues (e.g. <https://github.com/vllm-project/vllm/issues/8030>).

    If you need to use those dependencies (having accepted the license terms),
    create a custom Dockerfile on top of the base image with an extra layer that installs them:

    ```Dockerfile
    FROM vllm/vllm-openai:v0.11.0

    # e.g. install the `audio` optional dependencies
    # NOTE: Make sure the version of vLLM matches the base image!
    RUN uv pip install --system vllm[audio]==0.11.0
    ```

!!! tip
    Some new models may only be available on the main branch of [HF Transformers](https://github.com/huggingface/transformers).

    To use the development version of `transformers`, create a custom Dockerfile on top of the base image
    with an extra layer that installs their code from source:

    ```Dockerfile
    FROM vllm/vllm-openai:latest

    RUN uv pip install --system git+https://github.com/huggingface/transformers.git
    ```

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

You can build and run vLLM from source via the provided [docker/Dockerfile](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile). To build vLLM:

```bash
# optionally specifies: --build-arg max_jobs=8 --build-arg nvcc_threads=2
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --file docker/Dockerfile
```

!!! note
    By default vLLM will build for all GPU types for widest distribution. If you are just building for the
    current GPU type the machine is running on, you can add the argument `--build-arg torch_cuda_arch_list=""`
    for vLLM to find the current GPU type and build for that.

    If you are using Podman instead of Docker, you might need to disable SELinux labeling by
    adding `--security-opt label=disable` when running `podman build` command to avoid certain [existing issues](https://github.com/containers/buildah/discussions/4184).

!!! note
    If you have not changed any C++ or CUDA kernel code, you can use precompiled wheels to significantly reduce Docker build time.

    *   **Enable the feature** by adding the build argument: `--build-arg VLLM_USE_PRECOMPILED="1"`.
    *   **How it works**: By default, vLLM automatically finds the correct wheels from our [Nightly Builds](https://docs.vllm.ai/en/latest/contributing/ci/nightly_builds/) by using the merge-base commit with the upstream `main` branch.
    *   **Override commit**: To use wheels from a specific commit, provide the `--build-arg VLLM_PRECOMPILED_WHEEL_COMMIT=<commit_hash>` argument.

    For a detailed explanation, refer to the documentation on 'Set up using Python-only build (without compilation)' part in [Build wheel from source](https://docs.vllm.ai/en/latest/contributing/ci/nightly_builds/#precompiled-wheels-usage), these args are similar.

#### Building vLLM's Docker Image from Source for Arm64/aarch64

A docker container can be built for aarch64 systems such as the Nvidia Grace-Hopper and Grace-Blackwell. Using the flag `--platform "linux/arm64"` will build for arm64.

!!! note
    Multiple modules must be compiled, so this process can take a while. Recommend using `--build-arg max_jobs=` & `--build-arg nvcc_threads=`
    flags to speed up build process. However, ensure your `max_jobs` is substantially larger than `nvcc_threads` to get the most benefits.
    Keep an eye on memory usage with parallel jobs as it can be substantial (see example below).

??? console "Command"

    ```bash
    # Example of building on Nvidia GH200 server. (Memory usage: ~15GB, Build time: ~1475s / ~25 min, Image size: 6.93GB)
    DOCKER_BUILDKIT=1 docker build . \
    --file docker/Dockerfile \
    --target vllm-openai \
    --platform "linux/arm64" \
    -t vllm/vllm-gh200-openai:latest \
    --build-arg max_jobs=66 \
    --build-arg nvcc_threads=2 \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
    --build-arg RUN_WHEEL_CHECK=false
    ```

For (G)B300, we recommend using CUDA 13, as shown in the following command.

??? console "Command"

    ```bash
    DOCKER_BUILDKIT=1 docker build \
    --build-arg CUDA_VERSION=13.0.1 \
    --build-arg BUILD_BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu22.04 \
    --build-arg max_jobs=256 \
    --build-arg nvcc_threads=2 \
    --build-arg RUN_WHEEL_CHECK=false \
    --build-arg torch_cuda_arch_list='9.0 10.0+PTX' \
    --platform "linux/arm64" \
    --tag vllm/vllm-gb300-openai:latest \
    --target vllm-openai \
    -f docker/Dockerfile \
    .
    ```

!!! note
    If you are building the `linux/arm64` image on a non-ARM host (e.g., an x86_64 machine), you need to ensure your system is set up for cross-compilation using QEMU. This allows your host machine to emulate ARM64 execution.

    Run the following command on your host machine to register QEMU user static handlers:

    ```bash
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    ```

    After setting up QEMU, you can use the `--platform "linux/arm64"` flag in your `docker build` command.

#### Use the custom-built vLLM Docker image**

To run vLLM with the custom-built Docker image:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai <args...>
```

The argument `vllm/vllm-openai` specifies the image to run, and should be replaced with the name of the custom-built image (the `-t` tag from the build command).

!!! note
    **For version 0.4.1 and 0.4.2 only** - the vLLM docker images under these versions are supposed to be run under the root user since a library under the root user's home directory, i.e. `/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` is required to be loaded during runtime. If you are running the container under a different user, you may need to first change the permissions of the library (and all the parent directories) to allow the user to access it, then run vLLM with environment variable `VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1` .

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

See [Feature x Hardware](../../features/README.md#feature-x-hardware) compatibility matrix for feature support information.

# --8<-- [end:supported-features]