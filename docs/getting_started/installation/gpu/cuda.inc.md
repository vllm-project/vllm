# --8<-- [start:installation]

vLLM contains pre-compiled C++ and CUDA (12.8) binaries.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

!!! note
    PyTorch installed via `conda` will statically link `NCCL` library, which can cause issues when vLLM tries to use `NCCL`. See <gh-issue:8420> for more details.

In order to be performant, vLLM has to compile many cuda kernels. The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

Therefore, it is recommended to install vLLM with a **fresh new** environment. If either you have a different CUDA version or you want to use an existing PyTorch installation, you need to build vLLM from source. See [below][build-from-source] for more details.

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

You can install vLLM using either `pip` or `uv pip`:

```console
# Install vLLM with CUDA 12.8.
# If you are using pip.
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
# If you are using uv.
uv pip install vllm --torch-backend=auto
```

We recommend leveraging `uv` to [automatically select the appropriate PyTorch index at runtime](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection) by inspecting the installed CUDA driver version via `--torch-backend=auto` (or `UV_TORCH_BACKEND=auto`). To select a specific backend (e.g., `cu126`), set `--torch-backend=cu126` (or `UV_TORCH_BACKEND=cu126`). If this doesn't work, try running `uv self update` to update `uv` first.

!!! note
    NVIDIA Blackwell GPUs (B200, GB200) require a minimum of CUDA 12.8, so make sure you are installing PyTorch wheels with at least that version. PyTorch itself offers a [dedicated interface](https://pytorch.org/get-started/locally/) to determine the appropriate pip command to run for a given target configuration.

As of now, vLLM's binaries are compiled with CUDA 12.8 and public PyTorch release versions by default. We also provide vLLM binaries compiled with CUDA 12.6, 11.8, and public PyTorch release versions:

```console
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.6.1.post1
export PYTHON_VERSION=312
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
```

[](){ #install-the-latest-code }

#### Install the latest code

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides wheels for Linux running on a x86 platform with CUDA 12 for every commit since `v0.5.3`.

##### Install the latest code using `pip`

```console
pip install -U vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly
```

`--pre` is required for `pip` to consider pre-released versions.

Another way to install the latest code is to use `uv`:

```console
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

##### Install specific revisions using `pip`

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), due to the limitation of `pip`, you have to specify the full URL of the wheel file by embedding the commit hash in the URL:

```console
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
pip install https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

Note that the wheels are built with Python 3.8 ABI (see [PEP 425](https://peps.python.org/pep-0425/) for more details about ABI), so **they are compatible with Python 3.8 and later**. The version string in the wheel file name (`1.0.0.dev`) is just a placeholder to have a unified URL for the wheels, the actual versions of wheels are contained in the wheel metadata (the wheels listed in the extra index url have correct versions). Although we don't support Python 3.8 any more (because PyTorch 2.5 dropped support for Python 3.8), the wheels are still built with Python 3.8 ABI to keep the same wheel name as before.

##### Install specific revisions using `uv`

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), you can specify the commit hash in the URL:

```console
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # use full commit hash from the main branch
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}
```

The `uv` approach works for vLLM `v0.6.6` and later and offers an easy-to-remember command. A unique feature of `uv` is that packages in `--extra-index-url` have [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes). If the latest public release is `v0.6.6.post1`, `uv`'s behavior allows installing a commit before `v0.6.6.post1` by specifying the `--extra-index-url`. In contrast, `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a development version prior to the released version.

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

#### Set up using Python-only build (without compilation)

If you only need to change Python code, you can build and install vLLM without compilation. Using `pip`'s [`--editable` flag](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs), changes you make to the code will be reflected when you run vLLM:

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
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

```console
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069 # use full commit hash from the main branch
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
```

You can find more information about vLLM's wheels in [install-the-latest-code][install-the-latest-code].

!!! note
    There is a possibility that your source code may have a different commit ID compared to the latest vLLM wheel, which could potentially lead to unknown errors.
    It is recommended to use the same commit ID for the source code as the vLLM wheel you have installed. Please refer to [install-the-latest-code][install-the-latest-code] for instructions on how to install a specified wheel.

#### Full build (with compilation)

If you want to modify C++ or CUDA code, you'll need to build vLLM from source. This can take several minutes:

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

!!! tip
    Building from source requires a lot of compilation. If you are building from source repeatedly, it's more efficient to cache the compilation results.

    For example, you can install [ccache](https://github.com/ccache/ccache) using `conda install ccache` or `apt install ccache` .
    As long as `which ccache` command can find the `ccache` binary, it will be used automatically by the build system. After the first build, subsequent builds will be much faster.

    When using `ccache` with `pip install -e .`, you should run `CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .`. This is because `pip` creates a new folder with a random name for each build, preventing `ccache` from recognizing that the same files are being built.

    [sccache](https://github.com/mozilla/sccache) works similarly to `ccache`, but has the capability to utilize caching in remote storage environments.
    The following environment variables can be set to configure the vLLM `sccache` remote: `SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1`. We also recommend setting `SCCACHE_IDLE_TIMEOUT=0`.

##### Use an existing PyTorch installation

There are scenarios where the PyTorch dependency cannot be easily installed via pip, e.g.:

- Building vLLM with PyTorch nightly or a custom PyTorch build.
- Building vLLM with aarch64 and CUDA (GH200), where the PyTorch wheels are not available on PyPI. Currently, only the PyTorch nightly has wheels for aarch64 with CUDA. You can run `pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124` to [install PyTorch nightly](https://pytorch.org/get-started/locally/), and then build vLLM on top of it.

To build vLLM using an existing PyTorch installation:

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py
pip install -r requirements/build.txt
pip install --no-build-isolation -e .
```

##### Use the local cutlass for compilation

Currently, before starting the build process, vLLM fetches cutlass code from GitHub. However, there may be scenarios where you want to use a local version of cutlass instead.
To achieve this, you can set the environment variable VLLM_CUTLASS_SRC_DIR to point to your local cutlass directory.

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_CUTLASS_SRC_DIR=/path/to/cutlass pip install -e .
```

##### Troubleshooting

To avoid your system being overloaded, you can limit the number of compilation jobs
to be run simultaneously, via the environment variable `MAX_JOBS`. For example:

```console
export MAX_JOBS=6
pip install -e .
```

This is especially useful when you are building on less powerful machines. For example, when you use WSL it only [assigns 50% of the total memory by default](https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings), so using `export MAX_JOBS=1` can avoid compiling multiple files simultaneously and running out of memory.
A side effect is a much slower build process.

Additionally, if you have trouble building vLLM, we recommend using the NVIDIA PyTorch Docker image.

```console
# Use `--ipc=host` to make sure the shared memory is large enough.
docker run \
    --gpus all \
    -it \
    --rm \
    --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
```

If you don't want to use docker, it is recommended to have a full installation of CUDA Toolkit. You can download and install it from [the official website](https://developer.nvidia.com/cuda-toolkit-archive). After installation, set the environment variable `CUDA_HOME` to the installation path of CUDA Toolkit, and make sure that the `nvcc` compiler is in your `PATH`, e.g.:

```console
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:$PATH"
```

Here is a sanity check to verify that the CUDA Toolkit is correctly installed:

```console
nvcc --version # verify that nvcc is in your PATH
${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME
```

#### Unsupported OS build

vLLM can fully run only on Linux but for development purposes, you can still build it on other systems (for example, macOS), allowing for imports and a more convenient development environment. The binaries will not be compiled and won't work on non-Linux systems.

Simply disable the `VLLM_TARGET_DEVICE` environment variable before installing:

```console
export VLLM_TARGET_DEVICE=empty
pip install -e .
```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

See [deployment-docker-pre-built-image][deployment-docker-pre-built-image] for instructions on using the official Docker image.

Another way to access the latest code is to use the docker images:

```console
export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}
```

These docker images are used for CI and testing only, and they are not intended for production use. They will be expired after several days.

The latest code can contain bugs and may not be stable. Please use it with caution.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

See [deployment-docker-build-image-from-source][deployment-docker-build-image-from-source] for instructions on building the Docker image.

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

See [feature-x-hardware][feature-x-hardware] compatibility matrix for feature support information.

# --8<-- [end:supported-features]
# --8<-- [end:extra-information]
