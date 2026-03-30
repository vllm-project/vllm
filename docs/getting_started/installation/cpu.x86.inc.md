<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM supports basic model inferencing and serving on x86 CPU platform, with data types FP32, FP16 and BF16.

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: Linux
- CPU flags: `avx512f` (Recommended), `avx2` (Limited features)

!!! tip
    Use `lscpu` to check the CPU flags.

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

Pre-built vLLM wheels for x86 with AVX512/AVX2 are available since version 0.17.0. To install release wheels:

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')

# use uv
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --torch-backend cpu
```

??? console "pip"
    ```bash
    # use pip
    pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cpu
    ```
!!! warning "set `LD_PRELOAD`"
    Before use vLLM CPU installed via wheels, make sure TCMalloc and Intel OpenMP are installed and added to `LD_PRELOAD`:
    ```bash
    # install TCMalloc, Intel OpenMP is installed with vLLM CPU
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```

#### Install the latest code

To install the wheel built from the latest main branch:

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

#### Install specific revisions

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), you can specify the commit hash in the URL:

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # use full commit hash from the main branch
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index --torch-backend cpu
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

Install recommended compiler. We recommend to use `gcc/g++ >= 12.3.0` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

```bash
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

Clone the vLLM project:

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

Install the required dependencies:

```bash
uv pip install -r requirements/cpu-build.txt --torch-backend cpu
uv pip install -r requirements/cpu.txt --torch-backend cpu
```

??? console "pip"
    ```bash
    pip install --upgrade pip
    pip install -v -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
    pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```

Build and install vLLM:

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

If you want to develop vLLM, install it in editable mode instead.

```bash
VLLM_TARGET_DEVICE=cpu python3 setup.py develop
```

Optionally, build a portable wheel which you can then install elsewhere:

```bash
VLLM_TARGET_DEVICE=cpu uv build --wheel --no-build-isolation
```

```bash
uv pip install dist/*.whl
```

??? console "pip"
    ```bash
    VLLM_TARGET_DEVICE=cpu python -m build --wheel --no-isolation
    ```

    ```bash
    pip install dist/*.whl
    ```

!!! warning "set `LD_PRELOAD`"
    Before use vLLM CPU installed via wheels, make sure TCMalloc and Intel OpenMP are installed and added to `LD_PRELOAD`:
    ```bash
    # install TCMalloc, Intel OpenMP is installed with vLLM CPU
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    sudo find / -iname *libiomp5.so
    TC_PATH=...
    IOMP_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    ```

!!! example "Troubleshooting"
    - **NumPy ≥2.0 error**: Downgrade using `pip install "numpy<2.0"`.
    - **CMake picks up CUDA**: Add `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` to prevent CUDA detection during CPU builds, even if CUDA is installed.
    - `AMD` requires at least 4th gen processors (Zen 4/Genoa) or higher to support [AVX512](https://www.phoronix.com/review/amd-zen4-avx512) to run vLLM on CPU.
    - If you receive an error such as: `Could not find a version that satisfies the requirement torch==X.Y.Z+cpu+cpu`, consider updating [pyproject.toml](https://github.com/vllm-project/vllm/blob/main/pyproject.toml) to help pip resolve the dependency.
    ```toml title="pyproject.toml"
    [build-system]
    requires = [
      "cmake>=3.26.1",
      ...
      "torch==X.Y.Z+cpu"   # <-------
    ]
    ```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

You can pull the latest available CPU image from Docker Hub:

```bash
docker pull vllm/vllm-openai-cpu:latest-x86_64
```

To pull an image for a specific vLLM version:

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
docker pull vllm/vllm-openai-cpu:v${VLLM_VERSION}-x86_64
```

All available image tags are here: [https://hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags)

You can run these images via:

```bash
docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<secret>" \
    vllm/vllm-openai-cpu:latest-x86_64 <args...>
```

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

#### Building for your target CPU

```bash
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_X86=<false (default)|true> \ # For cross-compilation
        --tag vllm-cpu-env \
        --target vllm-openai .
```

#### Launching the OpenAI server

```bash
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            vllm-cpu-env \
            meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
