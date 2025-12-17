# --8<-- [start:installation]

vLLM supports basic model inferencing and serving on x86 CPU platform, with data types FP32, FP16 and BF16.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- OS: Linux
- CPU flags: `avx512f` (Recommended), `avx512_bf16` (Optional), `avx512_vnni` (Optional)

!!! tip
    Use `lscpu` to check the CPU flags.

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Pre-built vLLM wheels for x86 with AVX512 are available since version 0.13.0. To install release wheels:

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')

# use uv
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_VERSION}/cpu --index-strategy first-index --torch-backend cpu
```
??? console "pip"
    ```bash
    # use pip
    pip install vllm==${VLLM_VERSION}+cpu --extra-index-url https://wheels.vllm.ai/${VLLM_VERSION}/cpu --extra-index-url https://download.pytorch.org/whl/cpu
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

**Install the latest code**

To install the wheel built from the latest main branch:

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index --torch-backend cpu
```

**Install specific revisions**

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), you can specify the commit hash in the URL:

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # use full commit hash from the main branch
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index --torch-backend cpu
```

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

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
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

Optionally, build a portable wheel which you can then install elsewhere:

```bash
VLLM_TARGET_DEVICE=cpu uv build --wheel
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

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

[https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo](https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo)

!!! warning
    If deploying the pre-built images on machines without `avx512f`, `avx512_bf16`, or `avx512_vnni` support, an `Illegal instruction` error may be raised. See the build-image-from-source section below for build arguments to match your target CPU capabilities.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

## Building for your target CPU

vLLM supports building Docker images for specific CPU instruction sets, including cross-compilation (building on one system for deployment on another).

### Cross-compilation: Building on any machine

The `VLLM_CPU_AVX2` and `VLLM_CPU_AVX512` build arguments enable cross-compilation. This means you can build images for AVX512 or AVX2 CPUs **even if your build machine doesn't have those instruction sets**.

**Example: Building AVX512 image on a machine without AVX512**

```bash
# This works even if your build system lacks AVX512!
# The compiler generates AVX512 code that will run on your deployment servers
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_AVX512=true \
        --build-arg VLLM_CPU_AVX512BF16=true \
        --build-arg VLLM_CPU_AVX512VNNI=true \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

**Example: Building AVX2 image on a machine without AVX2**

```bash
# This works on any x86_64 build system
# The resulting image will run on AVX2-capable CPUs (2013+)
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_AVX2=true \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

!!! tip "When to use cross-compilation"
    - Building on CI/CD systems that may not have the same CPU as your production servers
    - Building on development machines for deployment to different server types
    - Creating portable images without needing access to specific hardware

    The cross-compilation flags tell the compiler which instruction set to target, regardless of the build system's CPU capabilities.

### Native builds (auto-detection)

If you're building directly on a machine with the same CPU as your target, you can omit the cross-compilation flags and let vLLM auto-detect the CPU features:

```bash
# Auto-detects AVX512/AVX2 from the build system's CPU
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

!!! warning "Auto-detection requires matching CPU"
    Without cross-compilation flags, the build will **fail** if your build system lacks AVX2/AVX512 support, even if you intend to deploy on a system that has these features. You must use the cross-compilation flags (`VLLM_CPU_AVX2` or `VLLM_CPU_AVX512`) when building on systems without the target CPU instruction sets.

### Advanced: Fine-tuning instruction set support

If you need to disable specific AVX512 features on systems that have AVX512 but lack certain extensions:

```bash
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_AVX512BF16=false (default)|true \
        --build-arg VLLM_CPU_AVX512VNNI=false (default)|true \
        --build-arg VLLM_CPU_AMXBF16=false|true (default) \
        --build-arg VLLM_CPU_DISABLE_AVX512=false (default)|true \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

!!! warning
    AVX2-only builds have limited feature support compared to AVX512. For best performance, use AVX512 if your target CPU supports it.

## Launching the OpenAI server

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

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]