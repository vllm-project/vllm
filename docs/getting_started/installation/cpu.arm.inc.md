# --8<-- [start:installation]

vLLM offers basic model inferencing and serving on Arm CPU platform, with support for NEON, data types FP32, FP16 and BF16.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- OS: Linux
- Compiler: `gcc/g++ >= 12.3.0` (optional, recommended)
- Instruction Set Architecture (ISA): NEON support is required

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Pre-built vLLM wheels for Arm are available since version 0.11.2. These wheels contain pre-compiled C++ binaries.

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_aarch64.whl
```

??? console "pip"
    ```bash
    pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_aarch64.whl
    ```

!!! warning "set `LD_PRELOAD`"
    Before use vLLM CPU installed via wheels, make sure TCMalloc is installed and added to `LD_PRELOAD`:
    ```bash
    # install TCMalloc
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    TC_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
    ```

The `uv` approach works for vLLM `v0.6.6` and later. A unique feature of `uv` is that packages in `--extra-index-url` have [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes). If the latest public release is `v0.6.6.post1`, `uv`'s behavior allows installing a commit before `v0.6.6.post1` by specifying the `--extra-index-url`. In contrast, `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a development version prior to the released version.

**Install the latest code**

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides working pre-built Arm CPU wheels for every commit since `v0.11.2` on <https://wheels.vllm.ai/nightly>. For native CPU wheels, this index should be used:

* `https://wheels.vllm.ai/nightly/cpu/vllm`

To install from nightly index, run:
```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cpu --index-strategy first-index
```

??? console "pip (there's a caveat)"

    Using `pip` to install from nightly indices is _not supported_, because `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a development version prior to the released version. In contrast, `uv` gives the extra index [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).

    If you insist on using `pip`, you have to specify the full URL (link address) of the wheel file (which can be obtained from https://wheels.vllm.ai/nightly/cpu/vllm).

    ```bash
    pip install https://wheels.vllm.ai/4fa7ce46f31cbd97b4651694caf9991cc395a259/vllm-0.13.0rc2.dev104%2Bg4fa7ce46f.cpu-cp38-abi3-manylinux_2_35_aarch64.whl # current nightly build (the filename will change!)
    ```

**Install specific revisions**

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), you can specify the commit hash in the URL:

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # use full commit hash from the main branch
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cpu --index-strategy first-index
```

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

First, install the recommended compiler. We recommend using `gcc/g++ >= 12.3.0` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

```bash
sudo apt-get update  -y
sudo apt-get install -y --no-install-recommends ccache git curl wget ca-certificates gcc-12 g++-12 libtcmalloc-minimal4 libnuma-dev ffmpeg libsm6 libxext6 libgl1 jq lsof
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

Second, clone the vLLM project:

```bash
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

Third, install required dependencies:

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

Finally, build and install vLLM:

```bash
VLLM_TARGET_DEVICE=cpu uv pip install . --no-build-isolation
```

If you want to develop vLLM, install it in editable mode instead.

```bash
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

Testing has been conducted on AWS Graviton3 instances for compatibility.

!!! warning "set `LD_PRELOAD`"
    Before use vLLM CPU installed via wheels, make sure TCMalloc is installed and added to `LD_PRELOAD`:
    ```bash
    # install TCMalloc
    sudo apt-get install -y --no-install-recommends libtcmalloc-minimal4

    # manually find the path
    sudo find / -iname *libtcmalloc_minimal.so.4
    TC_PATH=...

    # add them to LD_PRELOAD
    export LD_PRELOAD="$TC_PATH:$LD_PRELOAD"
    ```

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

See [Using Docker](../../deployment/docker.md) for instructions on using the official Docker image.

Stable vLLM Docker images are being pre-built for Arm from version 0.12.0. Available image tags are here: [https://gallery.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo](https://gallery.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo).

```bash
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
docker pull public.ecr.aws/q9t5s3a7/vllm-arm64-cpu-release-repo:v${VLLM_VERSION}
```

You can also access the latest code with Docker images. These are not intended for production use and are meant for CI and testing only. They will expire after several days.

The latest code can contain bugs and may not be stable. Please use it with caution.

```bash
export VLLM_COMMIT=6299628d326f429eba78736acb44e76749b281f5 # use full commit hash from the main branch
docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}-arm64-cpu
```

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

## Building for your target ARM CPU

```bash
docker build -f docker/Dockerfile.cpu \
        --platform=linux/arm64 \
        --build-arg VLLM_CPU_ARM_BF16=<false (default)|true> \
        --tag vllm-cpu-env \
        --target vllm-openai .
```

!!! note "Auto-detection by default"
    By default, ARM CPU instruction sets (BF16, NEON, etc.) are automatically detected from the build system's CPU flags. The `VLLM_CPU_ARM_BF16` build argument is used for cross-compilation:

    - `VLLM_CPU_ARM_BF16=true` - Force-enable ARM BF16 support (build with BF16 regardless of build system capabilities)
    - `VLLM_CPU_ARM_BF16=false` - Rely on auto-detection (default)

### Examples

**Auto-detection build (native ARM)**

```bash
# Building on ARM64 system - platform auto-detected
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-arm64 \
        --target vllm-openai .
```

**Cross-compile for ARM with BF16 support**

```bash
# Building on ARM64 for newer ARM CPUs with BF16
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_ARM_BF16=true \
        --tag vllm-cpu-arm64-bf16 \
        --target vllm-openai .
```

**Cross-compile from x86_64 to ARM64 with BF16**

```bash
# Requires Docker buildx with ARM emulation (QEMU)
docker buildx build -f docker/Dockerfile.cpu \
        --platform=linux/arm64 \
        --build-arg VLLM_CPU_ARM_BF16=true \
        --build-arg max_jobs=4 \
        --tag vllm-cpu-arm64-bf16 \
        --target vllm-openai \
        --load .
```

!!! note "ARM BF16 requirements"
    ARM BF16 support requires ARMv8.6-A or later (FEAT_BF16). Supported on AWS Graviton3/4, AmpereOne, and other recent ARM processors.

## Launching the OpenAI server

```bash
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
            vllm-cpu-arm64 \
            meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

!!! tip "Alternative to --privileged"
    Instead of `--privileged=true`, use `--cap-add SYS_NICE --security-opt seccomp=unconfined` for better security.

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]
