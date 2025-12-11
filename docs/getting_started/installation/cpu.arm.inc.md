# --8<-- [start:installation]

vLLM offers basic model inferencing and serving on Arm CPU platform, with support NEON, data types FP32, FP16 and BF16.

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
Please replace `<version>` in the commands below with a specific version string (e.g., `0.11.2`).

```bash
uv pip install --pre vllm==<version>+cpu --extra-index-url https://wheels.vllm.ai/<version>%2Bcpu/
```

??? console "pip"
    ```bash
    pip install --pre vllm==<version>+cpu --extra-index-url https://wheels.vllm.ai/<version>%2Bcpu/
    ```

The `uv` approach works for vLLM `v0.6.6` and later. A unique feature of `uv` is that packages in `--extra-index-url` have [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes). If the latest public release is `v0.6.6.post1`, `uv`'s behavior allows installing a commit before `v0.6.6.post1` by specifying the `--extra-index-url`. In contrast, `pip` combines packages from `--extra-index-url` and the default index, choosing only the latest version, which makes it difficult to install a development version prior to the released version.

**Install the latest code**

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides working pre-built Arm CPU wheels for every commit since `v0.11.2` on <https://wheels.vllm.ai/nightly>. For native CPU wheels, this index should be used:

* `https://wheels.vllm.ai/nightly/cpu/vllm`

To install from nightly index, copy the link address of the `*.whl` under this index to run, for example:

```bash
uv pip install -U https://wheels.vllm.ai/c756fb678184b867ed94e5613a529198f1aee423/vllm-0.13.0rc2.dev11%2Bgc756fb678.cpu-cp38-abi3-manylinux_2_31_aarch64.whl # current nightly build (the filename will change!)
```

**Install specific revisions**

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), specify the full commit hash in the index:
https://wheels.vllm.ai/${VLLM_COMMIT}/cpu/vllm .
Then, copy the link address of the `*.whl` under this index to run:

```bash
uv pip install -U <wheel-url>
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

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

Currently, there are no pre-built Arm CPU images.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]
```bash
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-env .

# Launching OpenAI server
docker run --rm \
            --privileged=true \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
            vllm-cpu-env \
            --model=meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

!!! tip
    An alternative of `--privileged=true` is `--cap-add SYS_NICE --security-opt seccomp=unconfined`.

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]
