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

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

Install recommended compiler. We recommend to use `gcc/g++ >= 12.3.0` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

```bash
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

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
    - If you are building vLLM from source and not using the pre-built images, remember to set `LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD"` on x86 machines before running vLLM.

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:pre-built-images]

[https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo](https://gallery.ecr.aws/q9t5s3a7/vllm-cpu-release-repo)

!!! warning
    If deploying the pre-built images on machines without `avx512f`, `avx512_bf16`, or `avx512_vnni` support, an `Illegal instruction` error may be raised. It is recommended to build images for these machines with the appropriate build arguments (e.g., `--build-arg VLLM_CPU_DISABLE_AVX512=true`, `--build-arg VLLM_CPU_AVX512BF16=false`, or `--build-arg VLLM_CPU_AVX512VNNI=false`) to disable unsupported features. Please note that without `avx512f`, AVX2 will be used and this version is not recommended because it only has basic feature support.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_AVX512BF16=false (default)|true \
        --build-arg VLLM_CPU_AVX512VNNI=false (default)|true \
        --build-arg VLLM_CPU_DISABLE_AVX512=false (default)|true \ 
        --tag vllm-cpu-env \
        --target vllm-openai .

# Launching OpenAI server
docker run --rm \
            --security-opt seccomp=unconfined \
            --cap-add SYS_NICE \
            --shm-size=4g \
            -p 8000:8000 \
            -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
            -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
            vllm-cpu-env \
            --model=meta-llama/Llama-3.2-1B-Instruct \
            --dtype=bfloat16 \
            other vLLM OpenAI server arguments
```

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]