<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM has experimental support for s390x architecture on IBM Z platform. For now, users must build from source to natively run on IBM Z platform.

Currently, the CPU implementation for s390x architecture supports FP32, BF16 and FP16.

--8<-- [end:installation]
--8<-- [start:requirements]

- OS: `Linux`
- SDK: `gcc/g++ >= 14.0.0` or later with Command Line Tools
- Instruction Set Architecture (ISA): VXE support is required. Works with Z14 and above.
- Build from source python packages (no pre-built s390x wheels): `torchvision`, `llvmlite`, `numba`, `opencv-python-headless`, `hf-xet`

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

Currently, there are no pre-built IBM Z CPU wheels.

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

Install the following packages from the package manager before building the vLLM. For example on RHEL 9.6:

```bash
dnf install -y \
    which procps findutils tar vim git patch xz ninja-build \
    gcc-toolset-14 gcc-toolset-14-binutils gcc-toolset-14-libatomic-devel zlib-devel \
    libjpeg-turbo-devel libtiff-devel libpng-devel libwebp-devel freetype-devel harfbuzz-devel \
    openssl-devel openblas openblas-devel autoconf automake libtool cmake numpy libsndfile \
    clang llvm-devel llvm-static clang-devel
```

Build and install `numactl` from source:

```bash
curl -LO https://github.com/numactl/numactl/archive/refs/tags/v2.0.19.tar.gz
tar -xvzf v2.0.19.tar.gz
cd numactl-2.0.19
./autogen.sh && ./configure && make && make install
cd ..
```

Install rust>=1.80 which is needed for `outlines-core`, `uvloop`, and `hf-xet` python packages installation.

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env"
```

Execute the following commands to build and install vLLM from source.

!!! tip
    Pre-built wheels are not available for s390x for the following packages. Build them from source before building vLLM: `torchvision`, `llvmlite`, `numba`, `opencv-python-headless`, `hf-xet`.
    See `docker/Dockerfile.s390x` for exact versions and build commands used in each multi-stage build.

!!! note "LLVM 20 required for llvmlite"
    `llvmlite v0.47` requires LLVM 20, but UBI 9.6 repos ship LLVM 21 which is
    not compatible. You must build LLVM 20 from source before building `llvmlite`:

    ```bash
    curl -LO https://github.com/llvm/llvm-project/releases/download/llvmorg-20.1.8/llvm-project-20.1.8.src.tar.xz
    tar -xf llvm-project-20.1.8.src.tar.xz
    cmake -G Ninja -S llvm-project-20.1.8.src/llvm -B llvm-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm20 \
        -DLLVM_TARGETS_TO_BUILD="SystemZ" \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_BUILD_TOOLS=OFF \
        -DLLVM_BUILD_UTILS=ON \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_BUILD_TESTS=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF
    ninja -C llvm-build install
    ```

    Then build `llvmlite` pointing to LLVM 20:

    ```bash
    CMAKE_PREFIX_PATH=/opt/llvm20 LLVM_CONFIG=/opt/llvm20/bin/llvm-config \
        python setup.py bdist_wheel
    ```

```bash
uv pip install -v \
    /path/to/torchvision.whl \
    /path/to/llvmlite.whl \
    /path/to/numba.whl \
    /path/to/opencv_python_headless.whl \
    /path/to/hf_xet.whl \
    -r requirements/build/cpu.txt \
    -r requirements/cpu.txt \
    --torch-backend cpu \
    --index-strategy unsafe-best-match && \
VLLM_TARGET_DEVICE=cpu VLLM_CPU_MOE_PREPACK=0 python setup.py bdist_wheel && \
    uv pip install dist/*.whl
```

??? console "pip"
    ```bash
    pip install -v \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        /path/to/torchvision.whl \
        /path/to/llvmlite.whl \
        /path/to/numba.whl \
        /path/to/opencv_python_headless.whl \
        /path/to/hf_xet.whl \
        -r requirements/build/cpu.txt \
        -r requirements/cpu.txt && \
    VLLM_TARGET_DEVICE=cpu VLLM_CPU_MOE_PREPACK=0 python setup.py bdist_wheel && \
        pip install dist/*.whl
    ```

!!! warning "Protobuf workaround for s390x"
    The C++ protobuf extension crashes on s390x. After installation, set the
    following environment variable and remove the C++ extensions:

    ```bash
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

    # Remove C++ protobuf extensions that crash on s390x
    SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
    rm -rf "$SITE_PKGS/google/_upb/"*.so \
           "$SITE_PKGS/google/protobuf/pyext/"*.so 2>/dev/null || true
    ```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

Currently, there are no pre-built IBM Z CPU images.

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.s390x \
    --tag vllm-cpu-env .

# Launch OpenAI server
docker run --rm \
    --security-opt seccomp=unconfined \
    --cap-add SYS_NICE \
    --shm-size 4g \
    -p 8000:8000 \
    -e VLLM_CPU_KVCACHE_SPACE=<KV cache space> \
    -e VLLM_CPU_OMP_THREADS_BIND=<CPU cores for inference> \
    vllm-cpu-env \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dtype bfloat16 \
    other vLLM OpenAI server arguments
```

!!! tip
    Alternatively, `--privileged=true` also works but is broader and not generally recommended.

--8<-- [end:build-image-from-source]
--8<-- [start:extra-information]
--8<-- [end:extra-information]
