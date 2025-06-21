# --8<-- [start:installation]

vLLM has been adapted to work on ARM64 CPUs with NEON support, leveraging the CPU backend initially developed for the x86 platform.

ARM CPU backend currently supports Float32, FP16 and BFloat16 datatypes.

!!! warning
    There are no pre-built wheels or images for this device, so you must build vLLM from source.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- OS: Linux
- Compiler: `gcc/g++ >= 12.3.0` (optional, recommended)
- Instruction Set Architecture (ISA): NEON support is required

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

First, install recommended compiler. We recommend to use `gcc/g++ >= 12.3.0` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

```console
sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```

Second, clone vLLM project:

```console
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```

Third, install Python packages for vLLM CPU backend building:

```console
pip install --upgrade pip
pip install "cmake>=3.26.1" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Finally, build and install vLLM CPU backend:

```console
VLLM_TARGET_DEVICE=cpu python setup.py install
```

If you want to develop vllm, install it in editable mode instead.

```console
VLLM_TARGET_DEVICE=cpu python setup.py develop
```
!!! note
    - Testing has been conducted on AWS Graviton3 instances for compatibility.

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

# --8<-- [end:build-image-from-source]
# --8<-- [start:extra-information]
# --8<-- [end:extra-information]
