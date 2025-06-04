# --8<-- [start:build-wheel-from-source]

!!! note
    **Install vs Build:**

    - To generate a wheel for reuse or distribution, use `python -m build` then `pip install dist/*.whl`.

    - For in-place installs (no-wheel) and dev testing, use `uv pip install . --no-build-isolation`.


## Build from source (Intel/AMD x86)

!!! note
    If you're building from source on CPU, here are a few tips to avoid common issues:

    - **NumPy ≥2.0 error**: Downgrade using `pip install "numpy<2.0"`.
    - **CMake picks up CUDA**: Add `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` to prevent CUDA detection.
    - **Torch CPU wheel not resolving**: Use `--index-url` during the `requirements/cpu.txt` install.
    - **`torch==2.6.0+cpu` not found**: Set `"torch==2.6.0+cpu"` in [`pyproject.toml`](https://github.com/vllm-project/vllm/blob/main/pyproject.toml).
    - **Deprecated `setup.py install`**: Use the [PEP 517-compliant](https://peps.python.org/pep-0517/) `python -m build` instead.



### 1. Install system dependencies:
 Install recommended compiler. We recommend to use `gcc/g++ >= 12.3.0` as the default compiler to avoid potential problems. For example, on Ubuntu 22.4, you can run:

```console
sudo apt-get update -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev python3-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12
```
### 2. Clone vLLM repository

```console
git clone https://github.com/vllm-project/vllm.git vllm_source
cd vllm_source
```
### 3. Install Python build requirements 
```console 
pip install --upgrade pip
pip install "cmake>=3.26.1" wheel packaging ninja "setuptools-scm>=8" numpy
pip install -v -r requirements/cpu.txt --index-url https://download.pytorch.org/whl/cpu
```
### 4. Build and install vLLM:
**Option A: Build a wheel**

You can do this using one of the following methods: 

- Using python `build` package (recommended)   
```console
# Specify kv cache in GiB
export VLLM_CPU_KVCACHE_SPACE=2
# Check how many cores your machine have with lscpu -e (i.e values : 1,2/0-2/2)
export VLLM_CPU_OMP_THREADS_BIND=0-4 
# Build the wheel
VLLM_TARGET_DEVICE=cpu CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON python -m build --wheel --no-isolation
```
- Using `uv` (fastest option)
```
VLLM_TARGET_DEVICE=cpu CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON  uv build --wheel

```
**Install the wheel (non-editable)**
```
uv pip install dist/*.whl
```
!!! tip 
    `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` prevents picking up CUDA during CPU builds, even if it's installed.

**Option B: Install directly from source**

- Standard install:
```console
VLLM_TARGET_DEVICE=cpu CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON uv pip install . --no-build-isolation
```
- Editable install (with `-e` flag): 
```console
VLLM_TARGET_DEVICE=cpu CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON uv pip install -e . --no-build-isolation
```

!!! tip
    If you recieve an error such as: `Could not find a version that satisfies the requirement torch==2.6.0+cpu`, consider updating [pyproject.toml](https://github.com/vllm-project/vllm/blob/main/pyproject.toml) to help pip resolve the dependency.

    ```toml title="pyproject.toml"
    [build-system]
    requires = [
      "cmake>=3.26.1",
      ...
      "torch==2.6.0+cpu"   # <-------
    ]
    ```
## --8<-- [end:build-wheel-from-source]