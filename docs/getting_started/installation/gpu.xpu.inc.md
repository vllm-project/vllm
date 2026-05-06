<!-- markdownlint-disable MD041 -->
--8<-- [start:installation]

vLLM initially supports basic model inference and serving on Intel GPU platform.

--8<-- [end:installation]
--8<-- [start:requirements]

- Supported Hardware: Intel Data Center GPU, Intel ARC GPU
- Dependency: [vllm-xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels): a package provide all necessary vllm custom kernel when running vLLM on Intel GPU platform,
- Python: 3.12
!!! warning
    The provided vllm-xpu-kernels whl is Python3.12 specific so this version is a MUST.

--8<-- [end:requirements]
--8<-- [start:set-up-using-python]

There is no extra information on creating a new Python environment for this device.

--8<-- [end:set-up-using-python]
--8<-- [start:pre-built-wheels]

Currently, there are no pre-built XPU wheels.

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

- First, install required [driver](https://dgpu-docs.intel.com/driver/installation.html#installing-gpu-drivers).
- Second, install Python packages for vLLM XPU backend building (Intel OneAPI dependencies are installed automatically as part of `torch-xpu`, see [PyTorch XPU get started](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)):

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- Then, install the correct Triton package for Intel XPU.

    The default `triton` package (for NVIDIA GPUs) may be installed as a transitive dependency (e.g., via `xgrammar`). For Intel XPU, you must replace it with `triton-xpu`:

    ```bash
    pip uninstall -y triton triton-xpu
    pip install triton-xpu==3.6.0 --extra-index-url https://download.pytorch.org/whl/xpu
    ```

    !!! note
        - `triton` (without suffix) is for NVIDIA GPUs only. On XPU, using it instead of `triton-xpu` can cause correctness or runtime issues.
        - For torch 2.11 (the version used in `requirements/xpu.txt`), the matching package is `triton-xpu==3.7.0`. If you use a different version of torch, check the corresponding `triton-xpu` version in [docker/Dockerfile.xpu](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.xpu).

- Finally, build and install vLLM XPU backend:

```bash
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

Currently, we release prebuilt XPU images at docker [hub](https://hub.docker.com/r/intel/vllm/tags) based on vLLM released version. For more information, please refer release [note](https://github.com/intel/ai-containers/blob/main/vllm).

--8<-- [end:pre-built-images]
--8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --device /dev/dri:/dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             --ipc=host \
             --privileged \
             vllm-xpu-env
```

--8<-- [end:build-image-from-source]
--8<-- [start:supported-features]

XPU platform supports **tensor parallel** inference/serving and also supports **pipeline parallel** as a beta feature for online serving. For **pipeline parallel**, we support it on single node with mp as the backend. For example, a reference execution like following:

```bash
vllm serve facebook/opt-13b \
     --dtype=bfloat16 \
     --max_model_len=1024 \
     --distributed-executor-backend=mp \
     --pipeline-parallel-size=2 \
     -tp=8
```

By default, a ray instance will be launched automatically if no existing one is detected in the system, with `num-gpus` equals to `parallel_config.world_size`. We recommend properly starting a ray cluster before execution, referring to the [examples/online_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/online_serving/run_cluster.sh) helper script.

--8<-- [end:supported-features]
--8<-- [start:distributed-backend]

XPU platform uses **torch-ccl** for torch<2.8 and **xccl** for torch>=2.8 as distributed backend, since torch 2.8 supports **xccl** as built-in backend for XPU.

--8<-- [end:distributed-backend]
