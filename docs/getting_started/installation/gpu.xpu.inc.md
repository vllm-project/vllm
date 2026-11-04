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

Pre-built vLLM XPU wheels are published to `wheels.vllm.ai`. Each XPU wheel
index also contains the `triton==3.7.2+xpu` shim described below. PyTorch XPU
packages are served from the PyTorch XPU index, so both index URLs are needed.

#### Install the latest code

To install the wheel built from the latest main branch:

```bash
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/xpu --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
```

#### Install specific revisions

If you want to access the wheels for previous commits (e.g. to bisect the behavior change, performance regression), you can specify the commit hash in the URL:

```bash
export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519 # use full commit hash from the main branch
uv pip install vllm --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/xpu --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
```

--8<-- [end:pre-built-wheels]
--8<-- [start:build-wheel-from-source]

- First, install required [driver](https://dgpu-docs.intel.com/driver/installation.html#installing-gpu-drivers).
- Second, install Python packages for vLLM XPU backend building (Intel OneAPI dependencies are installed automatically as part of `torch-xpu`, see [PyTorch XPU get started](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html)):
- Start from vllm-xpu-kernels v0.1.10, we recommend user upgrade driver to [compute runtime 26.18](https://github.com/intel/compute-runtime/releases/tag/26.18.38308.1) release, to avoid potential compatibility issue.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- Then, install vLLM XPU backend:

```bash
VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
```

!!! note
    `requirements/xpu.txt` pins `triton==3.7.2+xpu`, a compatibility shim
    hosted on `https://wheels.vllm.ai/xpu` that transparently resolves to
    the real Intel XPU implementation (`triton-xpu`). This exists because
    some transitive dependencies (e.g. `xgrammar`) unconditionally
    require a distribution literally named `triton`, which otherwise
    resolves to the NVIDIA-only PyPI `triton` package on XPU and can
    cause correctness or runtime issues. No manual uninstall/reinstall of
    `triton`/`triton-xpu` is needed; both `pip install` and `uv pip
    install --index-strategy unsafe-best-match` resolve the correct
    package automatically.

--8<-- [end:build-wheel-from-source]
--8<-- [start:pre-built-images]

vLLM offers official Docker images for deployment.
The images can be used to run OpenAI compatible server and are available on Docker Hub as [vllm/vllm-openai-xpu](https://hub.docker.com/r/vllm/vllm-openai-xpu/tags).

- `vllm/vllm-openai-xpu:latest` — stable release, available starting from v0.26.0
- `vllm/vllm-openai-xpu:nightly` — preview build from the latest development branch, use this if you want the latest features and fixes

```bash
docker run --rm \
    --network=host \
    --device /dev/dri:/dev/dri \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    --ipc=host \
    --privileged \
    vllm/vllm-openai-xpu:<tag> \
    --model Qwen/Qwen3-0.6B
```

To use the docker image as base for development, you can launch it in interactive session through overriding the entrypoint.

???+ console "Commands"
    ```bash
    docker run --rm -it \
        --network=host \
        --device /dev/dri:/dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --ipc=host \
        --privileged \
        --entrypoint /bin/bash \
        vllm/vllm-openai-xpu:<tag>
    ```

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

By default, a ray instance will be launched automatically if no existing one is detected in the system, with `num-gpus` equals to `parallel_config.world_size`. We recommend properly starting a ray cluster before execution, referring to the [examples/ray_serving/run_cluster.sh](https://github.com/vllm-project/vllm/blob/main/examples/ray_serving/run_cluster.sh) helper script.

--8<-- [end:supported-features]
--8<-- [start:distributed-backend]

XPU platform uses **torch-ccl** for torch<2.8 and **xccl** for torch>=2.8 as distributed backend, since torch 2.8 supports **xccl** as built-in backend for XPU.

--8<-- [end:distributed-backend]
