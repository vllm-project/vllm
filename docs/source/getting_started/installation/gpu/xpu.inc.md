# Installation

vLLM initially supports basic model inference and serving on Intel GPU platform.

:::{attention}
There are no pre-built wheels or images for this device, so you must build vLLM from source.
:::

## Requirements

- Supported Hardware: Intel Data Center GPU, Intel ARC GPU
- OneAPI requirements: oneAPI 2025.0

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built XPU wheels.

### Build wheel from source

- First, install required driver and Intel OneAPI 2025.0 or later.
- Second, install Python packages for vLLM XPU backend building:

```console
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- Then, build and install vLLM XPU backend:

```console
VLLM_TARGET_DEVICE=xpu python setup.py install
```

- Finally, due to a known issue of conflict dependency(oneapi related) in torch-xpu 2.6 and ipex-xpu 2.6, we install ipex here. This will be fixed in the ipex-xpu 2.7.

```console
pip install intel-extension-for-pytorch==2.6.10+xpu \
    --extra-index-url=https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

:::{note}
- FP16 is the default data type in the current XPU backend. The BF16 data
  type is supported on Intel Data Center GPU, not supported on Intel Arc GPU yet.
:::

## Set up using Docker

### Pre-built images

Currently, there are no pre-built XPU images.

### Build image from source

```console
$ docker build -f Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
$ docker run -it \
             --rm \
             --network=host \
             --device /dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             vllm-xpu-env
```

## Supported features

XPU platform supports **tensor parallel** inference/serving and also supports **pipeline parallel** as a beta feature for online serving. We require Ray as the distributed runtime backend. For example, a reference execution like following:

```console
python -m vllm.entrypoints.openai.api_server \
     --model=facebook/opt-13b \
     --dtype=bfloat16 \
     --device=xpu \
     --max_model_len=1024 \
     --distributed-executor-backend=ray \
     --pipeline-parallel-size=2 \
     -tp=8
```

By default, a ray instance will be launched automatically if no existing one is detected in the system, with `num-gpus` equals to `parallel_config.world_size`. We recommend properly starting a ray cluster before execution, referring to the <gh-file:examples/online_serving/run_cluster.sh> helper script.

There are some new features coming with ipex-xpu 2.6, e.g. **chunked prefill**, **V1 engine support**, **lora**, **MoE**, etc.
