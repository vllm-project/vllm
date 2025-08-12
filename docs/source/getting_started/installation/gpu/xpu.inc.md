# Installation

vLLM initially supports basic model inferencing and serving on Intel GPU platform.

## Requirements

- Supported Hardware: Intel Data Center GPU, Intel ARC GPU
- OneAPI requirements: oneAPI 2024.2

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built XPU wheels.

### Build wheel from source

- First, install required driver and intel OneAPI 2024.2 or later.
- Second, install Python packages for vLLM XPU backend building:

```console
source /opt/intel/oneapi/setvars.sh
pip install --upgrade pip
pip install -v -r requirements-xpu.txt
```

- Finally, build and install vLLM XPU backend:

```console
VLLM_TARGET_DEVICE=xpu python setup.py install
```

```{note}
- FP16 is the default data type in the current XPU backend. The BF16 data
  type will be supported in the future.
```

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

XPU platform supports tensor-parallel inference/serving and also supports pipeline parallel as a beta feature for online serving. We requires Ray as the distributed runtime backend. For example, a reference execution likes following:

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

By default, a ray instance will be launched automatically if no existing one is detected in system, with `num-gpus` equals to `parallel_config.world_size`. We recommend properly starting a ray cluster before execution, referring to the <gh-file:examples/online_serving/run_cluster.sh> helper script.
