# --8<-- [start:installation]

vLLM initially supports basic model inference and serving on Intel GPU platform.

!!! warning
    There are no pre-built wheels or images for this device, so you must build vLLM from source.

# --8<-- [end:installation]
# --8<-- [start:requirements]

- Supported Hardware: Intel Data Center GPU, Intel ARC GPU
- OneAPI requirements: oneAPI 2025.0

# --8<-- [end:requirements]
# --8<-- [start:set-up-using-python]

# --8<-- [end:set-up-using-python]
# --8<-- [start:pre-built-wheels]

Currently, there are no pre-built XPU wheels.

# --8<-- [end:pre-built-wheels]
# --8<-- [start:build-wheel-from-source]

- First, install required [driver](https://dgpu-docs.intel.com/driver/installation.html#installing-gpu-drivers) and [Intel OneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) 2025.0 or later.
- Second, install Python packages for vLLM XPU backend building:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- Then, build and install vLLM XPU backend:

```bash
VLLM_TARGET_DEVICE=xpu python setup.py install
```

!!! note
    - FP16 is the default data type in the current XPU backend. The BF16 data
      type is supported on Intel Data Center GPU, not supported on Intel Arc GPU yet.

# --8<-- [end:build-wheel-from-source]
# --8<-- [start:set-up-using-docker]

# --8<-- [end:set-up-using-docker]
# --8<-- [start:pre-built-images]

Currently, there are no pre-built XPU images.

# --8<-- [end:pre-built-images]
# --8<-- [start:build-image-from-source]

```bash
docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it \
             --rm \
             --network=host \
             --device /dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             vllm-xpu-env
```

# --8<-- [end:build-image-from-source]
# --8<-- [start:supported-features]

XPU platform supports **tensor parallel** inference/serving and also supports **pipeline parallel** as a beta feature for online serving. We require Ray as the distributed runtime backend. For example, a reference execution like following:

```bash
python -m vllm.entrypoints.openai.api_server \
     --model=facebook/opt-13b \
     --dtype=bfloat16 \
     --max_model_len=1024 \
     --distributed-executor-backend=ray \
     --pipeline-parallel-size=2 \
     -tp=8
```

By default, a ray instance will be launched automatically if no existing one is detected in the system, with `num-gpus` equals to `parallel_config.world_size`. We recommend properly starting a ray cluster before execution, referring to the <gh-file:examples/online_serving/run_cluster.sh> helper script.

# --8<-- [end:supported-features]
# --8<-- [end:extra-information]
