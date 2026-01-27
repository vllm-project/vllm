# --8<-- [start:use-official-docker-image]

vLLM offers an official Docker image for deployment.
The image can be used to run OpenAI compatible server and is available on Docker Hub as [vllm/vllm-openai-rocm](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags).

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm:latest \
    --model Qwen/Qwen3-0.6B
```


You can add any other [engine-args](../configuration/engine_args.md) you need after the image tag (`vllm/vllm-openai-rocm:latest`).

!!! note
    Optional dependencies are not included in order to avoid licensing issues (e.g. <https://github.com/vllm-project/vllm/issues/8030>).

    If you need to use those dependencies (having accepted the license terms),
    create a custom Dockerfile on top of the base image with an extra layer that installs them:

    ```Dockerfile
    FROM vllm/vllm-openai-rocm:v0.14.1

    # e.g. install the `audio` optional dependencies
    # NOTE: Make sure the version of vLLM matches the base image!
    RUN uv pip install --system vllm[audio]==0.14.1 --extra-index-url https://wheels.vllm.ai/rocm/0.14.1/rocm700
    ```

!!! note
    You can either use the `ipc=host` flag or `--shm-size` flag to allow the
    container to access the host's shared memory. vLLM uses PyTorch, which uses shared
    memory to share data between processes under the hood, particularly for tensor parallel inference.

!!! tip
    Some new models may only be available on the main branch of [HF Transformers](https://github.com/huggingface/transformers).

    To use the development version of `transformers`, create a custom Dockerfile on top of the base image
    with an extra layer that installs their code from source:

    ```Dockerfile
    FROM vllm/vllm-openai-rocm:latest

    RUN uv pip install --system git+https://github.com/huggingface/transformers.git
    ```

# --8<-- [end:use-official-docker-image]
# --8<-- [start:build-docker-image-from-source]

You can build and run vLLM from source via the provided [docker/Dockerfile.rocm](../../docker/Dockerfile.rocm).

??? info "(Optional) Build an image with ROCm software stack"

    Build a docker image from [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base) which setup ROCm software stack needed by the vLLM.
    **This step is optional as this rocm_base image is usually prebuilt and store at [Docker Hub](https://hub.docker.com/r/rocm/vllm-dev) under tag `rocm/vllm-dev:base` to speed up user experience.**
    If you choose to build this rocm_base image yourself, the steps are as follows.

    It is important that the user kicks off the docker build using buildkit. Either the user put `DOCKER_BUILDKIT=1` as environment variable when calling docker build command, or the user needs to set up buildkit in the docker daemon configuration `/etc/docker/daemon.json` as follows and restart the daemon:

    ```json
    {
        "features": {
            "buildkit": true
        }
    }
    ```

    To build vllm on ROCm 7.0 for MI200 and MI300 series, you can use the default:

    ```bash
    DOCKER_BUILDKIT=1 docker build \
        -f docker/Dockerfile.rocm_base \
        -t rocm/vllm-dev:base .
    ```

First, build a docker image from [docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) and launch a docker container from the image.
It is important that the user kicks off the docker build using buildkit. Either the user put `DOCKER_BUILDKIT=1` as environment variable when calling docker build command, or the user needs to set up buildkit in the docker daemon configuration /etc/docker/daemon.json as follows and restart the daemon:

```json
{
    "features": {
        "buildkit": true
    }
}
```

[docker/Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm) uses ROCm 7.0 by default, but also supports ROCm 5.7, 6.0, 6.1, 6.2, 6.3, and 6.4, in older vLLM branches.
It provides flexibility to customize the build of docker image using the following arguments:

- `BASE_IMAGE`: specifies the base image used when running `docker build`. The default value `rocm/vllm-dev:base` is an image published and maintained by AMD. It is being built using [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base)
- `ARG_PYTORCH_ROCM_ARCH`: Allows to override the gfx architecture values from the base docker image

Their values can be passed in when running `docker build` with `--build-arg` options.

To build vllm on ROCm 7.0 for MI200 and MI300 series, you can use the default (which build a docker image with `vllm serve` as entrypoint):


```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rocm -t vllm-rocm .
```

# --8<-- [end:build-docker-image-from-source]
# --8<-- [start:use-custom-docker-image]

To run vLLM with the custom-built Docker image:

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm-rocm <args...>
``` 


The argument `vllm/vllm-openai-rocm` specifies the image to run, and should be replaced with the name of the custom-built image (the `-t` tag from the build command).

To use the docker image as base for development, you can launch it in interactive session through overriding the entrypoint.

???+ console "Commands"
    ```bash
    docker run --rm -it \
        --group-add=video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HF_TOKEN=$HF_TOKEN" \
        --network=host \
        --ipc=host \
        --entrypoint bash \
        vllm-rocm
    ```

# --8<-- [end:use-custom-docker-image]



