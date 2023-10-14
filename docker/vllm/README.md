# vLLM Dockerfile

To build the docker image, run the following command:

```sh
DOCKER_BUILDKIT=1 docker build . --tag vllm
```

To run the container, ensure that the `nvidia-docker2` runtime is installed, and run

```sh
docker run --runtime nvidia --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    vllm python3 -m vllm.entrypoints.api_server <args...>
```

## Distributed

To run on multiple machines, set up a Ray cluster via one of the methods described in [the Ray documentation](https://docs.ray.io/en/master/cluster/getting-started.html), using this image rather than `rayproject/ray` where necessary. Then exec `python3 -m vllm.entrypoints.api_server <args...>` on the container hosting the Ray head node.

