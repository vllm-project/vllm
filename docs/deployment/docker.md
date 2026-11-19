---
toc_depth: 2
---

# Using Docker

## Pre-built images

--8<-- "docs/getting_started/installation/gpu.md:pre-built-images"

### Short image names

Release images (from v0.33.0) and nightly images are also published under the
shorter `vllm/vllm` name, with the platform as a tag suffix. These are the same
images (same digests) as the `vllm/vllm-openai*` tags.

| Image | `vllm/vllm` tag |
| --- | --- |
| `vllm/vllm-openai:{latest,v<version>,nightly}` | `vllm/vllm:{latest,v<version>,nightly}` |
| `vllm/vllm-openai:{latest,v<version>}-efa` | `vllm/vllm:{latest,v<version>}-efa` |
| `vllm/vllm-openai:cu134-nightly` | `vllm/vllm:nightly-cu134` |
| `vllm/vllm-openai-rocm:{latest,v<version>,nightly}` | `vllm/vllm:{latest,v<version>,nightly}-rocm` |
| `vllm/vllm-openai-rocm:{latest,v<version>,nightly}-<variant>` | `vllm/vllm:{latest,v<version>,nightly}-<variant>`, e.g. `-rocm100` |
| `vllm/vllm-openai-cpu:{latest,v<version>,nightly}` | `vllm/vllm:{latest,v<version>,nightly}-cpu` |
| `vllm/vllm-openai-xpu:{latest,v<version>,nightly}` | `vllm/vllm:{latest,v<version>,nightly}-xpu` |

For example, `docker pull vllm/vllm:v0.33.0-rocm` pulls the same image as
`docker pull vllm/vllm-openai-rocm:v0.33.0`. Per-architecture tags, ROCm `-base`
images and commit-pinned nightly tags (`nightly-<commit>`) are only published
under the `vllm/vllm-openai*` names.

Release images for ROCm and XPU are also published as `vllm/vllm-rocm` and
`vllm/vllm-xpu`, with the same tags as `vllm/vllm-openai-rocm` and
`vllm/vllm-openai-xpu` (e.g. `vllm/vllm-rocm:v0.33.0`, `vllm/vllm-xpu:latest`).

## Standalone Rust frontend

The `vllm/vllm-rust-frontend` image runs on Linux AMD64 and ARM64 CPUs.
Use it for engine-free rendering or to serve requests with separate headless
vLLM engines. The runtime includes FFmpeg for video and audio preprocessing.
Use matching release tags or `nightly-<full-commit-sha>` for the frontend
and engine. The most recent 14 nightly builds are retained. The short tags
`vllm/vllm:{latest,v<version>,nightly}-rust-frontend` select the same image.

To build from a checkout, run
`docker buildx bake -f docker/docker-bake.hcl rust-frontend` from the repository
root and use `local/vllm-rust-frontend:dev` in place of the published image.

For a standalone render service:

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
    -v frontend-cache:/home/vllm/.cache/huggingface \
    vllm/vllm-rust-frontend:latest render Qwen/Qwen3-0.6B --host 0.0.0.0
```

For inference, use two containers with a shared network namespace on one Linux
host. Only the engine needs a GPU. The handshake and engine I/O can use
loopback inside that namespace. On separate hosts use the reachable IP of the
frontend for `--data-parallel-address` on both commands.

```bash
# Set TAG to a release published for both images or a commit-pinned nightly.
: "${TAG:?Set TAG to the matching vLLM image tag}"
docker run -d --name rust-frontend -p 127.0.0.1:8000:8000 \
    -v frontend-cache:/home/vllm/.cache/huggingface \
    "vllm/vllm-rust-frontend:$TAG" serve Qwen/Qwen3-0.6B --host 0.0.0.0 \
    --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 62100 \
    --data-parallel-size 1 --data-parallel-size-local 0
docker run --rm --gpus all --shm-size 2g --network container:rust-frontend \
    --entrypoint vllm "vllm/vllm-openai:$TAG" serve Qwen/Qwen3-0.6B --headless \
    --data-parallel-address 127.0.0.1 --data-parallel-rpc-port 62100 \
    --data-parallel-size 1 --data-parallel-size-local 1
```

Send requests to `http://127.0.0.1:8000/v1`. After stopping the engine,
run `docker rm -f rust-frontend` to remove the frontend container.

## Run a vLLM Recipes configuration

[vLLM Recipes](https://recipes.vllm.ai/) can be converted into `config.yaml`
and `env.sh`. See the
[Recipes conversion tool README](../../tools/recipes/README.md) for usage.

For Docker, mount both files and source `env.sh` inside the container before
starting vLLM:

```bash
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$PWD/config.yaml:/recipe/config.yaml:ro" \
    -v "$PWD/env.sh:/recipe/env.sh:ro" \
    -p 8000:8000 \
    --ipc=host \
    --entrypoint /bin/bash \
    vllm/vllm-openai:latest \
    -lc 'source /recipe/env.sh && exec vllm serve --config /recipe/config.yaml'
```

## Persist the compile cache across containers

Mounting the Hugging Face cache keeps model weights across containers, but each
new container still starts with an empty `VLLM_CACHE_ROOT` (default
`~/.cache/vllm`) and recompiles the model's `torch.compile` artifacts. Mount a
named volume at that path to reuse the inductor, Triton, and AOT artifacts from
the second container onward:

```bash
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v vllm-cache:/root/.cache/vllm \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

See [Faster Startup](../configuration/optimization.md#faster-startup) for the
mechanism and for what invalidates the cache.

## Run as a non-root user

The CUDA `vllm/vllm-openai` image runs as root by default for backward
compatibility. It is also prepared to run as the built-in `vllm` user
(UID 2000, GID 0):

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

When mounting model or cache volumes for a non-root container, mount writable
paths under `/home/vllm` instead of `/root`. For example, mount the Hugging
Face cache at `/home/vllm/.cache/huggingface` and make the mounted directory
writable by group 0.

```bash
docker run --rm --gpus all \
    --user 2000:0 \
    -v ~/.cache/huggingface:/home/vllm/.cache/huggingface \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    meta-llama/Llama-3.1-8B-Instruct
```

To build an image that defaults to the non-root `vllm` user, use the opt-in
`vllm-openai-nonroot` target:

```bash
docker build --target vllm-openai-nonroot \
    -t vllm-openai-nonroot:local \
    -f docker/Dockerfile .

docker run --rm --gpus all \
    -p 8000:8000 \
    vllm-openai-nonroot:local \
    meta-llama/Llama-3.1-8B-Instruct
```

The `vllm-openai-nonroot` target also supports OpenShift-style arbitrary UIDs
when the runtime UID is a member of group 0. In Kubernetes manifests, set the
container security context accordingly and keep mounted cache/model paths
writable by group 0:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000540000
  runAsGroup: 0
  fsGroup: 0
```

Runtime UIDs outside group 0 are not part of the documented support matrix
because they may be unable to write to `/home/vllm` or `/opt/uv/cache`.

## Build image from source

--8<-- "docs/getting_started/installation/gpu.md:build-image-from-source"
