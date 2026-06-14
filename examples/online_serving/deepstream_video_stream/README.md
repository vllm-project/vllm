# DeepStream Backend for vLLM Video Inference

GPU-resident video decoding via NVIDIA DeepStream 9.0 for the vLLM
OpenAI-compatible server. Adds the `VLLM_VIDEO_LOADER_BACKEND=deepstream`
option which decodes video chunks directly in GPU memory via NVDEC +
GStreamer.

## Prerequisites

- NVIDIA GPU with NVDEC support (H100, H200, A100, L40, etc.)
- Host NVIDIA driver: CUDA 13.0+ compatible (R570+)
- Docker with `nvidia-container-toolkit`
- Ubuntu 24.04 base
- Python 3.12

## Files in this directory

| File | Purpose |
|---|---|
| `Dockerfile` | Build the DeepStream-enabled vLLM image |
| `start_server.sh` | Launch vLLM server with the DS video loader |
| `run_bench_client.sh` | Throughput benchmark client wrapper |
| `bench_h264_client.py` | H.264 multi-prompt benchmark client |

## 1. Build the base vLLM image

```bash
# From the repo root
docker build \
    --build-arg UBUNTU_VERSION=24.04 \
    --build-arg BUILD_BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04 \
    --build-arg torch_cuda_arch_list="9.0" \
    --build-arg max_jobs=40 \
    --build-arg nvcc_threads=2 \
    --build-arg RUN_WHEEL_CHECK=false \
    --target vllm-openai \
    -f docker/Dockerfile \
    -t forked_vllm:latest \
    .
```

Adjust `torch_cuda_arch_list` for your GPU (`9.0` H100/H200, `8.0` A100,
`8.9` L40, `8.6` A6000/RTX 30xx). Multi-arch is supported (e.g. `"8.0 9.0"`)
at the cost of longer build times.

## 2. Download the DeepStream runtime

From <https://developer.nvidia.com/deepstream-getting-started> (NVIDIA
developer account required, license must be accepted):

- `deepstream-libs-minimal_9.0.0-1_amd64.deb` (or whatever 9.0 build you have — pass via `--build-arg DS_DEB=<filename>` if it differs)

Place the file in this directory before the next step.

Note: the DeepStream Python bindings (`pyds`) are **not** required —
this backend talks to GStreamer via PyGObject (`python3-gi` +
`python3-gst-1.0`, already installed in step 3) and to CUDA via raw
`ctypes` bindings to `libcudart.so`. There are no `import pyds` calls
anywhere in the new code.

## 3. Build the DeepStream-enabled image

```bash
docker build \
    --build-arg VLLM_BASE=forked_vllm:latest \
    --build-arg DS_DEB=deepstream-libs-minimal_9.0.0-1_amd64.deb \
    -t vllm-deepstream:0.1 \
    -f examples/online_serving/deepstream_video_stream/Dockerfile \
    examples/online_serving/deepstream_video_stream
```

## 4. Launch the container

```bash
docker run --rm -it --gpus all --ipc=host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -p 8000:8000 \
    -v /path/to/videos:/data/video \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /path/to/forked_vllm:/work/forked_vllm \
    --entrypoint bash \
    --name vllm-ds \
    vllm-deepstream:0.1
```

The `-v /path/to/forked_vllm:/work/forked_vllm` mount lets the example
scripts find the vLLM source tree they reference; adjust the host path
to wherever you cloned the repo.

## 5. Inside the container

### Start the server (single GPU, DS backend)

```bash
GPU_MEM=0.7 ./vllm/examples/online_serving/deepstream_video_stream/start_server.sh deepstream
```

Environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL` | `Qwen2-VL-2B-Instruct` | HF model id or local path |
| `PORT` | `8000` | API port |
| `GPU_MEM` | `0.8` | vLLM `--gpu-memory-utilization` |
| `NUM_FRAMES` | `8` | Frames sampled per video request |

### Throughput benchmark on a file input

```bash
VIDEO="/data/video/sample_1080p_h264_10s_gop30.mp4" \
    vllm/examples/online_serving/deepstream_video_stream/run_bench_client.sh \
    --num-prompts 1000 --request-rate inf
```

## Architecture

```
HTTP request (OpenAI /v1/chat/completions)
    │
    │  video_url: file:///data/video/...
    │
    ▼
VideoMediaIO  (vllm/multimodal/media/video.py)
    │  dispatches on VLLM_VIDEO_LOADER_BACKEND
    ▼
DeepStream loader  (vllm/multimodal/video.py)
    │
    ├─ load_bytes()           single file decode
    └─ stream_file_chunked()  large file in chunks (parallel ThreadPool)
    │
    ▼
DecodePool  (vllm/multimodal/ds_decode_pool.py)
    │  N GStreamer pipelines (default 8, max 16), NVDEC → CUDA buffers
    ▼
CUDA tensor (uint8 NHWC, GPU-resident)
    │
    │  .cpu().numpy() at the DS exit point
    │  (boundary chosen so upstream multimodal parser stays unchanged)
    ▼
Standard vLLM multimodal preprocessor  (unchanged)
    ▼
ViT vision tower → LLM forward → text response
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_VIDEO_LOADER_BACKEND` | `opencv` | Set to `deepstream` to enable DS |
| `VLLM_MEDIA_LOADING_THREAD_COUNT` | `8` | DS decode pool worker count (clamped to `[1, 16]`) |
| `VLLM_MULTIMODAL_TENSOR_IPC` | `0` | Use shared-memory IPC for multimodal tensors |
| `NVIDIA_DRIVER_CAPABILITIES` | (image default) | Must include `video` for NVDEC; `all` is the safest catch-all |

## Troubleshooting

- **`libnppig.so.13 => not found` / `libnppidei.so.13 => not found`**
  The base CUDA image is stripped of NPP. The Dockerfile installs
  `cuda-libraries-13-0` which pulls in NPP and the other CUDA runtime libs.
  Verify with `ldconfig -p | grep nppig` inside the container.

- **`ImportError: libcudart.so.12: cannot open shared object file`**
  Stale `.so` files were copied into the container from a different
  build. Don't overlay host-built `.so` files; let the image's own
  CUDA-13 extensions stand.

- **GPU SM% low during file benchmark**
  vLLM's prefix-cache + multimodal-cache may be hitting 99%+ on a
  same-video benchmark (look for `Prefix cache hit rate` / `MM cache hit
  rate` in server logs). Use a directory of distinct videos to bypass
  the cache and measure real ViT throughput.

## License notice

The DeepStream SDK is NVIDIA proprietary. By downloading it you accept
NVIDIA's license terms. This PR does **not** redistribute the SDK; users
must obtain it directly from NVIDIA.
