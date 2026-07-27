# DeepStream backend extension for vLLM
# ======================================
# Adds GStreamer + DeepStream 9.0 to an existing vLLM image so the
# `VLLM_VIDEO_LOADER_BACKEND=deepstream` path can decode video on GPU
# (NVDEC) through the OpenAI chat-completions API.
#
# Prerequisites (place next to this Dockerfile before building):
#   - deepstream-libs-minimal_<version>-1_amd64.deb
#       Get it from https://developer.nvidia.com/deepstream-getting-started
#       (DeepStream 9.0 — adjust DS_DEB build-arg if your filename differs)
#
# Build the underlying vLLM image first (only once):
#   docker build \
#       --build-arg UBUNTU_VERSION=24.04 \
#       --build-arg BUILD_BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04 \
#       --build-arg torch_cuda_arch_list="9.0" \
#       --build-arg max_jobs=40 \
#       --build-arg nvcc_threads=2 \
#       --build-arg RUN_WHEEL_CHECK=false \
#       --target vllm-openai \
#       -f docker/Dockerfile \
#       -t forked_vllm:latest \
#       .
#
# Then build this DS layer:
#   docker build \
#       --build-arg VLLM_BASE=forked_vllm:latest \
#       --build-arg DS_DEB=deepstream-libs-minimal_9.0.0-1_amd64.deb \
#       -t vllm-deepstream:0.1 \
#       -f examples/online_serving/deepstream_video_stream/Dockerfile \
#       examples/online_serving/deepstream_video_stream
#
# Run:
#   docker run --rm -it --gpus all --ipc=host \
#       -e NVIDIA_DRIVER_CAPABILITIES=all \
#       -p 8000:8000 \
#       -v /path/to/videos:/data/video \
#       --name vllm-ds vllm-deepstream:0.1 \
#       --model <hf-model-id>

ARG VLLM_BASE=forked_vllm:latest
ARG DS_DEB=deepstream-libs-minimal_9.0.0-1_amd64.deb
FROM ${VLLM_BASE}
ARG DS_DEB

# 1. CUDA 13 runtime libs (NPP, cuBLAS, NVJPEG, cuFFT, cuSPARSE, ...)
#    + GStreamer 1.24 + Python bindings (PyGObject pulled transitively
#    by python3-gi).
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-13-0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev \
        python3-gi \
        python3-gst-1.0 \
        gir1.2-gstreamer-1.0 \
        gir1.2-gst-plugins-base-1.0 \
    && rm -rf /var/lib/apt/lists/*

# 2. DeepStream 9.0 SDK (minimal runtime — provides libnvbufsurface,
#    libnvbufsurftransform, DS GStreamer plugins). The .deb must be
#    downloaded by the user from NVIDIA's developer portal — it is not
#    bundled in this repo due to license terms.
COPY ${DS_DEB} /tmp/
RUN dpkg -i /tmp/${DS_DEB} \
    && rm /tmp/${DS_DEB}

# 3. Expose all GPU capabilities so NVDEC / NVENC / GL are accessible
#    from inside the container. Can be overridden at `docker run` time.
ENV NVIDIA_DRIVER_CAPABILITIES=all
