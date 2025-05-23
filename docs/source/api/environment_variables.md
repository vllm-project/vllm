# Environment Variables Documentation

This document lists all available environment variables and their configurations.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `str` | `not set` | Available options: "Debug", "Release", "RelWithDebInfo" |
| `CUDA_HOME` | `str` | `not set` | and lib directories. |
| `CUDA_VISIBLE_DEVICES` | `str` | `not set` | used to control the visible devices in the distributed setting |
| `DO_NOT_TRACK` | `bool` | `not set` | - |
| `FLASH_ATTN` | `bool` | `not set` | flag to control if vllm should use triton flash attention |
| `FLASHINFER` | `str` | `not set` | - "ROCM_FLASH": use ROCmFlashAttention |
| `FLASHMLA` | `str` | `not set` | - "FLASHINFER": use flashinfer |
| `INFO` | `str` | `not set` | - |
| `K_SCALE_CONSTANT` | `int` | `not set` | Divisor for dynamic key scale factor calculation for FP8 KV Cache |
| `LD_LIBRARY_PATH` | `str` | `not set` | when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl |
| `LOCAL_RANK` | `str` | `not set` | the GPU device id |
| `MAX_JOBS` | `str` | `not set` | By default this is the number of CPUs |
| `NVCC_THREADS` | `str` | `not set` | If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU. |
| `Q_SCALE_CONSTANT` | `int` | `not set` | Divisor for dynamic query scale factor calculation for FP8 KV Cache |
| `ROCM_FLASH` | `str` | `not set` | - "XFORMERS": use XFormers |
| `S3_ACCESS_KEY_ID` | `int` | `not set` | S3 access information, used for tensorizer to load model from S3 |
| `S3_ENDPOINT_URL` | `int` | `not set` | - |
| `S3_SECRET_ACCESS_KEY` | `int` | `not set` | - |
| `TORCH_SDPA` | `str` | `not set` | Available options: |
| `VERBOSE` | `bool` | `not set` | If set, vllm will print verbose logs during installation |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `bool` | `not set` | the max length derived from the model's config.json. |
| `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | `bool` | `not set` | If set, allow loading or unloading lora adapters in runtime, |
| `VLLM_API_KEY` | `str` | `not set` | API key for vLLM API server |
| `VLLM_ASSETS_CACHE` | `str` | `not set` | Path to the cache for storing downloaded assets |
| `VLLM_ATTENTION_BACKEND` | `str` | `not set` | - "FLASHMLA": use FlashMLA |
| `VLLM_AUDIO_FETCH_TIMEOUT` | `int` | `not set` | Default is 10 seconds |
| `VLLM_CACHE_ROOT` | `str` | `not set` | Defaults to `~/.cache/vllm` unless `XDG_CACHE_HOME` is set |
| `VLLM_CI_USE_S3` | `int` | `not set` | Whether to use S3 path for model loading in CI via RunAI Streamer |
| `VLLM_CONFIG_ROOT` | `str` | `not set` | files during **installation**. |
| `VLLM_CONFIGURE_LOGGING` | `str` | `not set` | or the configuration file specified by VLLM_LOGGING_CONFIG_PATH |
| `VLLM_CONTIGUOUS_PA` | `str` | `not set` | - |
| `VLLM_CPU_KVCACHE_SPACE` | `str` | `not set` | default is 4 GiB |
| `VLLM_CPU_MOE_PREPACK` | `bool` | `not set` | need to set this to "0" (False). |
| `VLLM_CPU_OMP_THREADS_BIND` | `str` | `not set` | "0,1,2", "0-31,33". CPU cores of different ranks are separated by '\|'. |
| `VLLM_CUDART_SO_PATH` | `str` | `not set` | In some system, find_loaded_library() may not work. So we allow users to |
| `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` | `str` | `not set` | Whether to log responses from API Server for debugging |
| `VLLM_DISABLE_COMPILE_CACHE` | `bool` | `not set` | - |
| `VLLM_DISABLED_KERNELS` | `str` | `not set` | (kernels: MacheteLinearKernel, MarlinLinearKernel, ExllamaLinearKernel) |
| `VLLM_DO_NOT_TRACK` | `bool` | `not set` | - |
| `VLLM_DP_MASTER_IP` | `str` | `not set` | IP address of the master node in the data parallel setting |
| `VLLM_DP_MASTER_PORT` | `str` | `not set` | Port of the master node in the data parallel setting |
| `VLLM_DP_RANK` | `str` | `not set` | Rank of the process in the data parallel setting |
| `VLLM_DP_RANK_LOCAL` | `str` | `not set` | Defaults to VLLM_DP_RANK when not set. |
| `VLLM_DP_SIZE` | `str` | `not set` | World size of the data parallel setting |
| `VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON` | `bool` | `not set` | i.e. moe_align_block_size_triton in fused_moe.py. |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `bool` | `not set` | If set, enable multiprocessing in LLM for the V1 code path. |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `int` | `not set` | timeout for each iteration in the engine |
| `VLLM_FLASH_ATTN_VERSION` | `str` | `not set` | when using the flash-attention backend. |
| `VLLM_FLASHINFER_FORCE_TENSOR_CORES` | `bool` | `not set` | otherwise will use heuristic based on model architecture. |
| `VLLM_FUSED_MOE_CHUNK_SIZE` | `int` | `not set` | - |
| `VLLM_IMAGE_FETCH_TIMEOUT` | `str` | `not set` | Default is 5 seconds |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `bool` | `not set` | AsyncLLMEngine errors and stops serving requests |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `str` | `not set` | - |
| `VLLM_LOGGING_CONFIG_PATH` | `str` | `not set` | If set to 1, vllm will configure logging using the default configuration |
| `VLLM_LOGGING_LEVEL` | `str` | `not set` | this is used for configuring the default logging level |
| `VLLM_LOGGING_PREFIX` | `str` | `not set` | if set, VLLM_LOGGING_PREFIX will be prepended to all log messages |
| `VLLM_LOGITS_PROCESSOR_THREADS` | `str` | `not set` | while not holding the python GIL, or both. |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `bool` | `not set` | Whether to use atomicAdd reduce in gptq/awq marlin kernel. |
| `VLLM_MLA_DISABLE` | `bool` | `not set` | If set, vLLM will disable the MLA attention optimizations. |
| `VLLM_MM_INPUT_CACHE_GIB` | `str` | `not set` | Default is 4 GiB |
| `VLLM_MODEL_REDIRECT_PATH` | `str` | `not set` | meta-llama/Llama-3.2-1B   /tmp/Llama-3.2-1B |
| `VLLM_NCCL_SO_PATH` | `str` | `not set` | by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234 |
| `VLLM_NO_DEPRECATION_WARNING` | `bool` | `not set` | If set, vllm will skip the deprecation warnings. |
| `VLLM_NO_USAGE_STATS` | `bool` | `not set` | - |
| `VLLM_PLUGINS` | `str` | `not set` | if this is set to an empty string, no plugins will be loaded |
| `VLLM_PP_LAYER_PARTITION` | `str` | `not set` | Pipeline stage partition strategy |
| `VLLM_PRECOMPILED_WHEEL_LOCATION` | `str` | `not set` | - |
| `VLLM_RAY_BUNDLE_INDICES` | `str` | `not set` | Format: comma-separated list of integers, e.g. "0,1,2,3" |
| `VLLM_RAY_PER_WORKER_GPUS` | `float` | `not set` | so that users can colocate other actors on the same GPUs as vLLM. |
| `VLLM_RINGBUFFER_WARNING_INTERVAL` | `int` | `not set` | Interval in seconds to log a warning message when the ring buffer is full |
| `VLLM_ROCM_CUSTOM_PAGED_ATTN` | `bool` | `not set` | custom paged attention kernel for MI3* cards |
| `VLLM_ROCM_FP8_PADDING` | `bool` | `not set` | Pad the fp8 weights to 256 bytes for ROCm |
| `VLLM_ROCM_MOE_PADDING` | `bool` | `not set` | Pad the weights for the moe kernel |
| `VLLM_ROCM_USE_AITER` | `bool` | `not set` | Acts as a parent switch to enable the rest of the other operations. |
| `VLLM_ROCM_USE_AITER_FP8_BLOCK_SCALED_MOE` | `bool` | `not set` | By default this is disabled. |
| `VLLM_ROCM_USE_AITER_LINEAR` | `bool` | `not set` | - scaled_mm (per-tensor / rowwise) |
| `VLLM_ROCM_USE_AITER_MOE` | `bool` | `not set` | By default is enabled. |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `bool` | `not set` | use aiter rms norm op if aiter ops are enabled. |
| `VLLM_RPC_TIMEOUT` | `int` | `not set` | server for simple data operations |
| `VLLM_SERVER_DEV_MODE` | `bool` | `not set` | e.g. `/reset_prefix_cache` |
| `VLLM_SKIP_P2P_CHECK` | `bool` | `not set` | and trust the driver's peer-to-peer capability report. |
| `VLLM_TARGET_DEVICE` | `str` | `not set` | rocm, neuron, cpu] |
| `VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE` | `str` | `not set` | Internal flag to enable Dynamo fullgraph capture |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `int` | `not set` | of the hardware support for FP8 compute. |
| `VLLM_TEST_FORCE_LOAD_FORMAT` | `str` | `not set` | - |
| `VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL` | `bool` | `not set` | This is used for testing the nightly wheel in python build. |
| `VLLM_TORCH_PROFILER_DIR` | `str` | `not set` | traces are saved. Note that it must be an absolute path. |
| `VLLM_TPU_BUCKET_PADDING_GAP` | `str` | `not set` | 8, we will run forward pass with [16, 24, 32, ...]. |
| `VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION` | `bool` | `not set` | If set, disables TPU-specific optimization for top-k & top-p sampling |
| `VLLM_TRACE_FUNCTION` | `str` | `not set` | Useful for debugging |
| `VLLM_USAGE_SOURCE` | `str` | `not set` | - |
| `VLLM_USAGE_STATS_SERVER` | `str` | `not set` | Usage stats collection |
| `VLLM_USE_DEEP_GEMM` | `bool` | `not set` | Allow use of DeepGemm kernels for fused moe ops. |
| `VLLM_USE_FLASHINFER_SAMPLER` | `str` | `not set` | If set, vllm will use flashinfer sampler |
| `VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH` | `bool` | `not set` | contiguous cache fetch will be used. |
| `VLLM_USE_MODELSCOPE` | `bool` | `not set` | note that the value is true or false, not numbers |
| `VLLM_USE_PRECOMPILED` | `bool` | `not set` | If set, vllm will use precompiled binaries (*.so) |
| `VLLM_USE_RAY_COMPILED_DAG` | `bool` | `not set` | control plane overhead. |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | `str` | `not set` | This flag is ignored if VLLM_USE_RAY_COMPILED_DAG is not set. |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | `bool` | `not set` | VLLM_USE_RAY_COMPILED_DAG is not set. |
| `VLLM_USE_RAY_SPMD_WORKER` | `bool` | `not set` | execution on all workers. |
| `VLLM_USE_TRITON_AWQ` | `bool` | `not set` | If set, vLLM will use Triton implementations of AWQ. |
| `VLLM_USE_TRITON_FLASH_ATTN` | `bool` | `not set` | flag to control if vllm should use triton flash attention |
| `VLLM_USE_V1` | `bool` | `not set` | If set, use the V1 code path. |
| `VLLM_V0_USE_OUTLINES_CACHE` | `bool` | `not set` | an environment with potentially malicious users. |
| `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE` | `int` | `not set` | TTFT and overall throughput. |
| `VLLM_VIDEO_FETCH_TIMEOUT` | `int` | `not set` | Default is 30 seconds |
| `VLLM_WORKER_MULTIPROC_METHOD` | `str` | `not set` | Both spawn and fork work |
| `VLLM_XGRAMMAR_CACHE_MB` | `str` | `not set` | It can be changed with this variable if needed for some reason. |
| `VLLM_XLA_CACHE_PATH` | `str` | `not set` | Only used for XLA devices such as TPUs. |
| `VLLM_XLA_CHECK_RECOMPILATION` | `bool` | `not set` | If set, assert on XLA recompilation after each execution step. |
| `V_SCALE_CONSTANT` | `int` | `not set` | Divisor for dynamic value scale factor calculation for FP8 KV Cache |
| `XDG_CACHE_HOME` | `str` | `not set` | Root directory for vLLM cache files |
| `XDG_CONFIG_HOME` | `str` | `not set` | Root directory for vLLM configuration files |
| `XFORMERS` | `str` | `not set` | - "FLASH_ATTN": use FlashAttention |
