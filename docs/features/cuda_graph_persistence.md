# CUDA graph persistence with Foundry

The optional Foundry adapter archives CUDA graphs during SAVE and reconstructs
them in a fresh LOAD process. LOAD uses the archived decode graphs without
recapturing them.

The current adapter targets the GPU model runner and FULL decode CUDA graphs.
It disables Torch compilation in both SAVE and LOAD so that uncaptured forwards,
including prefill, use the same kernels. CUDA graph capture and replay remain
enabled. Use the same model, parallelism, software stack, and capture sizes for
SAVE and LOAD. Foundry requires a wheel matching the Torch minor version and
CUDA major version; see [Foundry installation](https://github.com/foundry-org/foundry#installation).

SAVE and LOAD both run vLLM's native memory profiling and KV-cache initialization.
Profiling can capture temporary graphs to estimate memory usage, separately from
the persisted decode graphs. After profiling, LOAD reuses the per-rank budget in
`kv_cache_budget.json` to preserve KV-cache geometry and avoid counting archived
KV backing memory twice. Foundry uses fixed KV allocations; explicitly enabling
extensible KV cache is rejected because it changes mappings after capture.

## Prepare graphs during Fast Start preload

`vllm preload --prepare-cudagraphs` starts the weight daemon, then runs one
temporary SAVE engine in a separate process using the daemon's zero-copy
weights. Graph capture needs the full engine's model runner, KV cache, attention
backend, and decode inputs. After profiling and saving graphs, the temporary
engine exits and releases its resources; the daemon keeps the weights for
subsequent serving engines.

Preload reports READY and `/health` returns 200 after SAVE exits successfully.
Health remains 503 during loading and preparation. A daemon or preparation
failure makes preload exit and clean up its children. Without
`--prepare-cudagraphs`, readiness requires only the weight daemons to be serving.

Add `weight_cache_size` to the SAVE TOML, for example:

```toml
mode = "save"
workspace_root = "/tmp/qwen-foundry"
base_addr = "0x600000000000"
region_size = "256GB"
scratch_space_size = "2GB"
weight_cache_size = "2GB"
```

`region_size` reserves virtual address space; it does not allocate 256 GB of
GPU memory. `scratch_space_size` sets the address boundary between startup
allocations and the reproducible engine allocation sequence.
`weight_cache_size` is a physically allocated pool for the daemon, and must fit
its weights and startup allocations. Preload checks that every model tensor
lies inside the pool. The daemon uses a virtual region below the engine region
so engines can import the pool without relocating it. LOAD validates
parameter/buffer addresses, shapes, strides, and dtypes before restoring graphs.
Keep `rank_N_ipc_weight_layout.json` with the archive's other metadata.

For Qwen3-0.6B, save that configuration as `/tmp/qwen-save.toml`, and create
`/tmp/qwen-load.toml` with the same contents except `mode = "load"`. Start preload:

```bash
vllm preload --model Qwen/Qwen3-0.6B \
    --prepare-cudagraphs --dtype bfloat16 \
    --weight-cache-socket-dir /tmp/qwen-wc \
    --weight-cache-health-host 127.0.0.1 --weight-cache-health-port 8001 \
    --max-model-len 128 --max-num-seqs 2 \
    --kv-cache-memory-bytes 268435456 --disable-flashinfer-autotune \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2],"use_breakable_cudagraph":false,"graph_extension_config_path":"/tmp/qwen-save.toml"}'
```

Wait for `/health` on port 8001 to return 200, then start a serving engine from
another terminal with the same environment and model configuration:

```bash
vllm serve Qwen/Qwen3-0.6B --dtype bfloat16 \
    --load-format ipc_cache \
    --model-loader-extra-config '{"socket_dir":"/tmp/qwen-wc","mode":"zero_copy","fallback":false}' \
    --max-model-len 128 --max-num-seqs 2 \
    --kv-cache-memory-bytes 268435456 --disable-flashinfer-autotune \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","decode_cudagraph_capture_request_sizes":[1,2],"use_breakable_cudagraph":false,"graph_extension_config_path":"/tmp/qwen-load.toml"}'
```

Keep the daemon running across engine restarts. Graph preparation currently
supports one GPU, FULL graphs, and a target model without speculative decoding.
Use the same capture and KV-cache configuration during preparation and serving.

## Regression and benchmark

Run the dedicated GPU test with Foundry installed:

```bash
VLLM_FOUNDRY_LOG_DIR=/tmp/foundry-persistence \
python -m pytest --optional -v -s \
    tests/v1/cudagraph/test_foundry_persistence.py \
    tests/model_executor/model_loader/test_weight_cache_cudagraph.py
```

The test starts Native, SAVE, and LOAD in independent
processes. It uses Qwen3-0.6B in BF16, checks the first LOAD request, requires
exact output equality for the Paris prompt repeated in batches of two,
and runs `vllm bench serve`
with 64 requests of 32 input and 32 output tokens at concurrency 2. The two
decode graph sizes are 1 and 2. The test rejects recapture fallback and requires
LOAD throughput to reach at least 85% of the matched native reference.

The IPC test additionally runs `preload --prepare-cudagraphs`, then compares a
native IPC engine with LOAD and another LOAD restart against that same daemon.
Both graph sizes must appear in actual FULL dispatch statistics. All three
engines run the fixed workload and official serving benchmark. Its logs and
`ipc-results.json` live in the `ipc-preload` subdirectory; preparation time is
reported separately from engine startup.

Logs and benchmark JSON files are retained in `VLLM_FOUNDRY_LOG_DIR`.
`results.json` records startup time, graph construction time, exact outputs,
and benchmark duration/token counts. Startup time measures server creation
through readiness; graph construction time measures Foundry's completed
background builds. This small-model regression reports startup savings without
enforcing a noisy startup-speed threshold. Measure production startup savings
with the intended model and capture-size set.

The Buildkite job in `.buildkite/test_areas/cuda.yaml` runs both tests on one
H200 MIG 35 GB partition with a Torch 2.14/CUDA 13 matching Foundry wheel.
It is an optional nightly job: ordinary PR builds require manual selection.
The pytest `--optional` flag separately enables the tests' optional markers.

## Tensor parallelism

Use the same communication backend for SAVE and LOAD. The tested TP2/TP4
configuration uses `VLLM_ALLREDUCE_USE_FLASHINFER=0` and
`--disable-custom-all-reduce`. FlashInfer fused all-reduce and custom all-reduce
currently require additional support for restoring their communication buffers.
Daemon graph preparation currently supports a single GPU.
