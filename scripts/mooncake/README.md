# Mooncake KV Store for vLLM

## Setup

### 1. Install Mooncake

We need to use our own version of Mooncake for GB200.

#### Option A: Install from pre-built wheel (recommended)

NOTE: this wheel is pre-compiled only for Grace-Blackwell and Grace-Hopper (ARM64) platforms. For AMD64 (including Intel platforms), users still need to go with Option B and manual compile for now. We will add a pre-compiled wheel later.

```bash
# Make sure your vllm venv is activated
uv pip install scripts/mooncake/mooncake_transfer_engine-0.3.10.post1-cp312-cp312-manylinux_2_39_aarch64.whl
```

#### Option B: Build from source

```shell
git clone https://github.com/ivanium/Mooncake
cd Mooncake
git checkout yifan/dev
# NOTE: Install necessary dependencies
./scripts/dev_compile.sh
# NOTE: Please check to ensure compile succeeded before installing
# NOTE: Please ensure to activate vllm venv before installing
./scripts/dev_install.sh
```

### 2. Verify Installation

Start the Mooncake master server:

```bash
bash scripts/mooncake/start_mooncake_master.sh
```

In a separate terminal, run the example script to confirm everything works:

```bash
python scripts/mooncake/mooncake_example.py
```

You should see `Hello, Mooncake Store!` printed. Stop the master with Ctrl-C.

## Running Benchmarks

Mooncake benchmarks in this branch use an external-owner topology:

- each vLLM rank runs as a requester-only embedded real client
- one standalone owner real client per node owns CPU memory and disk offload
- the requester scripts do not launch or configure the owner process

### 1. Start the Master Server

```bash
bash scripts/mooncake/start_mooncake_master.sh
# With disk offloading support:
bash scripts/mooncake/start_mooncake_master.sh --enable-offload
# Start master and run it in background
bash scripts/mooncake/start_mooncake_master.sh --bg
# With disk offloading support and running in background:
bash scripts/mooncake/start_mooncake_master.sh --enable-offload --bg
```

This starts the master in the background. Logs go to `scripts/mooncake/mooncake_master.log`.

Default ports:

- RPC: 50051
- HTTP metadata: 8080
- Prometheus metrics: 9003

See the script header for environment variables (`MC_RPC_PORT`, `MC_HTTP_PORT`, etc.) to customize ports and eviction settings. Mooncake master is launched cluster-wise, so we may get port conflict if someone else has already launched it. Typically we don't need to change this, and multiple users should be able to share the same master. One can freely change ports here, but also remember to update the `mooncake_config.json` for the client (vLLM) below.

### 2. Configure Mooncake

Edit `scripts/mooncake/mooncake_config.json`:

```json
{
  "metadata_server": "http://127.0.0.1:8080/metadata",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "600GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": ""
}
```

- `protocol`: Use `"rdma"` for best performance. `"tcp"` works as a fallback but performs poorly.
- `global_segment_size` and `local_buffer_size` belong to the owner process, not requester ranks.
- The requester helper no longer edits `global_segment_size` or enables requester offload ownership.
- The requester still uses `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` as a batching hint.
- Start one owner per node before launching vLLM requester ranks.
- If you want disk offload, the owner process must be launched with disk support separately.

Owner launch contract:

- fixed owner RPC port: `50052`
- start the owner with `--host=<node-ip>:<segment-port>`
- start the owner with `--port=50052`
- use the owner’s advertised segment string as the locality target for requester ranks
- the owner’s registered segment name is what requester discovery compares against

Owner helper:

```bash
# Memory only
bash scripts/mooncake/start_mooncake_owner.sh --cpu-mem-size 80

# Memory + disk offload
bash scripts/mooncake/start_mooncake_owner.sh --cpu-mem-size 80 --disk-size 400

# Background mode
bash scripts/mooncake/start_mooncake_owner.sh --cpu-mem-size 80 --disk-size 400 --bg
```

This helper:

- launches `mooncake_client` with the stable owner contract `--host=<node-ip>:50053 --port=50052`
- sets owner-side disk offload env vars when `--disk-size` is given
- keeps requester setup separate from owner setup

### 3. Environment Setup (setup_vllm_env.sh)

Before running benchmarks with the Mooncake backend, source `setup_vllm_env.sh` to configure requester-side environment variables. The benchmark scripts do this automatically, but you can also use it directly:

```bash
# Requester-only setup with an 80 GB local-buffer hint
source scripts/mooncake/setup_vllm_env.sh --cpu-mem-size 80

# Requester-only setup, disk offload is owned externally
source scripts/mooncake/setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400
```

This script:

- Sets `MOONCAKE_CONFIG_PATH` if it is not already set
- Enables `MC_TCP_ENABLE_CONNECTION_POOL`
- Sets `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`
- Leaves Mooncake ownership and disk offload to the external owner service
- Accepts `--disk-size` for compatibility, but does not configure requester ownership from it

### 4a. Single-Turn Benchmark (benchmark_cpu_offloading.sh)

Measures raw KV offloading overhead using random (unique) prompts so no prefix cache is ever hit.

```bash
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json \
bash scripts/mooncake/benchmark_cpu_offloading.sh [MODEL] [INPUT_LEN] [OUTPUT_LEN] [NUM_PROMPTS]
```

Defaults: `meta-llama/Llama-3.1-8B-Instruct`, input 8192, output 1024, 200 prompts.

Supported backends (comma-separated via `BACKENDS`):

- `baseline` - No offloading
- `native` - Built-in vLLM KV offloading (`--kv-offloading-backend native`)
- `simple` - Simple native offload (`VLLM_USE_SIMPLE_KV_OFFLOAD=1` + native backend)
- `mooncake` - MooncakeStoreConnector via `--kv-transfer-config`

Mooncake benchmark expectations:

- start the owner service before launching the benchmark
- source `setup_vllm_env.sh` only for requester-side environment setup
- use `kv_connector_extra_config` for connector-specific overrides
- set `preferred_segment` explicitly in `kv_connector_extra_config` when you want to pin locality
- if you omit `preferred_segment`, requester startup uses best-effort admin discovery through `/get_all_segments`

Environment variables:

- `CPU_OFFLOAD_GIB` - CPU offload buffer in GiB (default: 80)
- `DISK_OFFLOAD_GIB` - Disk offload quota in GiB (default: 400)
- `REQUEST_RATE` - Requests/s (default: 1)
- `PORT` - Server port (default: 8192)
- `RESULT_DIR` - Output directory (default: `./bench_results`)
- `BACKENDS` - Comma-separated backends (default: `baseline,mooncake`)

Example:

```bash
# Only mooncake with disk offloading
BACKENDS=mooncake \
CPU_OFFLOAD_GIB=80 \
DISK_OFFLOAD_GIB=400 \
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json \
bash scripts/mooncake/benchmark_cpu_offloading.sh
```

### 4b. Multi-Turn Benchmark (benchmark_multi_turn.sh)

Measures multi-turn chat performance with prefix sharing (global + conversation prefixes).

```bash
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json \
bash scripts/mooncake/benchmark_multi_turn.sh [MODEL] [INPUT_LEN] [OUTPUT_LEN] [NUM_PROMPTS]
```

Defaults: `meta-llama/Llama-3.1-8B-Instruct`, input 70000, output 200, 200 prompts.

Additional backends:

- `mooncake-mem` - MooncakeStoreConnector with CPU memory only (no disk)

Additional environment variables:

- `MULTI_TURN_NUM_TURNS` - Turns per conversation (default: 3)
- `MULTI_TURN_CONCURRENCY` - Concurrent conversations (default: 16)
- `MULTI_TURN_DELAY_MS` - Delay between turns in ms (default: 500)
- `GLOBAL_PREFIX_RATIO` - Fraction of input as global prefix (default: 0.1)
- `CONV_PREFIX_RATIO` - Fraction of input as conversation prefix (default: 0.8)

For Mooncake runs, the owner must already be running when the benchmark starts. The benchmark scripts only set requester-side config and do not launch owner processes.

### 5. Compare Results (compare_results.py)

Both benchmark scripts automatically run the comparison at the end. To re-run manually:

```bash
# Single-turn results
python scripts/mooncake/compare_results.py ./bench_results

# Multi-turn results
python scripts/mooncake/compare_results.py ./bench_results --prefix mt_
```

### 6. Stop the Master

```bash
bash scripts/mooncake/start_mooncake_master.sh --stop
```

## Notes

### Cross-DP External Prefix Cache Hits

When running `MooncakeStoreConnector` with DP, set a fixed
`PYTHONHASHSEED` before launching vLLM if you want cross-DP external
prefix cache lookup to work reliably, for example:

```bash
PYTHONHASHSEED=0 BACKENDS=mooncake-mem \
bash scripts/mooncake/benchmark_multi_turn.sh
```

If `PYTHONHASHSEED` is unset, each engine process initializes the root
prefix-cache hash seed randomly. Identical prompts can then produce
different block-hash chains on different DP ranks, which means a rank
may query Mooncake external cache but still report
`external_prefix_cache_hits=0` even when another DP rank already stored
the same prefix.
