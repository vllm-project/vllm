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

- each vLLM rank is requester-only
- one standalone owner per node owns CPU memory and optional disk offload
- requester helpers do not launch or configure the owner automatically

### 1. Start the Master Server

```bash
bash scripts/mooncake/start_mooncake_master.sh
bash scripts/mooncake/start_mooncake_master.sh --bg
bash scripts/mooncake/start_mooncake_master.sh --bg --env-file /shared/logs/mooncake_master.env
```

Use `--env-file` for managed launches. It writes:

- `MOONCAKE_MASTER`
- `MOONCAKE_TE_META_DATA_SERVER`

Default ports:

- RPC: 50051
- HTTP metadata: 8080
- Prometheus metrics: 9003

Set `MC_MASTER_HOST` or `--host` if the detected advertise address is not the right control-plane IP for your cluster.

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

- `protocol`: use `"rdma"` for performance, `"tcp"` only as fallback.
- Keep this file generic. The managed launcher exports `MOONCAKE_DEVICE` as a comma-separated per-GPU worker list, and the owner helper auto-detects its local RNIC CSV when `MC_OWNER_DEVICE` is unset. If you are not using the managed launcher, set `MOONCAKE_DEVICE` explicitly or leave it unset to let Mooncake auto-select.
- `global_segment_size` and disk ownership belong to the owner process, not requester ranks.

Generate the node-local RNIC recommendations with:

```bash
bash scripts/mooncake/recommend_mooncake_rnic_config.sh
```

This prints:

- a ready-to-paste `MOONCAKE_DEVICE=...` line for workers
- a ready-to-paste `MC_OWNER_DEVICE=...` line for the owner

The helper exits non-zero on ambiguous topology, but still prints the best-effort recommendation for review. `run_vllm_with_mooncake_owner.sh` uses the same shell-side mapping logic for managed launches.

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

- auto-detects active owner RNICs when `MC_OWNER_DEVICE` is unset
- validates `MC_OWNER_DEVICE` when explicitly set
- launches `mooncake_client` with `--host=<node-ip>:50053 --port=50052`
- configures owner-side disk offload when `--disk-size` is given

### 3. Environment Setup (setup_vllm_env.sh)

Before running benchmarks with the Mooncake backend, source `setup_vllm_env.sh` to configure requester-side environment variables:

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
- Leaves ownership and disk offload to the external owner service

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

For Mooncake runs, start the owner before launching the benchmark. The benchmark scripts only set requester-side config.

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
