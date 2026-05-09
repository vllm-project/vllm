# Mooncake KV Store for vLLM

## Setup

### 1. Install Mooncake

We need to use our own version of Mooncake for GB200. Build from source against the `yifan/dev` branch of [`ivanium/Mooncake`](https://github.com/ivanium/Mooncake).

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

Build flags (set in `dev_compile.sh`):
- `-DUSE_CUDA=ON`
- `-DWITH_NVIDIA_PEERMEM=OFF` (required on GB200 — peermem kernel module won't load)
- `-DUSE_MNNVL=ON`

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

### Recommended Validation Flow

Validate Mooncake from the bottom up before interpreting vLLM or SGLang results:

1. Start the Mooncake master and confirm it stays in `serving` state.
2. Watch active RDMA/RoCE NIC bandwidth while traffic is running.
3. Run the standalone Mooncake smoke test and confirm put/get succeeds.
4. Run a standalone Mooncake put/get benchmark and record operations per second plus approximate bandwidth.
5. Only after Mooncake itself looks healthy, run the vLLM or SGLang integration benchmark.

### RDMA/RoCE Bandwidth Checks

Mooncake deployment validation should include real-time network checks, not only process liveness. First confirm the mapping between RDMA devices and Linux netdevs:

```bash
rdma link show
```

Then monitor NIC traffic with a short refresh interval, for example:

```bash
sar -n DEV 2 | grep -E 'IFACE|gpu[0-9]+rdma|roce|mlx'
```

For a more dedicated local helper, read the counters under `/sys/class/infiniband/*/ports/*/counters/` every 2 seconds, calculate the delta, and report per-RNIC bandwidth. The goal is to quickly answer how much RX/TX bandwidth each RDMA NIC is carrying.

### Standalone Mooncake Benchmark Expectations

Before connecting Mooncake to vLLM or SGLang, run Mooncake by itself. The goal is not final serving throughput; the goal is to verify that Mooncake Store, RDMA, and metadata discovery are healthy.

At minimum, record:

- Successful `put` operations per second.
- Successful `get` operations per second.
- Value or object size.
- Approximate bandwidth, using `ops_per_second * value_size`.
- Failure codes and failure ratio, especially metadata misses, RDMA timeouts, or resource-allocation failures.

Once standalone Mooncake is stable, framework-level benchmark failures are much easier to interpret.

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
- Adjust `global_segment_size` and `local_buffer_size` based on available memory.
    - global_segment_size: Memory contributed to the distributed pool
    - local_buffer_size: Private buffer for this node's own operations
    - For now we use a single node so they don't differ much
    - **These sizes are per GPU (per rank), not for the entire node.** For example, with 4 GPUs and `global_segment_size` set to `80GB`, each rank allocates 80 GB of CPU memory, totaling 320 GB across the node.
- Note: the benchmark script automatically updates `global_segment_size` and `local_buffer_size` to match `CPU_OFFLOAD_GIB`. The `CPU_OFFLOAD_GIB` and `DISK_OFFLOAD_GIB` env vars in the benchmark scripts are also per GPU (per rank).

### 3. Environment Setup (setup_vllm_env.sh)

Before running benchmarks with the mooncake backend, source `setup_vllm_env.sh` to configure all necessary environment variables. The benchmark scripts do this automatically, but you can also use it directly:

```bash
# CPU offloading only (80 GB)
source scripts/mooncake/setup_vllm_env.sh --cpu-mem-size 80

# CPU + disk offloading (80 GB CPU, 400 GB disk)
source scripts/mooncake/setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400

# Custom disk path
source scripts/mooncake/setup_vllm_env.sh --cpu-mem-size 80 --disk-size 400 --disk-path /mnt/data/mooncake_offload
```

This script:

- Sets `MOONCAKE_CONFIG_PATH` and updates `global_segment_size` in the config JSON
- Enables `MC_TCP_ENABLE_CONNECTION_POOL`
- When `--disk-size` is given, sets all disk offloading env vars (`MOONCAKE_ENABLE_OFFLOAD`, `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`, eviction policy, io_uring, etc.)

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
