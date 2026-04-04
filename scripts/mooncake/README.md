# Mooncake KV Store for vLLM

## Setup

### 1. Install Mooncake

We need to use our own version of Mooncake for GB200:

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

## Benchmarking CPU KV Cache Offloading

### 1. Start the Master Server

```bash
bash scripts/mooncake/start_mooncake_master.sh --bg
```

This starts the master in the background with default ports (RPC: 50051, HTTP metadata: 8080). Logs are written to `scripts/mooncake/mooncake_master.log`. See the script header for environment variables to customize ports and eviction settings.

### 2. Configure Mooncake

Edit `scripts/mooncake/mooncake_config.json` to match your setup:

```json
{
  "metadata_server": "http://127.0.0.1:8080/metadata",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "80GB",
  "protocol": "tcp",
  "device_name": ""
}
```

- I haven't tried `protocol` and `device_name` yet, but "rdma" is worth a try, although not sure its benefits on a single node, we will need it anyway.
- Adjust `global_segment_size` and `local_buffer_size` based on available memory.
    - global_segment_size: Memory contributed to the distributed pool
    - local_buffer_size: Private buffer for this node's own operations
    - For now we use a single node so they don't differ much
- Note: the benchmark script automatically updates `global_segment_size` and `local_buffer_size` to match `KV_OFFLOAD_GIB`.

### 3. Run the Benchmark

By default, the script runs two settings (`baseline,mooncake`) and builds a comparison table at the end. Backends with missing prerequisites are auto-skipped.
`baseline` is no offloading.

```bash
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json \
bash scripts/mooncake/benchmark_cpu_offloading.sh [MODEL] [INPUT_LEN] [OUTPUT_LEN] [NUM_PROMPTS]
```

Defaults: `meta-llama/Llama-3.1-8B`, input 8192 tokens, output 1024 tokens, 200 prompts.

To run only specific backends, set `BACKENDS` (comma-separated):

```bash
# Only mooncake (no baseline)
BACKENDS=mooncake \
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json \
bash scripts/mooncake/benchmark_cpu_offloading.sh

# mooncake vs simple vs baseline
BACKENDS=baseline,mooncake,simple \
MOONCAKE_CONFIG_PATH=scripts/mooncake/mooncake_config.json \
bash scripts/mooncake/benchmark_cpu_offloading.sh
```

Results are saved to `./bench_results/` and a comparison table is printed at the end.

### 4. Stop the Master

```bash
bash scripts/mooncake/start_mooncake_master.sh --stop
```
