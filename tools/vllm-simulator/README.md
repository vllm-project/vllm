# vllm_simulator
## Background
As large language models (LLMs) are rapidly deployed at scale for inference services, inference performance directly impacts user experience, service cost, and resource efficiency. Key metrics such as Time to First Token (TTFT), Time Per Output Token (TPOT), and system throughput are highly dependent on the complex interplay among model architecture, hardware platforms (e.g., A100/H100/B300), inference engines (e.g., vLLM, SGLang, TensorRT-LLM), and runtime configurations (e.g., quantization, batching, and parallelism strategies).

Traditional end-to-end stress testing on real GPU clusters is expensive and time-consuming, making it impractical to efficiently explore the vast space of configuration combinations. To address this, we propose **vllm_simulator**, a high-fidelity CPU-based simulation system. vllm_simulator enables fast, low-cost, and high-fidelity prediction of key performance metrics across different models, target hardware, and configurations by replaying real-world inference workload traces collected from production or representative scenarios, thereby accelerating the design and optimization of inference systems.

---
## Introduction
vllm_simulator is a simulation tool that hijacks a local vLLM installation. It launches a mock inference service—an OpenAI-compatible API server driven by the hijacked vLLM engine—that accepts user requests via real-world trace replay or synthetic load generation using standard benchmarking scripts. vllm_simulator outputs performance metrics identical to those produced by `vllm bench serve`. See **Quick Start** for usage examples.

---
## Installation
```bash
cd tools/vllm-simulator
pip install .
```

> **Note**: vllm_simulator hijacks the `vllm` package at runtime through class-level hooks, so a working vLLM installation must be present in the same Python environment.

---
## Quick Start
### Step 1: Mock Simulation
This example runs inference simulation using a synthetic random workload. You may also replay real-world traces (e.g., `--dataset-name timed_trace`).

#### Launch the Simulation Server
Run the following command from the project root directory (the folder containing this `README.md`):
```bash
python3 -m vllm_simulator.simulation.vllm.launch_server \
  --model "Qwen/Qwen3-32B-FP8" \
  --sim-config-path test/assets/config_vllm.json
```

> **Notes**:
> - Use `--sim-config-path` to specify the simulation configuration file, which is equivalent to the system environment variable `VLLM_SIMULATOR_CONFIG_PATH`.
> - The server accepts standard vLLM serving arguments (`--max-model-len`, `--gpu-memory-utilization`, etc.). Do **not** configure TP/EP parallelism here; see the `scheduler` section of the config file below.
> - The simulator uses `/start_profile` and `/stop_profile` as benchmark round markers (dump + reset simulation stats). vLLM only registers those routes when a profiler is configured, so the EngineArgs hook internally defaults the profiler to `"cuda"`—no `--profiler-config` needed. It stays inert: the hijacked `EngineCore.profile` never reaches the real profiler.
> - `CUDA_VISIBLE_DEVICES` defaults to `""` inside the simulator so vLLM's platform detection stays consistent with the simulated environment.
> - The provided [config file](test/assets/config_vllm.json) is for testing. Adjust the accelerator and other parameters to match your actual deployment scenario for higher fidelity.

#### Run the Simulation Benchmark
The benchmark client is a hijacked `vllm bench serve` and accepts the same arguments:
```bash
python3 -m vllm_simulator.simulation.bench_serving \
    --backend openai \
    --base-url http://127.0.0.1:8000 \
    --model "Qwen/Qwen3-32B-FP8" \
    --dataset-name random \
    --request-rate 4 \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts 10 \
    --save-result
```

The client does not pace requests in real time. Instead, every request carries a logical arrival timestamp (`simulation.created_time`), and the server replays the traffic pattern against its (virtual) simulation clock. Add `--save-result` (optionally with `--result-dir` / `--result-filename`) to persist the result JSON with the simulation metrics.

You have now completed an inference simulation using framework interception.

#### Example Output
```bash
============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Request rate configured (RPS):           4.00
Benchmark duration (s):                  1.15
Total input tokens:                      10240
Total generated tokens:                  10240
Request throughput (req/s):              8.69
Output token throughput (tok/s):         8901.74
...
=============== Simulation Metrics ===============
Completed requests:                      10
Simulated duration (s):                  2.53
Output token throughput (tok/s):         4053.36
Mean TTFT (ms):                          128.59
Mean TPOT (ms):                          12.14
Mean E2EL (ms):                          3431.84
==================================================
```

> The "Serving Benchmark Result" block is computed from client-side timings and is **not meaningful** in simulation mode. The authoritative numbers are the "Simulation Metrics" block and the saved result JSON, which are aggregated from the per-request stats dumped by the server (via `calc_metrics`); fields such as `mean_ttft_ms`, `p99_tpot_ms`, throughputs, and KV-cache hit ratios in the result JSON are substituted accordingly.

---
## Usage
### Mock Simulation (Inference Simulation)
vllm_simulator uses **dynamic interception** to hijack the execution flow of vLLM, bypassing actual LLM computation:

- **Platform / Worker**: stub CUDA operations and KV-cache allocation so the engine runs on CPU while preserving GPU deployment semantics. The model runner is replaced by a stand-in for Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1` is forced; the V1 runner is not modelled).
- **EngineCore** (`vllm.v1.engine.core.EngineCore`): holds requests in a future queue and dispatches them by `simulation.created_time`; records per-request queueing and prefix-cache hit stats.
- **Executor** (`UniProcExecutor.execute_model`): accounts the predicted GPU span of each step (AIConfigurator prediction + sampler/logprobs compensation) and advances the simulation clock.
- **EngineCore.profile**: `/start_profile` and `/stop_profile` double as benchmark round markers—each call dumps `request.jsonl` / `iteration.jsonl` to the output directory and resets the stats.

Key environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `VLLM_SIMULATOR_CONFIG_PATH` | Simulation config JSON (see below) | — (required) |
| `VLLM_SIMULATOR_OUTPUT_MODE` | `NON_BLOCKING` (virtual-clock replay) or `BLOCKING` (engine sleeps the predicted GPU span in real time) | `NON_BLOCKING` |
| `VLLM_SIMULATOR_OUTPUT_DIR` | Where the server dumps `request.jsonl` / `iteration.jsonl`; the benchmark client reads them from here | `/tmp/vllm_simulator/output/` |
| `VLLM_SIMULATOR_MAX_DECODE_STEPS` | Force every request's output length | unset |

The benchmark client and the server must share the same `VLLM_SIMULATOR_OUTPUT_DIR`.

- For supported launch options, run:
  ```bash
  python -m vllm_simulator.simulation.vllm.launch_server --help
  ```
- For benchmark options, run:
  ```bash
  python -m vllm_simulator.simulation.bench_serving --help
  ```

---
### Data Collection (Collector)
The collector runs a **real** vLLM server on GPUs and records per-step and per-request traces, used to calibrate the simulator, train `ml` predictor models, and build `replay` predictor tables. It accepts the same arguments as the `vllm` CLI:
```bash
VLLM_SIMULATOR_COLLECTOR_OUTPUT_DIR=/path/to/output \
python3 -m vllm_simulator.collector.launch_server serve "Qwen/Qwen3-8B" --port 8000

curl -X POST http://127.0.0.1:8000/start_profile
# ... send benchmark traffic ...
curl -X POST http://127.0.0.1:8000/stop_profile
```

Only traffic between `/start_profile` and `/stop_profile` is recorded. `/stop_profile` writes `rank0.schedule_batch.jsonl` (per-step batch composition and latency) and `requests.jsonl` (per-request timing, KV hits and token ids) synchronously, overwriting the previous window. Under TP only rank 0 is collected.

| Variable | Meaning | Default |
| --- | --- | --- |
| `VLLM_SIMULATOR_COLLECTOR_OUTPUT_DIR` | Export directory | current working directory |
| `VLLM_SIMULATOR_COLLECTOR_EXPORT_TOKEN_IDS` | Export `input_ids` / `output_ids` | `1` |
| `VLLM_SIMULATOR_COLLECTOR_EXPORT_GZIP` | Write `.jsonl.gz` instead of `.jsonl` | `1` |

---
### Configuration File Format (`VLLM_SIMULATOR_CONFIG_PATH`)
The config file is a JSON file with three main sections:

- **`platform`**: Hardware settings
  - `accelerator`: GPU model (e.g., `"h100_sxm"`, `"b300_sxm"`) and `hbm_capacity_gb`

- **`predictor`**: Time prediction module
  - `name`: predictor type (`"aiconfigurator"`, `"ml"`, `"replay"` or `"linear"`)
  - See the **TimePredictor** section below for details

- **`scheduler`**: Parallelism and backend metadata
  > ⚠️ **Note**: Multi-GPU parallelism (TP/EP) should not be configured at the framework level during server launch. Instead, specify `tp_size` and `ep_size` here; the predictor will simulate parallel execution overhead accordingly.
  - `backend_name` / `backend_version`: the simulated inference engine and version, used to locate the matching AIConfigurator performance database (e.g., `"vllm"` / `"0.19.0"`).

**Example Config**:
```json
{
    "platform": {
        "accelerator": {
            "name": "b300_sxm",
            "hbm_capacity_gb": 80000
        }
    },
    "predictor": {
        "name": "aiconfigurator",
        "database_mode": "SOL"
    },
    "scheduler": {
        "tp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "backend_name": "vllm",
        "backend_version": "0.24.0"
    }
}
```

---
## TimePredictor
### AIConfigurator
- Project: <https://github.com/ai-dynamo/aiconfigurator>
- Parameters:
  - `database_path`: (optional) path to custom operator profiling data
  - `database_mode`: (optional) performance database mode, e.g. `"SOL"` (speed-of-light) or `"SILICON"` (measured data); defaults to `"SILICON"`
  - `platform.accelerator.name`: device/system name; it should be one of those defined internally in AIConfigurator
  - `prefill_scale_factor` / `decode_scale_factor`: optional calibration factors to adjust predicted latency

**Example**:
```json
{
  "name": "aiconfigurator",
  "database_path": "path/to/aiconfigurator/data",
  "database_mode": "SOL",
  "prefill_scale_factor": 1.02040816,
  "decode_scale_factor": 1.01010101
}
```

### Linear
Predicts every non-empty step as `base_ms + per_token_ms * scheduled_tokens`, bypassing any performance model (debugging). KV-cache length is not modelled. Set `per_token_ms` to 0 for a constant per-step latency.
- Parameters:
  - `base_ms`: (optional) fixed per-step latency in milliseconds; defaults to 0
  - `per_token_ms`: (optional) latency per scheduled token in milliseconds; defaults to 0

**Example**:
```json
{
  "name": "linear",
  "base_ms": 10,
  "per_token_ms": 0.02
}
```
