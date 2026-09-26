# Profiling vLLM

!!! warning
    Profiling is only intended for vLLM developers and maintainers to understand the proportion of time spent in different parts of the codebase. **vLLM end-users should never turn on profiling** as it will significantly slow down the inference.

!!! tip "Choosing a profiler"
    - Use **Nsight Systems** for low-overhead, performance-critical profiling.
    - Use **PyTorch Profiler** for medium-overhead profiling with richer debugging information (e.g., stack traces, memory, shapes). Note that enabling these features adds overhead and is not recommended for benchmarking.

## Profile with PyTorch Profiler

We support tracing vLLM workers using different profilers. You can enable profiling by setting the `--profiler-config` flag when launching the server.

!!! note
    The `--profiler-config` flag is available in vLLM v0.13.0 and later. If you are using an earlier version, please upgrade to use this feature.

To use the `torch.profiler` module, set the `profiler` entry to `'torch'` and `torch_profiler_dir` to the directory where you want to save the traces. Additionally, you can control the profiling content by specifying the following additional arguments in the config:

- `torch_profiler_record_shapes` to enable recording Tensor Shapes, off by default
- `torch_profiler_with_memory` to record memory, off by default
- `torch_profiler_with_stack` to enable recording stack information, on by default
- `torch_profiler_with_flops` to enable recording FLOPs, off by default
- `torch_profiler_use_gzip` to control gzip-compressing profiling files, on by default
- `torch_profiler_dump_cuda_time_total` to control dumping and printing the aggregated CUDA self time table, on by default
- `torch_profiler_activities` to select worker activities. Defaults are
  platform-specific: `["CPU"]` on CPU, `["CPU", "CUDA"]` on NVIDIA GPUs, and
  `["CPU", "XPU"]` on Intel GPUs. Selecting `["CUDA"]` omits CPU annotations
  and the AsyncLLM CPU trace to reduce profiling overhead and trace size.

When using `vllm bench serve`, you can enable profiling by passing the `--profile` flag.

Traces can be visualized using <https://ui.perfetto.dev/>.

!!! tip
    You can directly call bench module without installing vLLM using `python -m vllm.entrypoints.cli.main bench`.

!!! tip
    Only send a few requests through vLLM when profiling, as the traces can get quite large. Also, no need to untar the traces, they can be viewed directly.

!!! tip
    To stop the profiler - it flushes out all the profile trace files to the directory. This takes time, for example for about 100 requests worth of data for a llama 70b, it takes about 10 minutes to flush out on a H100.
    The engine client waits for this flush to complete without timing out, so simply allow the stop call to run to completion.

### Example commands and usage

#### Offline Inference

Refer to [examples/features/profiling/simple_profiling_offline.py](../../examples/features/profiling/simple_profiling_offline.py) for an example.

#### OpenAI Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
```

vllm bench command:

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

Or use http request:

```shell
# We need first call /start_profile api to start profile.
$ curl -X POST http://localhost:8000/start_profile

# Call model generate.
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [
                        {
                                "role": "user",
                                "content": "San Francisco is a"
                        }
                ]
    }'

# After need call /stop_profile api to stop profile.
$ curl -X POST http://localhost:8000/stop_profile
```

## Profile with Triton Proton

[Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton)
is Triton's GPU profiler. It can collect a low-overhead aggregate tree or a
Chrome trace and works through the same vLLM profiling controls as the PyTorch
and CUDA profilers. Proton currently supports NVIDIA GPUs through CUPTI and
supports CUDA graph attribution.

Start a server with a local output directory and graph attribution:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --profiler-config '{
        "profiler": "proton",
        "proton_profiler_dir": "./proton_profile",
        "proton_output_format": "hatchet",
        "proton_hook": "triton",
        "proton_graph_attribution": true
    }'
```

Then use `/start_profile` and `/stop_profile` as shown above, or pass
`--profile` to a vLLM benchmark. Each worker uses a topology- and
rank-qualified output name, such as
`proton_dp0_pp0_tp0_dcp0_ep0_rank0_pid1234_0123456789abcdef0123456789abcdef_run0.hatchet`,
so distributed workers, restarted servers, and repeated profiling runs do not
overwrite one another. A `profile_prefix` is included when supplied. Each
profile is written by `/stop_profile` and is ready to inspect immediately.

### Benchmark with Proton

For offline benchmarks, pass both `--profile` and `--profiler-config` to the
benchmark itself. For example, profile one batch after warmup:

```bash
vllm bench latency --model facebook/opt-125m \
    --input-len 32 --output-len 16 --batch-size 4 \
    --num-iters-warmup 2 --profile \
    --profiler-config '{"profiler": "proton", "proton_profiler_dir": "./proton_latency", "proton_graph_attribution": true}'
```

To profile a throughput workload:

```bash
vllm bench throughput --model facebook/opt-125m \
    --dataset-name random --random-input-len 32 --random-output-len 16 \
    --num-prompts 16 --num-warmups 2 --profile \
    --profiler-config '{"profiler": "proton", "proton_profiler_dir": "./proton_throughput", "proton_graph_attribution": true}'
```

The throughput command also supports `--async-engine` with the same profiling
configuration. Warmup requests are excluded from the profiling interval. Offline
benchmarks stop the profiler on completion or a request error so collected
activity can be flushed and exported. Profiling adds overhead, so use a separate
run without `--profile` for performance measurements.

For `bench serve`, configure Proton on the **server**, as shown above, and only
pass `--profile` to the client:

```bash
vllm bench serve --model meta-llama/Llama-3.1-8B-Instruct \
    --backend vllm --endpoint /v1/completions \
    --dataset-name random --random-input-len 32 --random-output-len 16 \
    --num-prompts 16 --num-warmups 2 --profile
```

Use the model name served by your server. Profiles are written to the server's
`proton_profiler_dir`, not the benchmark client's filesystem. Each benchmark run
starts and stops a profiling interval; repeated runs produce separate files.

### Proton configuration

The Proton-specific options are:

- `proton_context`: `shadow` (default) or `python`
- `proton_data`: `tree` (default) or `trace`
- `proton_backend`: `cupti` or automatic
- `proton_mode`: an optional mode string; `periodic_flushing` enables vLLM-managed
  periodic output
- `proton_flush_interval`: worker steps per periodic output part (default: 100)
- `proton_hook`: `triton` to record Triton launch metadata, or unset
- `proton_output_format`: `hatchet`, `hatchet_msgpack`, `chrome_trace`, or unset
- `proton_graph_attribution`: observe CUDA graph capture for replay attribution;
  disabled by default and requires `proton_data: "tree"`

`hatchet` and `hatchet_msgpack` require `proton_data: "tree"`, while
`chrome_trace` requires `proton_data: "trace"`.

Automatic backend selection is recommended. vLLM currently supports Proton's
`cupti` backend on NVIDIA GPUs. ROCm support is not yet available. vLLM does not
expose Proton's experimental instrumentation backend because current upstream
can produce profiles without timing metrics. When `proton_graph_attribution` is
enabled, Proton observes vLLM's CUDA graph capture with the configured profiling
session active, then deactivates that same session until profiling starts. This
lets later profiles attribute replayed kernels without retaining model-startup
activity. Backend-specific modes can be selected with `proton_mode`;
`pcsampling` synchronizes the CUDA context and therefore requires
`--enforce-eager`. When CUDA graphs are enabled (including encoder graphs),
Proton requires `proton_graph_attribution: true` to collect replayed kernels.
For Chrome traces, disable CUDA graphs with `--enforce-eager`.

CUDA graph-attributed profiles support repeated `start_profile`/`stop_profile`
runs. Without periodic flushing, each stop writes one tree-data phase, preserving the
graph-aware session. Without `proton_graph_attribution`, each `stop_profile`
instead finalizes and writes an independent Proton session.

CUDA graph attribution requires Triton 3.7 or newer. It uses the phase data API
to discard graph-capture activity and separate profiling runs. The `hatchet_msgpack`
output format and `periodic_flushing` mode also require Triton 3.7 or newer.
Ordinary Proton profiling remains available with Triton 3.6.

Graph attribution retains capture metadata for the worker lifetime. With eager
execution or no graphs to capture, each profiling run uses an independent session.

### Periodic output

For long benchmarks, enable `periodic_flushing` to export completed activity
during the run and release its memory:

```bash
vllm bench throughput --model facebook/opt-125m \
    --dataset-name random --random-input-len 32 --random-output-len 128 \
    --num-prompts 16 --num-warmups 2 --profile \
    --profiler-config '{
        "profiler": "proton",
        "proton_profiler_dir": "./proton_periodic",
        "proton_graph_attribution": true,
        "proton_mode": "periodic_flushing",
        "proton_flush_interval": 32
    }'
```

This works with latency, synchronous/asynchronous throughput, and serving
benchmarks. For serving, put the profiler configuration on the server and pass
`--profile` to `bench serve`. Eager execution is also supported: use
`--enforce-eager` and omit `proton_graph_attribution`.

The interval counts worker steps, not requests or seconds. At the next step
after the interval, vLLM advances the phase while collection continues. Worker
steps poll Proton's phase-completion status and queue completed phases for a
background exporter. GPU buffers may complete later than the phase boundary,
especially for short workloads, so the interval does not set a deadline for
files to appear. Stopping synchronously flushes the remaining activity and waits
for all queued exports, even for a run shorter than the interval. Delayed
profiling starts counting once collection begins.
Idle worker steps can produce parts without GPU activity. A part containing only
eager execution, such as prefill, has no graph-capture attribution.

Each run produces files named `proton_..._run0.part_0.hatchet`,
`proton_..._run0.part_1.hatchet`, and so on. The part counter resets for each
run. Files become visible only after they have been fully written. Capture
activity is discarded, while graph metadata survives phase boundaries and
subsequent runs.

Periodic output supports `tree` data with `hatchet` or `hatchet_msgpack`.
Select the format with `proton_output_format`, or the equivalent mode string
`periodic_flushing:format=hatchet_msgpack`; specifying conflicting formats is
an error. Chrome traces are not supported by this mode.

vLLM owns periodic phase advancement, export, and cleanup, and does not enable
Proton's native periodic exporter. Periodic boundaries do not synchronize the
GPU or wait for disk writes. A single background thread serializes, writes, and
clears completed phases; profiling and serialization still add overhead.
If flushing or exporting fails, vLLM logs the failure and retains the phase and
its original output path for retry. Pending data can increase memory use if GPU
buffers complete slowly or the output device cannot keep up. A failed stop
retains the session so a later profiling run or shutdown can retry the export.

Inspect tree profiles with:

```bash
proton-viewer -m time/ns \
    proton_profile/proton_dp0_pp0_tp0_dcp0_ep0_rank0_pid1234_0123456789abcdef0123456789abcdef_run0.hatchet
```

Chrome traces (`proton_data: "trace"`) can be opened in
<https://ui.perfetto.dev/>. Proton is imported lazily, so selecting another
profiler does not require a Proton-capable Triton installation.

## Profile with NVIDIA Nsight Systems

Nsight systems is an advanced tool that exposes more profiling details, such as register and shared memory usage, annotated code regions and low-level CUDA APIs and events.

[Install nsight-systems](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html) using your package manager.
The following block is an example for Ubuntu.

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

!!! tip
    When profiling with `nsys`, it is advisable to set the environment variable `VLLM_WORKER_MULTIPROC_METHOD=spawn`. The default is to use the `fork` method instead of `spawn`. More information on the topic can be found in the [Nsight Systems release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues).

The Nsight Systems profiler can be launched with `nsys profile ...`, with a few recommended flags for vLLM: `--trace-fork-before-exec=true --cuda-graph-trace=node`.

### Example commands and usage

#### Offline Inference

For basic usage, you can just append the profiling command before any existing script you would run for offline inference.

The following is an example using the `vllm bench latency` script:

```bash
nsys profile  \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 16 \
    --input-len 512 \
    --output-len 8
```

#### OpenAI Server

To profile the server, you will want to prepend your `vllm serve` command with `nsys profile` just like for offline inference, but you will need to specify a few other arguments to enable dynamic capture similarly to the Torch Profiler:

```bash
# server
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config.profiler cuda

# client
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

With `--profile`, vLLM will capture a profile for each run of `vllm bench serve`. Once the server is killed, the profiles will all be saved.

#### Analysis

You can view these profiles either as summaries in the CLI, using `nsys stats [profile-file]`, or in the GUI by installing Nsight [locally following the directions here](https://developer.nvidia.com/nsight-systems/get-started).

??? console "CLI example"

    ```bash
    nsys stats report1.nsys-rep
    ...
    ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

    Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)                                                  Name
    --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
        46.3   10,327,352,338     17,505    589,965.9    144,383.0    27,040  3,126,460    944,263.8  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_of…
        14.8    3,305,114,764      5,152    641,520.7    293,408.0   287,296  2,822,716    867,124.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_of…
        12.1    2,692,284,876     14,280    188,535.4     83,904.0    19,328  2,862,237    497,999.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off…
        9.5    2,116,600,578     33,920     62,399.8     21,504.0    15,326  2,532,285    290,954.1  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_…
        5.0    1,119,749,165     18,912     59,208.4      9,056.0     6,784  2,578,366    271,581.7  void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel<c10::BFloat16>, (bool)1>(T1 *, cons…
        4.1      916,662,515     21,312     43,011.6     19,776.0     8,928  2,586,205    199,790.1  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMa…
        2.6      587,283,113     37,824     15,526.7      3,008.0     2,719  2,517,756    139,091.1  std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, void>::type vllm::fused_add_rms_norm_kern…
        1.9      418,362,605     18,912     22,121.5      3,871.0     3,328  2,523,870    175,248.2  void vllm::rotary_embedding_kernel<c10::BFloat16, (bool)1>(const long *, T1 *, T1 *, const T1 *, in…
        0.7      167,083,069     18,880      8,849.7      2,240.0     1,471  2,499,996    101,436.1  void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0…
    ...
    ```

GUI example:

<img width="1799" alt="Screenshot 2025-03-05 at 11 48 42 AM" src="https://github.com/user-attachments/assets/c7cff1ae-6d6f-477d-a342-bd13c4fc424c" />

## Continuous Profiling

There is a [GitHub CI workflow](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-profiling.yml) in the PyTorch infrastructure repository that provides continuous profiling for different models on vLLM. This automated profiling helps track performance characteristics over time and across different model configurations.

### How It Works

The workflow currently runs weekly profiling sessions for selected models, generating detailed performance traces that can be analyzed using different tools to identify performance regressions or optimization opportunities. But, it can be triggered manually as well, using the Github Action tool.

### Adding New Models

To extend the continuous profiling to additional models, you can modify the [profiling-tests.json](https://github.com/pytorch/pytorch-integration-testing/blob/main/vllm-profiling/cuda/profiling-tests.json) configuration file in the PyTorch integration testing repository. Simply add your model specifications to this file to include them in the automated profiling runs.

### Viewing Profiling Results

The profiling traces generated by the continuous profiling workflow are publicly available on the [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm). Look for the **Profiling traces** table to access and download the traces for different models and runs.

## Profiling vLLM Python Code

The Python standard library includes
[cProfile](https://docs.python.org/3/library/profile.html) for profiling Python
code.

### Example usage - function call

If a filename is specified, the profile will be saved to that file. If no
filename is specified, profile data can be printed to stdout.

```python
import cProfile


def expensive_function():
    # some expensive code
    pass


profiler = cProfile.Profile()
profiler.runcall(expensive_function)
profiler.dump_stats("expensive_function.prof")
```

### Example usage - context manager style

```python
import cProfile


def another_function():
    # more expensive code
    pass


profiler = cProfile.Profile()
profiler.enable()
try:
    another_function()
finally:
    profiler.disable()
    profiler.dump_stats("another_function.prof")
```

### Analyzing Profile Results

There are multiple tools available that can help analyze the profile results.
One example is [snakeviz](https://jiffyclub.github.io/snakeviz/).

```bash
pip install snakeviz
snakeviz expensive_function.prof
```

### Analyzing Garbage Collection Costs

Leverage VLLM_GC_DEBUG environment variable to debug GC costs.

- VLLM_GC_DEBUG=1: enable GC debugger with gc.collect elapsed times
- VLLM_GC_DEBUG='{"top_objects":5}': enable GC debugger to log top 5
  collected objects for each gc.collect
