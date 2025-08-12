# Profiling vLLM

:::{warning}
Profiling is only intended for vLLM developers and maintainers to understand the proportion of time spent in different parts of the codebase. **vLLM end-users should never turn on profiling** as it will significantly slow down the inference.
:::

## Profile with PyTorch Profiler

We support tracing vLLM workers using the `torch.profiler` module. You can enable tracing by setting the `VLLM_TORCH_PROFILER_DIR` environment variable to the directory where you want to save the traces: `VLLM_TORCH_PROFILER_DIR=/mnt/traces/`

The OpenAI server also needs to be started with the `VLLM_TORCH_PROFILER_DIR` environment variable set.

When using `benchmarks/benchmark_serving.py`, you can enable profiling by passing the `--profile` flag.

Traces can be visualized using <https://ui.perfetto.dev/>.

:::{tip}
Only send a few requests through vLLM when profiling, as the traces can get quite large. Also, no need to untar the traces, they can be viewed directly.
:::

:::{tip}
To stop the profiler - it flushes out all the profile trace files to the directory. This takes time, for example for about 100 requests worth of data for a llama 70b, it takes about 10 minutes to flush out on a H100.
Set the env variable VLLM_RPC_TIMEOUT to a big number before you start the server. Say something like 30 minutes.
`export VLLM_RPC_TIMEOUT=1800000`
:::

### Example commands and usage

#### Offline Inference

Refer to <gh-file:examples/offline_inference/simple_profiling.py> for an example.

#### OpenAI Server

```bash
VLLM_TORCH_PROFILER_DIR=./vllm_profile python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B
```

benchmark_serving.py:

```bash
python benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Meta-Llama-3-70B --dataset-name sharegpt --dataset-path sharegpt.json --profile --num-prompts 2
```

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

### Example commands and usage

#### Offline Inference

For basic usage, you can just append `nsys profile -o report.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node` before any existing script you would run for offline inference.

The following is an example using the `benchmarks/benchmark_latency.py` script:

```bash
nsys profile -o report.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node python benchmarks/benchmark_latency.py --model meta-llama/Llama-3.1-8B-Instruct --num-iters-warmup 5 --num-iters 1 --batch-size 16 --input-len 512 --output-len 8
```

#### OpenAI Server

To profile the server, you will want to prepend your `vllm serve` command with `nsys profile` just like for offline inference, however you must specify `--delay XX --duration YY` parameters according to the needs of your benchmark. After the duration time has been used up, the server will be killed.

```bash
# server
nsys profile -o report.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node --delay 30 --duration 60 vllm serve meta-llama/Llama-3.1-8B-Instruct

# client
python benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --num-prompts 1 --dataset-name random --random-input 1024 --random-output 512
```

In practice, you should set the `--duration` argument to a large value. Whenever you want the server to stop profiling, run:

```
nsys sessions list
```

to get the session id in the form of `profile-XXXXX`, then run:

```
nsys stop --session=profile-XXXXX
```

to manually kill the profiler and generate your `nsys-rep` report.

#### Analysis

You can view these profiles either as summaries in the CLI, using `nsys stats [profile-file]`, or in the GUI by installing Nsight [locally following the directions here](https://developer.nvidia.com/nsight-systems/get-started).

CLI example:

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
