# vllm FP8 Latency and Throughput benchmarks on AMD MI300x

Documentation for vLLM Inferencing on AMD Instinct platforms. 

## Overview

vLLM is a toolkit and library for large language model (LLM) inference and serving. It deploys the PagedAttention algorithm, which reduces memory consumption and increases throughput by leveraging dynamic key and value allocation in GPU memory. vLLM also incorporates many recent LLM acceleration and quantization algorithms, such as fp8 GeMM, fp8 KV cache, continuous batching, flash attention, hip graph, tensor parallel, GPTQ, AWQ, and token speculation. In addition, AMD implements high-performance custom kernels and modules in vLLM to enhance performance further.

This documentation shows some reference performance numbers and the steps to reproduce it for the popular Llama 3.1 series models from Meta with a pre-built AMD vLLM docker optimized for an AMD Instinct™ MI300X accelerator.

It includes:

   -  ROCm™ 6.2.2

   - vLLM 0.6.3

   - PyTorch 2.5dev (nightly)

## System configuration

The performance data below was measured on a server with MI300X accelerators with the following system configuration. The performance might vary with different system configurations.

| System  | MI300X with 8 GPUs  |
|---|---|
| BKC | 24.13 |
| ROCm | version ROCm 6.2.2 |
| amdgpu | build 2009461 |
| OS | Ubuntu 22.04 |
| Linux Kernel | 5.15.0-117-generic |
| BMCVersion | C2789.BC.0809.00 |
| BiosVersion | C2789.5.BS.1C11.AG.1 |
| CpldVersion | 02.02.00 |
| DCSCMCpldVersion | 02.02.00 |
| CX7 | FW 28.40.1000 |
| RAM | 1 TB |
| Host CPU | Intel(R) Xeon(R) Platinum 8480C |
| Cores | 224 |
| VRAM | 192 GB |
| Power cap | 750 W |
| SCLK/MCLK | 2100 Mhz / 1300 Mhz |

## Pull latest 

You can pull the image with `docker pull rocm/vllm-dev:main`

### What is New

   - MoE optimizations for Mixtral 8x22B, FP16
   - Llama 3.2 stability improvements
   - Llama 3.3 support
      
     
Gemms are tuned using PyTorch's Tunable Ops  feature (https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/tunable/README.md)
The  gemms are automatically enabled in the docker image, and all stored gemm configs are kept in /app/_gemm_csv in the same image

### Reproducing benchmark results

### Use pre-quantized models

To make it easier to run fp8 Llama 3.1 models on MI300X, the quantized checkpoints are available on AMD Huggingface space as follows 

- https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV 
- https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV 
- https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV
- https://huggingface.co/amd/grok-1-FP8-KV

Currently these models are private. Please join https://huggingface.co/amd to access. 

Download the model you want to run.  

These FP8 quantized checkpoints were generated with AMD’s Quark Quantizer. For more information about Quark, please refer to https://quark.docs.amd.com/latest/quark_example_torch_llm_gen.html

### Quantize your own models
This step is optional for you to use quantized models on your own. Take Llama 3.1 405B as an example. 

Download the Model View the Llama-3.1-405B model at https://huggingface.co/meta-llama/Llama-3.1-405B. Ensure that you have been granted access, and apply for it if you do not have access.

If you do not already have a HuggingFace token, open your user profile (https://huggingface.co/settings/profile), select "Access Tokens", press "+ Create New Token", and create a new Read token.

Install the `huggingface-cli` (if not already available on your system) and log in with the token you created earlier and download the model. The instructions in this document assume that the model will be stored under `/data/llama-3.1`. You can store the model in a different location, but then you'll need to update other commands accordingly. The model is quite large and will take some time to download; it is recommended to use tmux or screen to keep your session running without getting disconnected.

    sudo pip install -U "huggingface_hub[cli]"
    
    huggingface-cli login

Enter the token you created earlier; you do NOT need to save it as a git credential

Create the directory for Llama 3.1 models (if it doesn't already exist)

    sudo mkdir -p /data/llama-3.1
    
    sudo chmod -R a+w /data/llama-3.1

Download the model

    huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*" --local-dir /data/llama-3.1/Llama-3.1-405B-Instruct

Similarly, you can download Llama-3.1-70B and Llama-3.1-8B.

[Download and install Quark](https://quark.docs.amd.com/latest/install.html)

Run the quantization script in the example folder using the following command line:
export MODEL_DIR = [local model checkpoint folder] or meta-llama/Llama-3.1-405B-Instruct
#### single GPU
        python3 quantize_quark.py \ 
        --model_dir $MODEL_DIR \
        --output_dir Llama-3.1-405B-Instruct-FP8-KV \                           
        --quant_scheme w_fp8_a_fp8 \
        --kv_cache_dtype fp8 \
        --num_calib_data 128 \
        --model_export quark_safetensors \
        --no_weight_matrix_merge

#### If model size is too large for single GPU, please use multi GPU instead.
        python3 quantize_quark.py \ 
        --model_dir $MODEL_DIR \
        --output_dir Llama-3.1-405B-Instruct-FP8-KV \                           
        --quant_scheme w_fp8_a_fp8 \
        --kv_cache_dtype fp8 \
        --num_calib_data 128 \
        --model_export quark_safetensors \
        --no_weight_matrix_merge \
        --multi_gpu


### Launch AMD vLLM Docker

Download and launch the docker,

    docker run -it --rm --ipc=host --network=host --group-add render \
    --privileged --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -v /data/llama-3.1:/data/llm \
    rocm/vllm-dev:main

### Benchmark with AMD vLLM Docker

There are some system settings to be configured for optimum performance on MI300X. 

#### NUMA balancing setting

To optimize performance, disable automatic NUMA balancing. Otherwise, the GPU might hang until the periodic balancing is finalized. For further details, refer to the AMD Instinct MI300X system optimization guide.

Disable automatic NUMA balancing

    sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

Check if NUMA balancing is disabled (returns 0 if disabled)

    cat /proc/sys/kernel/numa_balancing
    0

#### LLM performance settings

Some environment variables enhance the performance of the vLLM kernels and PyTorch's tunableOp on the MI300X accelerator. The settings below are already preconfigured in the Docker image. See the AMD Instinct MI300X workload optimization guide for more information.

##### vLLM performance environment variables

    export VLLM_USE_TRITON_FLASH_ATTN=0
    export NCCL_MIN_NCHANNELS=112
    export VLLM_FP8_PADDING=1

You can set both PYTORCH_TUNABLEOP_ENABLED and PYTORCH_TUNABLEOP_TUNING to 1 to performance GEMM tuning for the 1st benchmark run. 
It will take some time to complete the tuning during the benchmark. After tuning, it will generate several csv files as the performance lookup database. For the subsequent benchmark runs, you can keep 

PYTORCH_TUNABLEOP_ENABLED as 1 and set 
PYTORCH_TUNABLEOP_TUNING to 0 to use the selected kernels. 

##### vLLM engine performance settings
vLLM provides a number of engine options which can be changed to improve performance. 
Refer https://docs.vllm.ai/en/stable/models/engine_args.html for the complete list of vLLM engine options.
Below is a list of options which are useful:
- **--max-model-len** : Maximum context length supported by the model instance. Can be set to a lower value than model configuration value to improve performance and gpu memory utilization.
- **--max-num-batched-tokens** : The maximum prefill size, i.e., how many prompt tokens can be packed together in a single prefill. Set to a higher value to improve prefill performance at the cost of higher gpu memory utilization. 65536 works well for LLama models.
- **--max-num-seqs** : The maximum decode batch size. Set to a value higher than the default(256) to improve decode throughput. Higher values will also utilize more KV cache memory. Too high values can cause KV cache space to run out which will lead to decode preemption. 512/1024 works well for LLama models.
- **--max-seq-len-to-capture** : Maximum sequence length for which Hip-graphs are captured and utilized. It's recommended to use Hip-graphs for the best decode performance. The default value of this parameter is 8K, which is lower than the large context lengths supported by recent models such as LLama. Set this parameter to max-model-len or maximum context length supported by the model for best performance.
- **--gpu-memory-utilization** : The ratio of GPU memory reserved by a vLLM instance. Default value is 0.9. It's recommended to set this to 0.99 to increase KV cache space.

Note: vLLM's server creation command line (vllm serve) supports the above parameters as command line arguments.
  
##### Online Gemm Tuning
Online Gemm tuning for small decode batch sizes can improve performance in some cases. e.g. Llama 70B upto Batch size 8

If you want to do limited online tuning use --enforce-eager and tune for particular batch sizes. See example below.

        export PYTORCH_TUNABLEOP_TUNING=1
        export PYTORCH_TUNABLEOP_ENABLED=1
        export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=100
        export PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=10
        export PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1024
        export PYTORCH_TUNABLEOP_FILENAME=/app/tuned_gemm_csv/bench_latency_tune_device_%d_full.csv

 Run the following command for BS=1/2/4/8:

        python /app/vllm/benchmarks/benchmark_latency.py \
        --model <path to Meta-Llama-3.1-70B-Instruct-FP8-KV> \
        --quantization fp8 \
        --kv-cache-dtype fp8 \
        --dtype float16 \
        --max-model-len 8192 \
        --num-iters-warmup 5 \
        --num-iters 5 \
        --tensor-parallel-size 8 \
        --input-len 4096 \
        --output-len 512 \
        --batch-size <BS> \
        --num-scheduler-steps 10 \
        --enforce-eager

The tuned file will be generated for device 0 only at /app/tuned_gemm_csv/bench_latency_tune_device_0_full.csv. Copy this file to /app/tuned_gemm_csv/bench_latency_tune_device_<D>_full.csv for D=1 through 7.

After the above steps, retain the environment variables set earlier, but set export PYTORCH_TUNABLEOP_TUNING=0 to disable online tuning, and use the tuned solutions.

##### Latency Benchmark

Benchmark Meta-Llama-3.1-405B FP8 with input 128 tokens, output 128 tokens, batch size 32 and tensor parallelism 8 as an example,

    python /app/vllm/benchmarks/benchmark_latency.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct-FP8-KV \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype half \
    --gpu-memory-utilization 0.99 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 8 \
    --batch size 32 \
    --input-len 128 \
    --output-len 128

If you want to run Meta-Llama-3.1-405B FP16, please run

    python /app/vllm/benchmarks/benchmark_latency.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.99 \
    --distributed-executor-backend mp \
    --tensor-parallel-size 8 \
    --batch size 32 \
    --input-len 128 \
    --output-len 128

You can change various input-len, output-len, batch size and run the benchmark as well. When output-len is 1, it measures prefill latency (TTFT). 
Decoding latency (TPOT) can be calculated based on the measured latency. 

For more information about the parameters, please run

    /app/vllm/benchmarks/benchmark_latency.py -h

##### Throughput Benchmark

Benchmark Meta-Llama-3.1-405B FP8 with input 128 tokens, output 128 tokens and tensor parallelism 8 as an example,

    python /app/vllm/benchmarks/benchmark_throughput.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct-FP8-KV \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype half \
    --gpu-memory-utilization 0.99 \
    --num-prompts 2000 \
    --distributed-executor-backend mp \
    --num-scheduler-steps 10 \
    --tensor-parallel-size 8 \
    --input-len 128 \
    --output-len 128 

If you want to run Meta-Llama-3.1-405B FP16, please run

    python /app/vllm/benchmarks/benchmark_throughput.py \
    --model /data/llm/Meta-Llama-3.1-405B-Instruct \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --num-prompts 2000 \
    --distributed-executor-backend mp \
    --num-scheduler-steps 10 \
    --tensor-parallel-size 8 \
    --input-len 128 \
    --output-len 128 \
    --swap-space 16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 65536 \
    --swap-space
    --max-model-len
    --gpu-memory-utilization 0.99

For fp8 quantized Llama3.18B/70B models:

   Recommend TP:1 for Llama3.1-8B, 8 for Llama3.1-70B
   Recommend NSCHED: 10 for Llama3.1-8B, 8 for Llama3.1-70B

You can change various input-len, output-len, num-prompts and run the benchmark as well.
Please note num-scheduler-step is a new feature added in vLLM 0.6.0. It can improve the decoding latency and throughput, however, it may increase the prefill latency.

For more information about the parameters, please run

    /app/vllm/benchmarks/benchmark_throughput.py -h

Tensor parallelism (TP) parameters depends on the model size. For Llama 3.1 70B and 8B model, TP 1 can be used as well for MI300X. In general, TP 8 and 1 is recommended to achieve the optimum performance. 

##### Online Server Benchmark
 
Make the following changes if required
 
/app/vllm/benchmarks/backend_request_func.py
 
line 242 + "ignore_eos": True,
 
/app/vllm/benchmarks/benchmark_serving.py
line 245 -         interval = np.random.exponential(1.0 / request_rate)
line 245 +         ## interval = np.random.exponential(1.0 / request_rate)
line 246 +         interval = 1.0 / request_rate
 
Benchmark Meta-Llama-3.1-70B with input 4096 tokens, output 512 tokens and tensor parallelism 8 as an example,
 
    vllm serve /data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.99 \
    --num_scheduler-steps 10
 
Change port (for example --port 8005) if port=8000 is currently being used by other processes.
 
run client in a separate terminal. Use port_id from previous step else port-id=8000.
 
    python /app/vllm/benchmarks/benchmark_serving.py \
    --port 8000 \
    --model /data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 512 \
    --request-rate 1 \
    --num-prompts 500 \
    --percentile-metrics ttft,tpot,itl,e2el
 
Once all prompts are processed, terminate the server gracefully (ctrl+c).
 
##### CPX mode
 
Currently only CPX-NPS1 mode is supported. So ONLY tp=1 is supported in CPX mode.
But multiple instances can be started simultaneously (if needed) in CPX-NPS1 mode.
 
Set GPUs in CPX mode
 
    rocm-smi --setcomputepartition cpx
 
Example of running Llama3.1-8B on 1 CPX-NPS1 GPU with input 4096 and output 512. As mentioned above, tp=1.

    HIP_VISIBLE_DEVICES=0 \
    python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --max-model-len 4608 \
    --num-scheduler-steps 10 \
    --num-prompts 100 \
    --model /data/llm/Meta-Llama-3.1-70B-Instruct-FP8-KV \
    --input-len 4096 \
    --output-len 512 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --output-json <path/to/output.json> \
    --quantization fp8 \
    --gpu-memory-utilization 0.99
 
Set GPU to SPX mode.

    rocm-smi --setcomputepartition spx

### Speculative Decoding

Speculative decoding is one of the key features in vLLM. It has been supported on MI300. Here below is an example of the performance benchmark w/wo speculative decoding for Llama 3.1 405B with Llama 3.1 8B as the draft model. 

Without Speculative Decoding - 

     python benchmark_latency.py --model /models/models--amd--Meta-Llama-3.1-405B-Instruct-FP8-KV/ --max-model-len 26720 -tp 8 --batch-size 1 --use-v2-block-manager --input-len 1024 --output-len 128

With Speculative Decoding - 

     python benchmark_latency.py --model /models/models--amd--Meta-Llama-3.1-405B-Instruct-FP8-KV/ --max-model-len 26720 -tp 8 --batch-size 1 --use-v2-block-manager --input-len 1024 --output-len 128 --speculative-model /models/models--amd--Meta-Llama-3.1-8B-Instruct-FP8-KV/ --num-speculative-tokens 5

You should see some performance improvement about the e2e latency. 

### MMLU_PRO_Biology Accuracy Eval
 
### fp16
vllm (pretrained=models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64
 
| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|
|-------|------:|--------------|-----:|-----------|---|-----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.8466|±  |0.0135|
 
### fp8
vllm (pretrained=models--meta-llama--Meta-Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,quantization=fp8,quantized_weights_path=/llama.safetensors,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32
 
| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value|   |Stderr|
|-------|------:|--------------|-----:|-----------|---|----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.848|±  |0.0134|


## Performance

### LLaMA2/3 *MLPerf* 70B

Please refer to the MLPerf instructions for recreating the MLPerf numbers.

## Version

### Release Notes
20240906a: Legacy quantization formats required `--quantization fp8_rocm` as a flag instead of `--quantization fp8`

Updated:

vLLM: https://github.com/ROCm/vllm/commit/2c60adc83981ada77a77b2adda78ef109d2e2e2b
### Docker Manifest

To reproduce the release docker:

```
git clone https://github.com/ROCm/vllm.git
cd vllm
git checkout 2c60adc83981ada77a77b2adda78ef109d2e2e2b
docker build -f Dockerfile.rocm -t <your_tag> --build-arg BUILD_HIPBLASLT=1 --build-arg USE_CYTHON=1 .
```
