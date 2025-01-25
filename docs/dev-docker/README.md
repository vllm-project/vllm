# vllm FP8 Latency and Throughput benchmarks on AMD MI300x

Documentation for vLLM Inferencing on AMD Instinct platforms.

## Overview

vLLM is a toolkit and library for large language model (LLM) inference and serving. It deploys the PagedAttention algorithm, which reduces memory consumption and increases throughput by leveraging dynamic key and value allocation in GPU memory. vLLM also incorporates many recent LLM acceleration and quantization algorithms, such as fp8 GeMM, fp8 KV cache, continuous batching, flash attention, hip graph, tensor parallel, GPTQ, AWQ, and token speculation. In addition, AMD implements high-performance custom kernels and modules in vLLM to enhance performance further.

This documentation includes information for running the popular Llama 3.1 series models from Meta using a pre-built AMD vLLM docker image optimized for an AMD Instinct™ MI300X or MI325X accelerator.

The pre-built image includes:

- ROCm™ 6.3
- vLLM 0.6.3
- PyTorch 2.6dev (nightly)

## Pull latest

You can pull the most recent validated docker image with `docker pull rocm/vllm-dev:main`

## What is New

- ROCm 6.3 support
- Potential bug with Tunable Ops not saving due to a PyTorch issue

## Preparation

### Obtaining access to models

The vllm-dev docker image should work with any model supported by vLLM.  When running with FP8, AMD has quantized models available for a variety of popular models, or you can quantize models yourself using Quark.  The vLLM benchmark scripts will download models automatically if needed, and then store them in a HuggingFace cache directory for reuse in future tests.  Alternatively you can choose to download the model to the cache (or to another directory on the system) in advance.

Many HuggingFace models, including Llama-3.1, have gated access.  You will need to an account at (https://huggingface.co), search for the model of interest, and request access to it if necessary.  You will also need to create a token for accessing these models from vLLM: open your user profile (https://huggingface.co/settings/profile), select "Access Tokens", press "+ Create New Token", and create a new Read token.

### System optimization

Before running performance tests you should ensure that the system is optimized according to the [ROCm Documentation](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html).  In particular, it is important to ensure that NUMA auto-balancing is disabled.

### Launch AMD vLLM Docker

Download and launch the docker.  The HF_TOKEN is required to be set (either here or after launching the container) if you want to allow vLLM to download gated models automatically; use your HuggingFace token in place of `<token>` in the command below:

```bash
docker run -it --rm --ipc=host --network=host --group-add render \
    --privileged --security-opt seccomp=unconfined \
    --cap-add=CAP_SYS_ADMIN --cap-add=SYS_PTRACE \
    --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
    -e HF_HOME=/data \
    -e HF_TOKEN=<token> \
    -v /data:/data \
    rocm/vllm-dev:main
```

Note: The instructions in this document use `/data` to store the models.  If you choose a different directory, you will also need to make that change to the host volume mount when launching the docker container.  For example, `-v /home/username/models:/data` in place of `-v /data:/data` would store the models in /home/username/models on the host.  Some models can be quite large; please ensure that you have sufficient disk space prior to downloading the model.  Since the model download may take a long time, you may wish to use `tmux` or `screen` to avoid getting disconnected.

### Downloading models with huggingface-cli

If you would like to download models directly (instead of allowing vLLM to download them automatically) you can use the huggingface-cli inside the running docker container.  Login using the token that you created earlier.  (Note, it is not necessary to save it as a git credential.)

```bash
huggingface-cli login
```

You can download a model to the huggingface-cache directory using a command similar to the following (substituting the name of the model you wish to download):

```bash
sudo mkdir -p /data/huggingface-cache
sudo chmod -R a+w /data/huggingface-cache
HF_HOME=/data/huggingface-cache huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*"
```

Alternatively, you may wish to download the model to a specific directory, e.g. so you can quantize the model with Quark:

```bash
sudo mkdir -p /data/llama-3.1
sudo chmod -R a+w /data/llama-3.1
huggingface-cli download meta-llama/Llama-3.1-405B-Instruct --exclude "original/*" --local-dir /data/llama-3.1/Llama-3.1-405B-Instruct
```

In the benchmark commands provided later in this document, replace the model name (e.g. `amd/Llama-3.1-405B-Instruct-FP8-KV`) with the path to the model (e.g. `/data/llama-3.1/Llama-3.1-405B-Instruct`)

### Use pre-quantized models

AMD has provided [FP8-quantized versions](https://huggingface.co/collections/amd/quark-quantized-ocp-fp8-models-66db7936d18fcbaf95d4405c) of several models in order to make them easier to run on MI300X / MI325X, including:

- <https://huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV>
- <https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV>

Some models may be private to those who are members of <https://huggingface.co/amd>.

These FP8 quantized checkpoints were generated with AMD’s Quark Quantizer. For more information about Quark, please refer to <https://quark.docs.amd.com/latest/quark_example_torch_llm_gen.html>

### Quantize your own models

This is an optional step if you would like to quantize your own model instead of using AMD's pre-quantized models.  These instructions use Llama-3.1-405B as an example, but the commands are similar for other models.

First download the model from <https://huggingface.co/meta-llama/Llama-3.1-405B> to the /data/llama-3.1 directory as described above.

[Download and install Quark](https://quark.docs.amd.com/latest/install.html)

Run the quantization script in the example folder using the following command line:

```bash
export MODEL_DIR = /data/llama-3.1/Llama-3.1-405B-Instruct
    python3 quantize_quark.py \
    --model_dir $MODEL_DIR \
    --output_dir Llama-3.1-405B-Instruct-FP8-KV \                           
    --quant_scheme w_fp8_a_fp8 \
    --kv_cache_dtype fp8 \
    --num_calib_data 128 \
    --model_export quark_safetensors \
    --no_weight_matrix_merge \
    --multi_gpu
```

Note: the `--multi_gpu` parameter can be omitted for small models that fit on a single GPU.

## Performance testing with AMD vLLM Docker

### Performance environment variables

Some environment variables enhance the performance of the vLLM kernels on the MI300X / MI325X accelerator. See the AMD Instinct MI300X workload optimization guide for more information.

```bash
export VLLM_USE_TRITON_FLASH_ATTN=0
export NCCL_MIN_NCHANNELS=112
```

### vLLM engine performance settings

vLLM provides a number of engine options which can be changed to improve performance.  Refer to the [vLLM Engine Args](https://docs.vllm.ai/en/stable/usage/engine_args.html) documentation for the complete list of vLLM engine options.

Below is a list of a few of the key vLLM engine arguments for performance; these can be passed to the vLLM benchmark scripts:
- **--max-model-len** : Maximum context length supported by the model instance. Can be set to a lower value than model configuration value to improve performance and gpu memory utilization.
- **--max-num-batched-tokens** : The maximum prefill size, i.e., how many prompt tokens can be packed together in a single prefill. Set to a higher value to improve prefill performance at the cost of higher gpu memory utilization. 65536 works well for LLama models.
- **--max-num-seqs** : The maximum decode batch size (default 256). Using larger values will allow more prompts to be processed concurrently, resulting in increased throughput (possibly at the expense of higher latency).  If the value is too large, there may not be enough GPU memory for the KV cache, resulting in requests getting preempted.  The optimal value will depend on the GPU memory, model size, and maximum context length.
- **--max-seq-len-to-capture** : Maximum sequence length for which Hip-graphs are captured and utilized. It's recommended to use Hip-graphs for the best decode performance. The default value of this parameter is 8K, which is lower than the large context lengths supported by recent models such as LLama. Set this parameter to max-model-len or maximum context length supported by the model for best performance.
- **--gpu-memory-utilization** : The ratio of GPU memory reserved by a vLLM instance. Default value is 0.9.  Increasing the value (potentially as high as 0.99) will increase the amount of memory available for KV cache.  When running in graph mode (i.e. not using `--enforce-eager`), it may be necessary to use a slightly smaller value of 0.92 - 0.95 to ensure adequate memory is available for the HIP graph.

### Latency Benchmark

vLLM's benchmark_latency.py script measures end-to-end latency for a specified model, input/output length, and batch size.

You can run latency tests for FP8 models with:

```bash
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
BS=1
IN=128
OUT=2048
TP=8

python3 /app/vllm/benchmarks/benchmark_latency.py \
    --distributed-executor-backend mp \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --batch-size $BS \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-iters-warmup 3 \
    --num-iters 5 \
    --output-json output.json
```

For FP16 models, remove `--quantization fp8 --kv-cache-dtype fp8`.

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value.  It is important, however, to ensure that the `--max-model-len` is at least as large as the IN + OUT token counts.

To estimate Time To First Token (TTFT) with the benchmark_latency.py script, set the OUT to 1 token.  It is also recommended to use `--enforce-eager` to get a more accurate measurement of the time that it actually takes to generate the first token.  (For a more comprehensive measurement of TTFT, use the Online Serving Benchmark.)

For additional information about the available parameters run:

```bash
/app/vllm/benchmarks/benchmark_latency.py -h
```

### Throughput Benchmark

vLLM's benchmark_throughput.py script measures offline throughput.  It can either use an input dataset or random prompts with fixed input/output lengths.

You can run latency tests for FP8 models with:

```bash
MODEL=amd/Llama-3.1-405B-Instruct-FP8-KV
BS=1
IN=128
OUT=2048
TP=8
PROMPTS=1000
MAX_NUM_SEQS=2000

python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --distributed-executor-backend mp \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --num-scheduler-steps 10 \
    --enable-chunked-prefill False \
    --model $MODEL \
    --max-model-len 8192 \
    --max-num-batched-tokens 131072 \
    --max-seq-len-to-capture 131072 \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-prompts $PROMPTS \
    --max-num-seqs $MAX_NUM_SEQS \
    --output-json output.json
```

For FP16 models, remove `--quantization fp8 --kv-cache-dtype fp8`.

When measuring models with long context lengths, performance may improve by setting `--max-model-len` to a smaller value (8192 in this example).  It is important, however, to ensure that the `--max-model-len` is at least as large as the IN + OUT token counts.

It is important to tune vLLM’s --max-num-seqs value to an appropriate value depending on the model and input/output lengths.  Larger values will allow vLLM to leverage more of the GPU memory for KV Cache and process more prompts concurrently.  But if the value is too large, the KV cache will reach its capacity and vLLM will have to cancel and re-process some prompts.  Suggested values for various models and configurations are listed below.

For models that fit on a single GPU, it is usually best to run with `--tensor-parallel-size 1`.  Requests can be distributed across multiple copies of vLLM running on different GPUs.  This will be more efficient than running a single copy of the model with `--tensor-parallel-size 8`.  (Note: the benchmark_throughput.py script does not include direct support for using multiple copies of vLLM)

For optimal performance, the PROMPTS value should be a multiple of the MAX_NUM_SEQS value -- for example, if MAX_NUM_SEQS=1500 then the PROMPTS value could be 1500, 3000, etc.  If PROMPTS is smaller than MAX_NUM_SEQS then there won’t be enough prompts for vLLM to maximize concurrency.

For additional information about the available parameters run:

```bash
/app/vllm/benchmarks/benchmark_throughput.py -h
```

### Online Serving Benchmark

Benchmark Llama-3.1-70B with input 4096 tokens, output 512 tokens and tensor parallelism 8 as an example,

```bash
vllm serve amd/Llama-3.1-70B-Instruct-FP8-KV \
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
```

Change port (for example --port 8005) if port=8000 is currently being used by other processes.

Run client in a separate terminal. Use port_id from previous step else port-id=8000.

```bash
python /app/vllm/benchmarks/benchmark_serving.py \
    --port 8000 \
    --model amd/Llama-3.1-70B-Instruct-FP8-KV \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 512 \
    --request-rate 1 \
    --ignore-eos \
    --num-prompts 500 \
    --percentile-metrics ttft,tpot,itl,e2el
```

Once all prompts are processed, terminate the server gracefully (ctrl+c).

### CPX mode

Currently only CPX-NPS1 mode is supported. So ONLY tp=1 is supported in CPX mode.
But multiple instances can be started simultaneously (if needed) in CPX-NPS1 mode.

Set GPUs in CPX mode with:

```bash
rocm-smi --setcomputepartition cpx
```

Example of running Llama3.1-8B on 1 CPX-NPS1 GPU with input 4096 and output 512. As mentioned above, tp=1.

```bash
HIP_VISIBLE_DEVICES=0 \
python3 /app/vllm/benchmarks/benchmark_throughput.py \
    --max-model-len 4608 \
    --num-scheduler-steps 10 \
    --num-prompts 100 \
    --model amd/Llama-3.1-70B-Instruct-FP8-KV \
    --input-len 4096 \
    --output-len 512 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --output-json <path/to/output.json> \
    --quantization fp8 \
    --gpu-memory-utilization 0.99
```

Set GPU to SPX mode.

```bash
rocm-smi --setcomputepartition spx
```

### Speculative Decoding

Speculative decoding is one of the key features in vLLM. It has been supported on MI300. Here below is an example of the performance benchmark w/wo speculative decoding for Llama 3.1 405B with Llama 3.1 8B as the draft model.

Without Speculative Decoding -

```bash
python benchmark_latency.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --max-model-len 26720 -tp 8 --batch-size 1 --use-v2-block-manager --input-len 1024 --output-len 128
```

With Speculative Decoding -

```bash
python benchmark_latency.py --model amd/Llama-3.1-405B-Instruct-FP8-KV --max-model-len 26720 -tp 8 --batch-size 1 --use-v2-block-manager --input-len 1024 --output-len 128 --speculative-model amd/Llama-3.1-8B-Instruct-FP8-KV --num-speculative-tokens 5
```

You should see some performance improvement about the e2e latency.

## MMLU_PRO_Biology Accuracy Eval

### fp16

vllm (pretrained=models--meta-llama--Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64

| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value |   |Stderr|
|-------|------:|--------------|-----:|-----------|---|-----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.8466|±  |0.0135|

### fp8

vllm (pretrained=models--meta-llama--Llama-3.1-405B-Instruct/snapshots/069992c75aed59df00ec06c17177e76c63296a26,dtype=float16,quantization=fp8,quantized_weights_path=/llama.safetensors,tensor_parallel_size=8), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32

| Tasks |Version|    Filter    |n-shot|  Metric   |   |Value|   |Stderr|
|-------|------:|--------------|-----:|-----------|---|----:|---|-----:|
|biology|      0|custom-extract|     5|exact_match|↑  |0.848|±  |0.0134|

## Performance

### *MLPerf* Llama-2-70B

Please refer to the [Benchmarking Machine Learning using ROCm and AMD GPUs: Reproducing Our MLPerf Inference Submission — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inf-4-1/README.html) for information on reproducing MLPerf 4.1 Inference results.  Note that due to changes in vLLM, it is not possible to use these instructions with the current rocm/vllm-dev docker image.

## Docker Manifest

To reproduce the release docker:

```bash
    git clone https://github.com/ROCm/vllm.git
    cd vllm
    git checkout 2c60adc83981ada77a77b2adda78ef109d2e2e2b
    docker build -f Dockerfile.rocm -t <your_tag> --build-arg BUILD_HIPBLASLT=1 --build-arg USE_CYTHON=1 .
```
