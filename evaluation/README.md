# Guideline

## Set Enviroment

1. Docker image:

   ```shell
   rocm/ali-private:ubuntu22.04_rocm7.0.1.42_vllm_5b842c2_aiter_6b586ae_torch2.8.0_20250917
   ```

2. Upgrade PyBind:

   ```shell
   pip install --upgrade pybind11
   ```

3. Install Aiter dev/perf branch:

   ```shell
   pip uninstall aiter
   git clone -b dev/perf git@github.com:ROCm/aiter.git
   cd aiter
   git submodule sync && git submodule update --init --recursive
   python3 setup.py install
   ```

4. Install Rocm/vLLM dev/perf branch:

   ```shell
   pip uninstall vllm
   git clone -b dev/perf git@github.com:ROCm/vllm.git
   cd vllm
   python3 -m pip install -r requirements/common.txt
   export PYTORCH_ROCM_ARCH="gfx942"
   python3 setup.py develop
   ```

## Launch server

1. deepseek-r1 PTPC FP8

- download weight: <https://huggingface.co/EmbeddedLLM/deepseek-r1-FP8-Dynamic>

    ```shell
    huggingface-cli download EmbeddedLLM/deepseek-r1-FP8-Dynamic --local-dir EmbeddedLLM/deepseek-r1-FP8-Dynamic
    ```

- launch server:

    ```shell
    bash launch_deepseekr1_ptpc_fp8.sh
    ```

    We currently use pure tp8 since it gives better performance than TP8 + EP8, which is subject to change as optimization continues.

    The example command:

    ```shell
    export VLLM_USE_V1=1
    export SAFETENSORS_FAST_GPU=1
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_USE_TRITON_FLASH_ATTN=0
    export NCCL_DEBUG=WARN
    export VLLM_RPC_TIMEOUT=1800000
    export VLLM_ROCM_USE_AITER_MHA=0
    export VLLM_ROCM_USE_TRITON_ROPE=1

    # for profiling
    export VLLM_TORCH_PROFILER_DIR="deepseek_in3k_out1k"
    export VLLM_TORCH_PROFILER_WITH_STACK=1
    export VLLM_TORCH_PROFILER_RECORD_SHAPES=1

    model_path="/path-to-model/deepseek-r1-FP8-Dynamic/"
    vllm serve $model_path \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --gpu_memory_utilization 0.9 \
    --block-size 1
    ```

## Curl request

1. curl a single request to quickly check the functionality

   ```shell
    curl -X POST "http://localhost:8000/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "The capital of China", "temperature": 0, "top_p": 1, "top_k": 0, "repetition_penalty": 1.0, "presence_penalty": 0, "frequency_penalty": 0, "stream": false, "ignore_eos": false, "n": 1, "seed": 123 
    }'
   ```

   The result should be:

   ```shell
   {"id":"cmpl-026a60769119489587e46d571b6ebb6a","object":"text_completion","created":1760272161,"model":"/mnt/raid0/zhangguopeng/deepseek-r1-FP8-Dynamic/","choices":[{"index":0,
   "text":" is Beijing, and Shanghai is its most populous city by urban area population. China","logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"prompt_logprobs":null,"prompt_token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":5,"total_tokens":21,"completion_tokens":16,"prompt_tokens_details":null},"kv_transfer_params":null}
   ```

## Benchmark

1. Take deepseek as example, you can use the following command to benchmark serve.

    ```shell
    model="/path-to-model/deepseek-r1-FP8-Dynamic/"
    vllm bench serve \
        --host localhost \
        --port 8000 \
        --model ${model} \
        --dataset-name random \
        --random-input-len 3584 \
        --random-output-len 1024 \
        --max-concurrency 64 \
        --num-prompts 128 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --ignore-eos \
        # --profile
        # --seed 123 \
        # --request-rate 2 \
        2>&1 | tee log.client.log
    ```

## Evaluation

### Text Model Evaluation

Text model is evaluated using lm-eval (<https://github.com/EleutherAI/lm-evaluation-harness.git>).

1. Install dependencies. `python3 -m pip install lm_eval tenacity`.
2. Start lm-eval. Example:

    ```shell
    #!/bin/bash
    model="/path-to-model/deepseek-r1-FP8-Dynamic/"
    lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model=${model},base_url=http://127.0.0.1:8000/v1/completions \
    --batch_size 100 
    ```

    The eager-mode result should be:

    ```shell
    |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9522|±  |0.0059|
    |     |       |strict-match    |     5|exact_match|↑  |0.9530|±  |0.0058|
    ```

    The FULL_AND_PIECEWISE graph-mode result should be:

    ```shell
    |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9500|±  |0.0060|
    |     |       |strict-match    |     5|exact_match|↑  |0.9477|±  |0.0061|
    ```

    **Take notes:**

    - It is required to set --batch_size to larger value as the default value is 1.
    Setting --batch_size > 1 to evaluate if the batching logic is correctly implemented or not.
    - Extra details: lm-eval send seed requests. Thus, in vLLM sampling class, it will use the per-request sampling.

### Visual Model Evaluation

Vision Language Model accuracy evualuation is done using the tool from
<https://github.com/EmbeddedLLM/mistral-evals.git> (it is modified from
<https://github.com/mistralai/mistral-evals.git> to support batch size > 1 evaluation)

1. Install dependency. `python3 -m pip install fire`
2. Launch vLLM server. Example:

    ```shell
    #!/bin/bash
    rm -rf /root/.cache/vllm
    export  GPU_ARCHS=gfx942
    VLLM_USE_V1=1 \
    VLLM_ROCM_USE_AITER=1 \
    SAFETENSORS_FAST_GPU=1 \
    vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    -tp 4 \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --mm-encoder-tp-mode "data" \
    --trust_remote_code \
    > server_Qwen_Qwen2.5-VL-72B-Instruct.log 2>&1
    ```

3. Start evaluation. (Recommended chartqa dataset as the variance of the score is smaller). Example:

    ```shell
    #!/bin/bash
    pushd ./mistral-evals
    python3 -m eval.run eval_vllm \
            --model_name Qwen/Qwen2.5-VL-72B-Instruct\
            --url http://0.0.0.0:8000 \
            --output_dir ./chartqa \
            --eval_name "chartqa" \
            --max_new_tokens 1024 > lmeval_server_Qwen_Qwen2.5-VL-72B-Instruct.log 2>&1
    popd
    ```

    **Take notes:** The batch size is hard coded to 32 in the repository.

### Helper script

The launch scripts are attached to give an idea what are the configuration that was validated
at some point in time that works.
It also covers the models that are of interested in this branch.
