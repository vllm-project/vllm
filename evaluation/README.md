# Evaluation

## Text Model Evaluation

Text model is evaluated using lm-eval (<https://github.com/EleutherAI/lm-evaluation-harness.git>).

1. Install dependencies. `python3 -m pip install lm_eval tenacity`.
2. Launch vLLM server. Example:

    ```bash
    #!/bin/bash
    rm -rf /root/.cache/vllm
    export  GPU_ARCHS=gfx942
    MODEL=deepseek-ai/DeepSeek-R1
    AITER_ENABLE_VSKIP=0 \
    VLLM_USE_V1=1 \
    VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_CUSTOM_ALL_REDUCE=1 \
    vllm serve $MODEL \
    --tensor-parallel-size 8 \
    --disable-log-requests \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
    --trust-remote-code \
    --block-size 1 \
    --port 6789 \
    > server-deepseek-ai_DeepSeek-R1-aiter-v1.log 2>&1
    ```

3. Start lm-eval. Example:

    ```bash
    #!/bin/bash
    lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model=deepseek-ai/DeepSeek-R1,base_url=http://127.0.0.1:6789/v1/completions \
    --batch_size 100 \
    > lmeval_server-deepseek-ai_DeepSeek-R1-aiter-v1.log 2>&1
    ```

    **Take notes:**

    1. It is required to set --batch_size to larger value as the default value is 1.
    Setting --batch_size > 1 to evaluate if the batching logic is correctly implemented or not.
    2. Extra details: lm-eval send seed requests. Thus, in vLLM sampling class, it will use the per-request sampling.

## Visual Model Evaluation

Vision Language Model accuracy evualuation is done using the tool from
<https://github.com/EmbeddedLLM/mistral-evals.git> (it is modified from
<https://github.com/mistralai/mistral-evals.git> to support batch size > 1 evaluation)

1. Install dependency. `python3 -m pip install fire`
2. Launch vLLM server. Example:

    ```bash
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

    ```bash
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

## Helper script

The launch scripts are attached to give an idea what are the configuration that was validated
at some point in time that works.
It also covers the models that are of interested in this branch.
