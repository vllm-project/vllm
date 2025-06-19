# vllm-completion.template.bash
# 
# To help auto-generate vllm-completion.bash
#
#############################################

_vllm_completions(){
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local subcommands="chat complete serve bench collect-env run-batch"

    # Nested subcommand
    local bench_subcommands="latency serve throughput"

    # Subcommands args
    local chat_args="--api-key --help --model-name --quick --system-prompt --url -h -q"

    local complete_args="--api-key --help --model-name --quick --url -h -q"

    local serve_args="--additional-config --allow-credentials --allowed-headers --allowed-local-media-path --allowed-methods --allowed-origins --api-key --api-server-count --block-size --calculate-kv-scales --chat-template --chat-template-content-format --code-revision --collect-detailed-traces --compilation-config --config --config-format --cpu-offload-gb --cuda-graph-sizes --data-parallel-address --data-parallel-backend --data-parallel-rpc-port --data-parallel-size --data-parallel-size-local --data-parallel-start-rank --device --disable-async-output-proc --disable-cascade-attn --disable-chunked-mm-input --disable-custom-all-reduce --disable-fastapi-docs --disable-frontend-multiprocessing --disable-hybrid-kv-cache-manager --disable-log-requests --disable-log-stats --disable-mm-preprocessor-cache --disable-sliding-window --disable-uvicorn-access-log --distributed-executor-backend --download-dir --dtype --enable-auto-tool-choice --enable-chunked-prefill --enable-expert-parallel --enable-lora --enable-lora-bias --enable-multimodal-encoder-data-parallel --enable-prefix-caching --enable-prompt-adapter --enable-prompt-embeds --enable-prompt-tokens-details --enable-reasoning --enable-request-id-headers --enable-server-load-tracking --enable-sleep-mode --enable-ssl-refresh --enforce-eager --fully-sharded-loras --generation-config --gpu-memory-utilization --guided-decoding-backend --guided-decoding-disable-additional-properties --guided-decoding-disable-any-whitespace --guided-decoding-disable-fallback --headless --help --hf-config-path --hf-overrides --hf-token --host --ignore-patterns --json-arg --kv-cache-dtype --kv-events-config --kv-transfer-config --limit-mm-per-prompt --load-format --log-config-file --logits-processor-pattern --long-lora-scaling-factors --long-prefill-token-threshold --lora-dtype --lora-extra-vocab-size --lora-modules --max-cpu-loras --max-log-len --max-logprobs --max-long-partial-prefills --max-lora-rank --max-loras --max-model-len --max-num-batched-tokens --max-num-partial-prefills --max-num-seqs --max-parallel-loading-workers --max-prompt-adapter-token --max-prompt-adapters --max-seq-len-to-capture --middleware --mm-processor-kwargs --model-impl --model-loader-extra-config --multi-step-stream-outputs --no-calculate-kv-scales --no-disable-cascade-attn --no-disable-chunked-mm-input --no-disable-custom-all-reduce --no-disable-hybrid-kv-cache-manager --no-disable-mm-preprocessor-cache --no-disable-sliding-window --no-enable-chunked-prefill --no-enable-expert-parallel --no-enable-lora --no-enable-lora-bias --no-enable-multimodal-encoder-data-parallel --no-enable-prefix-caching --no-enable-prompt-adapter --no-enable-prompt-embeds --no-enable-reasoning --no-enable-sleep-mode --no-enforce-eager --no-fully-sharded-loras --no-guided-decoding-disable-additional-properties --no-guided-decoding-disable-any-whitespace --no-guided-decoding-disable-fallback --no-multi-step-stream-outputs --no-ray-workers-use-nsight --no-skip-tokenizer-init --no-trust-remote-code --no-use-tqdm-on-load --num-gpu-blocks-override --num-lookahead-slots --num-scheduler-steps --otlp-traces-endpoint --override-attention-dtype --override-generation-config --override-neuron-config --override-pooler-config --pipeline-parallel-size --port --preemption-mode --prefix-caching-hash-algo --prompt-adapters --pt-load-map-location --qlora-adapter-name-or-path --quantization --ray-workers-use-nsight --reasoning-parser --response-role --return-tokens-as-token-ids --revision --root-path --rope-scaling --rope-theta --scheduler-cls --scheduler-delay-factor --scheduling-policy --seed --served-model-name --show-hidden-metrics-for-version --skip-tokenizer-init --speculative-config --ssl-ca-certs --ssl-cert-reqs --ssl-certfile --ssl-keyfile --swap-space --task --tensor-parallel-size --tokenizer --tokenizer-mode --tokenizer-pool-extra-config --tokenizer-pool-size --tokenizer-pool-type --tokenizer-revision --tool-call-parser --tool-parser-plugin --trust-remote-code --use-tqdm-on-load --use-v2-block-manager --uvicorn-log-level --worker-cls --worker-extension-cls -O -h -q"

    local run_batch_args="--additional-config --allowed-local-media-path --block-size --calculate-kv-scales --code-revision --collect-detailed-traces --compilation-config --config-format --cpu-offload-gb --cuda-graph-sizes --data-parallel-address --data-parallel-backend --data-parallel-rpc-port --data-parallel-size --data-parallel-size-local --device --disable-async-output-proc --disable-cascade-attn --disable-chunked-mm-input --disable-custom-all-reduce --disable-hybrid-kv-cache-manager --disable-log-requests --disable-log-stats --disable-mm-preprocessor-cache --disable-sliding-window --distributed-executor-backend --download-dir --dtype --enable-chunked-prefill --enable-expert-parallel --enable-lora --enable-lora-bias --enable-metrics --enable-multimodal-encoder-data-parallel --enable-prefix-caching --enable-prompt-adapter --enable-prompt-embeds --enable-prompt-tokens-details --enable-reasoning --enable-sleep-mode --enforce-eager --fully-sharded-loras --generation-config --gpu-memory-utilization --guided-decoding-backend --guided-decoding-disable-additional-properties --guided-decoding-disable-any-whitespace --guided-decoding-disable-fallback --help --hf-config-path --hf-overrides --hf-token --ignore-patterns --input-file --json-arg --kv-cache-dtype --kv-events-config --kv-transfer-config --limit-mm-per-prompt --load-format --logits-processor-pattern --long-lora-scaling-factors --long-prefill-token-threshold --lora-dtype --lora-extra-vocab-size --max-cpu-loras --max-log-len --max-logprobs --max-long-partial-prefills --max-lora-rank --max-loras --max-model-len --max-num-batched-tokens --max-num-partial-prefills --max-num-seqs --max-parallel-loading-workers --max-prompt-adapter-token --max-prompt-adapters --max-seq-len-to-capture --mm-processor-kwargs --model --model-impl --model-loader-extra-config --multi-step-stream-outputs --no-calculate-kv-scales --no-disable-cascade-attn --no-disable-chunked-mm-input --no-disable-custom-all-reduce --no-disable-hybrid-kv-cache-manager --no-disable-mm-preprocessor-cache --no-disable-sliding-window --no-enable-chunked-prefill --no-enable-expert-parallel --no-enable-lora --no-enable-lora-bias --no-enable-multimodal-encoder-data-parallel --no-enable-prefix-caching --no-enable-prompt-adapter --no-enable-prompt-embeds --no-enable-reasoning --no-enable-sleep-mode --no-enforce-eager --no-fully-sharded-loras --no-guided-decoding-disable-additional-properties --no-guided-decoding-disable-any-whitespace --no-guided-decoding-disable-fallback --no-multi-step-stream-outputs --no-ray-workers-use-nsight --no-skip-tokenizer-init --no-trust-remote-code --no-use-tqdm-on-load --num-gpu-blocks-override --num-lookahead-slots --num-scheduler-steps --otlp-traces-endpoint --output-file --output-tmp-dir --override-attention-dtype --override-generation-config --override-neuron-config --override-pooler-config --pipeline-parallel-size --port --preemption-mode --prefix-caching-hash-algo --pt-load-map-location --qlora-adapter-name-or-path --quantization --ray-workers-use-nsight --reasoning-parser --response-role --revision --rope-scaling --rope-theta --scheduler-cls --scheduler-delay-factor --scheduling-policy --seed --served-model-name --show-hidden-metrics-for-version --skip-tokenizer-init --speculative-config --swap-space --task --tensor-parallel-size --tokenizer --tokenizer-mode --tokenizer-pool-extra-config --tokenizer-pool-size --tokenizer-pool-type --tokenizer-revision --trust-remote-code --url --use-tqdm-on-load --use-v2-block-manager --worker-cls --worker-extension-cls -O -h -i -o -q"

    local bench_latency_args="--additional-config --allowed-local-media-path --batch-size --block-size --calculate-kv-scales --code-revision --collect-detailed-traces --compilation-config --config-format --cpu-offload-gb --cuda-graph-sizes --data-parallel-address --data-parallel-backend --data-parallel-rpc-port --data-parallel-size --data-parallel-size-local --device --disable-async-output-proc --disable-cascade-attn --disable-chunked-mm-input --disable-custom-all-reduce --disable-detokenize --disable-hybrid-kv-cache-manager --disable-log-stats --disable-mm-preprocessor-cache --disable-sliding-window --distributed-executor-backend --download-dir --dtype --enable-chunked-prefill --enable-expert-parallel --enable-lora --enable-lora-bias --enable-multimodal-encoder-data-parallel --enable-prefix-caching --enable-prompt-adapter --enable-prompt-embeds --enable-reasoning --enable-sleep-mode --enforce-eager --fully-sharded-loras --generation-config --gpu-memory-utilization --guided-decoding-backend --guided-decoding-disable-additional-properties --guided-decoding-disable-any-whitespace --guided-decoding-disable-fallback --help --hf-config-path --hf-overrides --hf-token --ignore-patterns --input-len --json-arg --kv-cache-dtype --kv-events-config --kv-transfer-config --limit-mm-per-prompt --load-format --logits-processor-pattern --long-lora-scaling-factors --long-prefill-token-threshold --lora-dtype --lora-extra-vocab-size --max-cpu-loras --max-logprobs --max-long-partial-prefills --max-lora-rank --max-loras --max-model-len --max-num-batched-tokens --max-num-partial-prefills --max-num-seqs --max-parallel-loading-workers --max-prompt-adapter-token --max-prompt-adapters --max-seq-len-to-capture --mm-processor-kwargs --model --model-impl --model-loader-extra-config --multi-step-stream-outputs --n --no-calculate-kv-scales --no-disable-cascade-attn --no-disable-chunked-mm-input --no-disable-custom-all-reduce --no-disable-hybrid-kv-cache-manager --no-disable-mm-preprocessor-cache --no-disable-sliding-window --no-enable-chunked-prefill --no-enable-expert-parallel --no-enable-lora --no-enable-lora-bias --no-enable-multimodal-encoder-data-parallel --no-enable-prefix-caching --no-enable-prompt-adapter --no-enable-prompt-embeds --no-enable-reasoning --no-enable-sleep-mode --no-enforce-eager --no-fully-sharded-loras --no-guided-decoding-disable-additional-properties --no-guided-decoding-disable-any-whitespace --no-guided-decoding-disable-fallback --no-multi-step-stream-outputs --no-ray-workers-use-nsight --no-skip-tokenizer-init --no-trust-remote-code --no-use-tqdm-on-load --num-gpu-blocks-override --num-iters --num-iters-warmup --num-lookahead-slots --num-scheduler-steps --otlp-traces-endpoint --output-json --output-len --override-attention-dtype --override-generation-config --override-neuron-config --override-pooler-config --pipeline-parallel-size --preemption-mode --prefix-caching-hash-algo --profile --pt-load-map-location --qlora-adapter-name-or-path --quantization --ray-workers-use-nsight --reasoning-parser --revision --rope-scaling --rope-theta --scheduler-cls --scheduler-delay-factor --scheduling-policy --seed --served-model-name --show-hidden-metrics-for-version --skip-tokenizer-init --speculative-config --swap-space --task --tensor-parallel-size --tokenizer --tokenizer-mode --tokenizer-pool-extra-config --tokenizer-pool-size --tokenizer-pool-type --tokenizer-revision --trust-remote-code --use-beam-search --use-tqdm-on-load --use-v2-block-manager --worker-cls --worker-extension-cls -O -h -q"

    local bench_serve_args="--append-result --base-url --burstiness --custom-output-len --custom-skip-chat-template --dataset-name --dataset-path --disable-tqdm --endpoint --endpoint-type --goodput --help --hf-output-len --hf-split --hf-subset --host --ignore-eos --label --logprobs --lora-modules --max-concurrency --metadata --metric-percentiles --min-p --model --num-prompts --percentile-metrics --port --profile --random-input-len --random-output-len --random-prefix-len --random-range-ratio --request-rate --result-dir --result-filename --save-detailed --save-result --seed --served-model-name --sharegpt-output-len --sonnet-input-len --sonnet-output-len --sonnet-prefix-len --temperature --tokenizer --tokenizer-mode --top-k --top-p --trust-remote-code --use-beam-search -h"

    local bench_throughput_args="--additional-config --allowed-local-media-path --async-engine --backend --block-size --calculate-kv-scales --code-revision --collect-detailed-traces --compilation-config --config-format --cpu-offload-gb --cuda-graph-sizes --data-parallel-address --data-parallel-backend --data-parallel-rpc-port --data-parallel-size --data-parallel-size-local --dataset --dataset-name --dataset-path --device --disable-async-output-proc --disable-cascade-attn --disable-chunked-mm-input --disable-custom-all-reduce --disable-detokenize --disable-frontend-multiprocessing --disable-hybrid-kv-cache-manager --disable-log-requests --disable-log-stats --disable-mm-preprocessor-cache --disable-sliding-window --distributed-executor-backend --download-dir --dtype --enable-chunked-prefill --enable-expert-parallel --enable-lora --enable-lora-bias --enable-multimodal-encoder-data-parallel --enable-prefix-caching --enable-prompt-adapter --enable-prompt-embeds --enable-reasoning --enable-sleep-mode --enforce-eager --fully-sharded-loras --generation-config --gpu-memory-utilization --guided-decoding-backend --guided-decoding-disable-additional-properties --guided-decoding-disable-any-whitespace --guided-decoding-disable-fallback --help --hf-config-path --hf-max-batch-size --hf-overrides --hf-split --hf-subset --hf-token --ignore-patterns --input-len --json-arg --kv-cache-dtype --kv-events-config --kv-transfer-config --limit-mm-per-prompt --load-format --logits-processor-pattern --long-lora-scaling-factors --long-prefill-token-threshold --lora-dtype --lora-extra-vocab-size --lora-path --max-cpu-loras --max-logprobs --max-long-partial-prefills --max-lora-rank --max-loras --max-model-len --max-num-batched-tokens --max-num-partial-prefills --max-num-seqs --max-parallel-loading-workers --max-prompt-adapter-token --max-prompt-adapters --max-seq-len-to-capture --mm-processor-kwargs --model --model-impl --model-loader-extra-config --multi-step-stream-outputs --n --no-calculate-kv-scales --no-disable-cascade-attn --no-disable-chunked-mm-input --no-disable-custom-all-reduce --no-disable-hybrid-kv-cache-manager --no-disable-mm-preprocessor-cache --no-disable-sliding-window --no-enable-chunked-prefill --no-enable-expert-parallel --no-enable-lora --no-enable-lora-bias --no-enable-multimodal-encoder-data-parallel --no-enable-prefix-caching --no-enable-prompt-adapter --no-enable-prompt-embeds --no-enable-reasoning --no-enable-sleep-mode --no-enforce-eager --no-fully-sharded-loras --no-guided-decoding-disable-additional-properties --no-guided-decoding-disable-any-whitespace --no-guided-decoding-disable-fallback --no-multi-step-stream-outputs --no-ray-workers-use-nsight --no-skip-tokenizer-init --no-trust-remote-code --no-use-tqdm-on-load --num-gpu-blocks-override --num-lookahead-slots --num-prompts --num-scheduler-steps --otlp-traces-endpoint --output-json --output-len --override-attention-dtype --override-generation-config --override-neuron-config --override-pooler-config --pipeline-parallel-size --preemption-mode --prefix-caching-hash-algo --prefix-len --pt-load-map-location --qlora-adapter-name-or-path --quantization --random-range-ratio --ray-workers-use-nsight --reasoning-parser --revision --rope-scaling --rope-theta --scheduler-cls --scheduler-delay-factor --scheduling-policy --seed --served-model-name --show-hidden-metrics-for-version --skip-tokenizer-init --speculative-config --swap-space --task --tensor-parallel-size --tokenizer --tokenizer-mode --tokenizer-pool-extra-config --tokenizer-pool-size --tokenizer-pool-type --tokenizer-revision --trust-remote-code --use-tqdm-on-load --use-v2-block-manager --worker-cls --worker-extension-cls -O -h -q"

    # Option value completion mapping (centralized definition)
    declare -A option_value_map=(
            [--backend]="vllm hf mii vllm-chat"
    [--block-size]="1 128 16 32 64 8"
    [--chat-template-content-format]="auto string openai"
    [--collect-detailed-traces]="None all model worker"
    [--config-format]="auto hf mistral"
    [--dataset-name]="burstgpt custom hf random sharegpt sonnet"
    [--device]="auto cpu cuda hpu neuron tpu xpu"
    [--distributed-executor-backend]="None external_launcher mp ray uni"
    [--dtype]="auto bfloat16 float float16 float32 half"
    [--endpoint-type]="vllm openai openai-chat openai-audio"
    [--guided-decoding-backend]="auto guidance lm-format-enforcer outlines xgrammar"
    [--kv-cache-dtype]="auto fp8 fp8_e4m3 fp8_e5m2"
    [--load-format]="auto bitsandbytes dummy fastsafetensors gguf mistral npcache pt runai_streamer runai_streamer_sharded safetensors sharded_state tensorizer"
    [--lora-dtype]="auto bfloat16 float16"
    [--model-impl]="auto transformers vllm"
    [--preemption-mode]="None recompute swap"
    [--prefix-caching-hash-algo]="builtin sha256"
    [--quantization]="None aqlm auto-round awq awq_marlin bitblas bitsandbytes compressed-tensors deepspeedfp experts_int8 fbgemm_fp8 fp8 gguf gptq gptq_bitblas gptq_marlin gptq_marlin_24 hqq ipex marlin modelopt modelopt_fp4 moe_wna16 neuron_quant ptpc_fp8 qqq quark torchao tpu_int8"
    [--reasoning-parser]="deepseek_r1 granite qwen3"
    [--scheduling-policy]="fcfs priority"
    [--task]="auto classify draft embed embedding generate reward score transcription"
    [--tokenizer-mode]="auto custom mistral slow"
    [--tool-call-parser]="deepseek_v3 granite-20b-fc granite hermes internlm jamba llama4_pythonic llama4_json llama3_json mistral phi4_mini_json pythonic"
    [--uvicorn-log-level]="debug info warning error critical trace"
    )

    if [[ -n "${option_value_map[$prev]}" ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${option_value_map[$prev]}" -- "$cur")
        return 0
    fi

    # Top-level subcommands
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${subcommands}" -- "$cur")
        return 0
    fi

    # Second-level handling
    case "${COMP_WORDS[1]}" in
        bench)
            if [[ ${COMP_CWORD} -eq 2 ]]; then
                mapfile -t COMPREPLY < <(compgen -W "${bench_subcommands}" -- "$cur")
                return 0
            fi

            # bench subcommands （latency, serve, throughput）
            local bench_sub=${COMP_WORDS[2]}
            case "$bench_sub" in
                latency)
                    mapfile -t COMPREPLY < <(compgen -W "${bench_latency_args}" -- "$cur")
                    ;;
                serve)
                    mapfile -t COMPREPLY < <(compgen -W "${bench_serve_args}" -- "$cur")
                    ;;
                throughput)
                    mapfile -t COMPREPLY < <(compgen -W "${bench_throughput_args}" -- "$cur")
                    ;;
            esac
            return 0
            ;;
        chat)
            mapfile -t COMPREPLY < <(compgen -W "${chat_args}" -- "$cur")
            return 0
            ;;
        complete)
            mapfile -t COMPREPLY < <(compgen -W "${complete_args}" -- "$cur")
            return 0
            ;;
        serve)
            mapfile -t COMPREPLY < <(compgen -W "${serve_args}" -- "$cur")
            return 0
            ;;
        run-batch)
            mapfile -t COMPREPLY < <(compgen -W "${run_batch_args}" -- "$cur")
            return 0
            ;;
        *)
            ;;
    esac
}

complete -F _vllm_completions vllm
