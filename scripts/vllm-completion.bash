# ~/.vllm-completion.bash

_vllm_completions()
{
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local subcommands="chat complete serve bench collect-env run-batch"
    local bench_types="latency serve throughput"

    local chat_args="--api-key --model-name --system-prompt --url --help -h -q --quick"
    local complete_args="--api-key --model-name --url --help -h -q --quick"

    local serve_options="
    --allow-credentials --allowed-headers --allowed-methods
    --allowed-origins --api-key --api-server-count --chat-template
    --chat-template-content-format --config --data-parallel-start-rank
    --disable-fastapi-docs --disable-frontend-multiprocessing
    --disable-log-requests --disable-log-stats --disable-uvicorn-access-log
    --enable-auto-tool-choice --enable-prompt-tokens-details --enable-request-id-headers
    --enable-server-load-tracking --enable-ssl-refresh --headless --help
    --host --log-config-file --lora-modules --max-log-len --middleware --port
    --prompt-adapters --response-role --return-tokens-as-token-ids --root-path
    --ssl-ca-certs --ssl-cert-reqs --ssl-certfile --ssl-keyfile --tool-call-parser
    --tool-parser-plugin --use-v2-block-manager --uvicorn-log-level
    "

    local run_batch_options="
    --disable-log-requests --disable-log-stats --enable-metrics
    --enable-prompt-tokens-details --max-log-len --output-tmp-dir
    --url --use-v2-block-manager --port --response-role -h --help
    -i --input-file -o --output-file
    "

    local bench_latency_options="
    --batch-size --disable-detokenize --disable-log-stats --input-len
    --n --num-iters --num-iters-warmup --output-json --output-len
    --profile --use-beam-search --use-v2-block-manager -h --help
    "

    local bench_serve_options="
    --append-result --base-url --burstiness --dataset-name
    --dataset-path --disable-tqdm --endpoint --endpoint-type
    --goodput --host --ignore-eos --label --logprobs --lora-modules
    --max-concurrency --metadata --metric-percentiles --model
    --num-prompts --percentile-metrics --port --profile --request-rate
    --result-dir --result-filename --save-detailed --save-result
    --seed --served-model-name --tokenizer --tokenizer-mode
    --trust-remote-code --use-beam-search -h --help
    "

    local bench_throughput_options="
    --async-engine --backend --dataset --dataset-name
    --dataset-path --disable-detokenize --disable-frontend-multiprocessing
    --disable-log-requests --disable-log-stats --hf-max-batch-size
    --hf-split --hf-subset --input-len --lora-path --n --num-prompts
    --output-json --output-len --prefix-len --random-range-ratio
    --use-v2-block-manager -h --help
    "

    local modelconfig_args="
    --allowed-local-media-path --code-revision --config-format
    --disable-async-output-proc --disable-cascade-attn --disable-sliding-window
    --no-disable-sliding-window --dtype --enable-prompt-embeds --enable-sleep-mode
    --enforce-eager --generation-config --hf-config-path --hf-overrides --hf-token
    --logits-processor-pattern --max-logprobs --max-model-len --max-seq-len-to-capture
    --model --model-impl --override-attention-dtype --override-generation-config
    --override-neuron-config --override-pooler-config --quantization
    --revision --rope-scaling --rope-theta --seed --served-model-name
    --skip-tokenizer-init --task --tokenizer --tokenizer-mode --tokenizer-revision
    --trust-remote-code
    "

    local loadconfig_args="
    --download-dir --ignore-patterns --load-format
    --model-loader-extra-config --pt-load-map-location
    --qlora-adapter-name-or-path --use-tqdm-on-load
    "

    local decodingconfig_args="
    --enable-reasoning --no-enable-reasoning --guided-decoding-backend
    --guided-decoding-disable-additional-properties
    --no-guided-decoding-disable-additional-properties
    --guided-decoding-disable-any-whitespace
    --no-guided-decoding-disable-any-whitespace
    --guided-decoding-disable-fallback
    --no-guided-decoding-disable-fallback
    --reasoning-parser
    "

    local parallelconfig_args="
    --data-parallel-address --data-parallel-backend --data-parallel-rpc-port
    --data-parallel-size --data-parallel-size-local
    --disable-custom-all-reduce --no-disable-custom-all-reduce
    --distributed-executor-backend --enable-expert-parallel --no-enable-expert-parallel
    --enable-multimodal-encoder-data-parallel --no-enable-multimodal-encoder-data-parallel
    --max-parallel-loading-workers --pipeline-parallel-size --ray-workers-use-nsight
    --tensor-parallel-size --worker-cls --worker-extension-cls
    "

    local cacheconfig_args="
    --block-size --calculate-kv-scales --no-calculate-kv-scales
    --cpu-offload-gb --enable-prefix-caching --no-enable-prefix-caching
    --gpu-memory-utilization --kv-cache-dtype
    --num-gpu-blocks-override --prefix-caching-hash-algo --swap-space
    "

    local tokenizerpoolconfig_args="
    --tokenizer-pool-extra-config --tokenizer-pool-size --tokenizer-pool-type
    "

    local multimodalconfig_args="
    --disable-mm-preprocessor-cache --no-disable-mm-preprocessor-cache
    --limit-mm-per-prompt --mm-processor-kwargs
    "

    local loraconfig_args="
    --enable-lora --no-enable-lora
    --enable-lora-bias --no-enable-lora-bias
    --fully-sharded-loras --no-fully-sharded-loras
    --long-lora-scaling-factors --lora-dtype --lora-extra-vocab-size
    --max-cpu-loras --max-lora-rank --max-loras
    "

    local promptadapter_args="
    --enable-prompt-adapter --no-enable-prompt-adapter
    --max-prompt-adapter-token --max-prompt-adapters
    "

    local deviceconfig_args="
    --device
    "

    local speculative_args="
    --speculative-config
    "

    local observability_args="
    --collect-detailed-traces --otlp-traces-endpoint
    --show-hidden-metrics-for-version
    "

    local scheduler_args="
    --cuda-graph-sizes --disable-chunked-mm-input
    --no-disable-chunked-mm-input --disable-hybrid-kv-cache-manager
    --no-disable-hybrid-kv-cache-manager --enable-chunked-prefill
    --no-enable-chunked-prefill --long-prefill-token-threshold
    --max-long-partial-prefills --max-num-batched-tokens
    --max-num-partial-prefills --max-num-seqs --multi-step-stream-outputs
    --num-lookahead-slots --num-scheduler-steps --preemption-mode
    --scheduler-cls --scheduler-delay-factor --scheduling-policy
    "

    local vllm_args="
    --additional-config --compilation-config
    --kv-events-config --kv-transfer-config
    "

    # bench serve
    local custom_dataset_options="
    --custom-output-len --custom-skip-chat-template
    "
    local sonnet_dataset_options="
    --sonnet-input-len --sonnet-output-len --sonnet-prefix-len
    "
    local sharegpt_dataset_options="
    --sharegpt-output-len
    "
    local random_dataset_options="
    --random-input-len --random-output-len --random-prefix-len
    --random-range-ratio
    "
    local hf_dataset_options="
    --hf-output-len --hf-split --hf-subset
    "
    local sampling_parameters="
    --min-p --temperature --top-k --top-p
    "

    local serve_args="\
    $serve_options            $modelconfig_args         $loadconfig_args    \
    $decodingconfig_args      $parallelconfig_args      $cacheconfig_args   \
    $tokenizerpoolconfig_args $multimodalconfig_args    $loraconfig_args    \
    $promptadapter_args       $deviceconfig_args        $observability_args \
    $speculative_args         $scheduler_args           $vllm_args"

    local run_batch_args="\
    $run_batch_options        $modelconfig_args         $loadconfig_args    \
    $decodingconfig_args      $parallelconfig_args      $cacheconfig_args   \
    $tokenizerpoolconfig_args $multimodalconfig_args    $loraconfig_args    \
    $promptadapter_args       $deviceconfig_args        $observability_args \
    $speculative_args         $scheduler_args           $vllm_args"

    local bench_latency_args="\
    $bench_latency_options    $modelconfig_args         $loadconfig_args    \
    $decodingconfig_args      $parallelconfig_args      $cacheconfig_args   \
    $tokenizerpoolconfig_args $multimodalconfig_args    $loraconfig_args    \
    $promptadapter_args       $deviceconfig_args        $observability_args \
    $speculative_args         $scheduler_args           $vllm_args"

    local bench_serve_args="\
    $bench_serve_options      $custom_dataset_options   $sonnet_dataset_options \
    $sharegpt_dataset_options $random_dataset_options   $hf_dataset_options \
    $sampling_parameters"

    local bench_throughput_args="\
    $bench_throughput_options $modelconfig_args         $loadconfig_args    \
    $decodingconfig_args      $parallelconfig_args      $cacheconfig_args   \
    $tokenizerpoolconfig_args $multimodalconfig_args    $loraconfig_args    \
    $promptadapter_args       $deviceconfig_args        $observability_args \
    $speculative_args         $scheduler_args           $vllm_args"

    # Option value completion mapping (centralized definition)
    declare -A option_value_map=(
        [--config-format]="auto hf mistral"
        [--dtype]="auto bfloat16 float float16 float32 half"
        [--model-impl]="auto vllm transformers"
        [--quantization]="
            aqlm auto-round awq awq_marlin bitblas bitsandbytes compressed-tensors
            deepspeedfp experts_int8 fbgemm_fp8 fp8 gguf gptq gptq_bitblas
            gptq_marlin gptq_marlin_24 hqq ipex marlin modelopt modelopt_fp4
            moe_wna16 neuron_quant ptpc_fp8 qqq quark torchao tpu_int8 None"
        [--task]="auto classify draft embed embedding generate reward score transcription"
        [--tokenizer-mode]="auto custom mistral slow"
        [--load-format]="auto pt safetensors npcache dummy tensorizer sharded_state
        gguf bitsandbytes mistral runai_streamer runai_streamer_sharded fastsafetensors"
        [--distributed-executor-backend]="external_launcher mp ray uni None"
        [--block-size]="1 8 16 32 64 128"
        [--kv-cache-dtype]="auto fp8 fp8_e4m3 fp8_e5m2"
        [--prefix-caching-hash-algo]="builtin sha25"
        [--lora-dtype]="auto bfloat16 float16"
        [--device]="auto cpu cuda hpu neuron tpu xpu"
        [--collect-detailed-traces]="all model worker None"
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
                mapfile -t COMPREPLY < <(compgen -W "${bench_types}" -- "$cur")
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
