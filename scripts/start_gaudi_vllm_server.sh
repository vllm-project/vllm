#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

export PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache,True,16384

Help() {
    # Display Help
    echo "Start vllm server for a huggingface model on Gaudi."
    echo
    echo "Syntax: bash start_gaudi_vllm_server.sh <-w> [-n:m:u:p:d:i:o:t:l:b:e:c:sfza] [-h]"
    echo "options:"
    echo "w  Weights of the model, could be model id in huggingface or local path"
    echo "n  Number of HPU to use, [1-8], default=1"
    echo "m  Module IDs of the HPUs to use, [0-7], default=None"
    echo "u  URL of the server, str, default=127.0.0.1"
    echo "p  Port number for the server, int, default=30001"
    echo "d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'"
    echo "i  Input range, str, format='input_min,input_max', default='4,1024'"
    echo "o  Output range, str, format='output_min,output_max', default='4,2048'"
    echo "t  max_num_batched_tokens for vllm, int, default=8192"
    echo "l  max_model_len for vllm, int, default=4096"
    echo "b  max_num_seqs for vllm, int, default=128"
    echo "e  number of scheduler steps, int, default=1"
    echo "c  Cache HPU recipe to the specified path, str, default=None"
    echo "s  Skip warmup or not, bool, default=false"
    echo "f  Enable profiling or not, bool, default=false"
    echo "z  Disable zero-padding, bool, default=false"
    echo "a  Disable FusedFSDPA, bool, default=false"
    echo "h  Help info"
    echo
}

model_path=""
num_hpu=1
module_ids=None
host=127.0.0.1
#port=30001
#host=10.239.129.9
port=8080
dtype=bfloat16
input_range=(4 1024)
output_range=(4 2048)
max_num_batched_tokens=8192
max_model_len=4096
max_num_seqs=128
cache_path=""
skip_warmup=false
profile=false
disable_zero_padding=false
disable_fsdpa=false

block_size=128
scheduler_steps=1

# Get the options
while getopts hw:n:m:u:p:d:i:o:t:l:b:e:c:sfza flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    w) # get model path
        model_path=$OPTARG ;;
    n) # get number of HPUs
        num_hpu=$OPTARG ;;
    m) # get module ids to use
        module_ids=$OPTARG ;;
    u) # get the URL of the server
        host=$OPTARG ;;
    p) # get the port of the server
        port=$OPTARG ;;
    d) # get data type
        dtype=$OPTARG ;;
    i) # input range
        IFS="," read -r -a input_range <<< "$OPTARG" ;;
    o) # output range
        IFS="," read -r -a output_range <<< "$OPTARG" ;;
    t) # max-num-batched-tokens
        max_num_batched_tokens=$OPTARG ;;
    l) # max-model-len
        max_model_len=$OPTARG ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    e) # number of scheduler steps
        scheduler_steps=$OPTARG ;;
    c) # use_recipe_cache
        cache_path=$OPTARG ;;
    s) # skip_warmup
        skip_warmup=true ;;
    f) # enable profiling
        profile=true ;;
    z) # disable zero-padding
        disable_zero_padding=true ;;
    a) # disable FusedSDPA
        disable_fsdpa=true ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done

if [ "$model_path" = "" ]; then
    echo "[ERROR]: No model specified. Usage:"
    Help
    exit
fi

model_name=$( echo "$model_path" | awk -F/ '{print $NF}' )
input_min=${input_range[0]}
input_max=${input_range[1]}
output_min=${output_range[0]}
output_max=${output_range[1]}

if [ "$input_min" == "$input_max" ]; then
    disable_zero_padding=true
fi

if [ "$num_hpu" -gt 1 ]; then
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
    unset HLS_MODULE_ID
    if [ "$module_ids" != "None" ]; then
        export HABANA_VISIBLE_MODULES=$module_ids
    fi
else
    unset HABANA_VISIBLE_MODULES
    if [ "$module_ids" != "None" ]; then
        export HLS_MODULE_ID=$module_ids
    fi
fi

echo "Starting vllm server for ${model_name} from ${model_path} with input_range=[${input_min}, ${input_max}], output_range=[${output_min}, ${output_max}], max_num_seqs=${max_num_seqs}, max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"

device=$(hl-smi -Q name -f csv | tail -n 1)
case_name=serve_${model_name}_${dtype}_${device}_in${input_min}-${input_max}_out${output_min}-${output_max}_bs${max_num_seqs}_tp${num_hpu}_steps${scheduler_steps}_$(date +%F-%H-%M-%S)

case "$dtype" in
    "bfloat16" | "float16")
        echo Running with dtype="$dtype" ;;
    "fp8")
        echo Running with dtype="$dtype"
        export QUANT_CONFIG=quantization/${model_name}/maxabs_quant_g2.json
        QUANT_FLAGS=(--quantization inc --kv-cache-dtype fp8_inc)
        dtype="bfloat16"
        ;;
    "awq")
        echo Running with AWQ
        QUANT_FLAGS=(--quantization awq_hpu)
        dtype="bfloat16"
        ;;
    "gptq")
        echo Running with GPTQ
        QUANT_FLAGS=(--quantization gptq_hpu)
        dtype="bfloat16"
        ;;
    *)
        echo Invalid dtype: "$dtype"
        exit
        ;;
esac

if [ "$cache_path" != "" ]; then
    echo "HPU recipe cache will be saved to $cache_path"
    export PT_HPU_RECIPE_CACHE_CONFIG=${cache_path},false,4096
    mkdir -p "${cache_path}"
fi

if [ "$skip_warmup" = "true" ]; then
    echo "VLLM_SKIP_WARMUP is set to True"
    export VLLM_SKIP_WARMUP=True
fi

if [ "$profile" = "true" ]; then
    echo "VLLM_PROFILER_ENABLED is set to True"
    export VLLM_PROFILER_ENABLED=True
    export VLLM_PROFILE_FILE=${case_name}_profile.json
fi

if [ "$disable_zero_padding" = "true" ]; then
    echo "VLLM_ZERO_PADDING is disabled"
    export VLLM_ZERO_PADDING=false
else
    echo "VLLM_ZERO_PADDING is enabled"
    export VLLM_ZERO_PADDING=true
fi

if [ "$disable_fsdpa" = "true" ]; then
    echo "VLLM_PROMPT_USE_FUSEDSDPA is disabled"
    export VLLM_PROMPT_USE_FUSEDSDPA=false
else
    echo "VLLM_PROMPT_USE_FUSEDSDPA is enabled"
    export VLLM_PROMPT_USE_FUSEDSDPA=true
fi

export TOKENIZERS_PARALLELISM=true
export PT_HPU_WEIGHT_SHARING=0
export VLLM_MLA_DISABLE_REQUANTIZATION=1
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export VLLM_GRAPH_RESERVED_MEM=${VLLM_GRAPH_RESERVED_MEM:-"0.2"}
export VLLM_GRAPH_PROMPT_RATIO=${VLLM_GRAPH_PROMPT_RATIO:-"0"}
export VLLM_EP_SIZE=${VLLM_EP_SIZE:-"8"}
export VLLM_MOE_N_SLICE=${VLLM_MOE_N_SLICE:-"16"}
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-"0.9"}


# set up bucketing based on input/output range and max_num_batched_tokens
set_bucketing_new(){
    max_model_len=${max_model_len:-8192}
    max_num_seqs=${max_num_seqs:-128}

    prompt_bs_min=1
    prompt_bs_step=$(( $max_num_seqs > 32 ? 32 : $max_num_seqs ))
    prompt_bs_max=$(( $max_num_seqs > 64 ? 64 : $max_num_seqs ))
    export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
    export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
    export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

    prompt_seq_step=128
    prompt_seq_min=128
    prompt_seq_max=$max_model_len
    export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
    export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
    export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

    decode_bs_min=1
    decode_bs_step=$(( $max_num_seqs > 32 ? 32 : $max_num_seqs ))
    decode_bs_max=$max_num_seqs
    export VLLM_DECODE_BS_BUCKET_MIN=${VLLM_DECODE_BS_BUCKET_MIN:-$decode_bs_min}
    export VLLM_DECODE_BS_BUCKET_STEP=${VLLM_DECODE_BS_BUCKET_STEP:-$decode_bs_step}
    export VLLM_DECODE_BS_BUCKET_MAX=${VLLM_DECODE_BS_BUCKET_MAX:-$decode_bs_max}

    decode_block_min=128
    decode_block_step=128
    block_size=128
    decode_block_max=$(( ((max_num_seqs * max_model_len / block_size) > 128) ? (max_num_seqs * max_model_len / block_size) : 128 ))
    export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
    export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
    export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}
}


set_numactl
set_bucketing_new

${NUMA_CTL} \
python3 -m vllm.entrypoints.openai.api_server \
    --host "${host}" --port "${port}" \
    --model  "${model_path}" \
    --trust-remote-code \
    --tensor-parallel-size "${num_hpu}" \
    --dtype "${dtype}" \
    "${QUANT_FLAGS[@]}" \
    --block-size "${block_size}" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_num_batched_tokens" \
    --max-model-len "$max_model_len" \
    --disable-log-requests \
    --use-v2-block-manager \
    --use-padding-aware-scheduling \
    --num-scheduler-steps "${scheduler_steps}" \
    --distributed_executor_backend mp \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    |& tee "${case_name}".log
