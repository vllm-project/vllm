#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

Help() {
    # Display Help
    echo "Benchmark vllm throughput for a huggingface model on Gaudi."
    echo
    echo "Syntax: bash benchmark_throughput.sh <-w> [-n:m:d:i:o:r:j:t:l:b:c:sfza] [-h]"
    echo "options:"
    echo "w  Weights of the model, could be model id in huggingface or local path"
    echo "n  Number of HPU to use, [1-8], default=1"
    echo "m  Module IDs of the HPUs to use, [0-7], default=None"
    echo "d  Data type, str, ['bfloat16'|'float16'|'fp8'|'awq'|'gptq'], default='bfloat16'"
    echo "i  Input length, int, default=1024"
    echo "o  Output length, int, default=512"
    echo "r  Ratio for min input/output length to generate an uniform distributed input/out length, float, default=1.0"
    echo "j  Json path of the ShareGPT dataset, str, default=None"
    echo "t  max_num_batched_tokens for vllm, int, default=8192"
    echo "l  max_model_len for vllm, int, default=4096"
    echo "b  max_num_seqs for vllm, int, default=128"
    echo "p  number of prompts, int, default=1000"
    echo "e  number of scheduler steps, int, default=1"
    echo "c  Cache HPU recipe to the specified path, str, default=None"
    echo "s  Skip warmup or not, bool, default=false"
    echo "f  Enable profiling or not, bool, default=false"
    echo "z  Disable zero-padding, bool, default=false"
    echo "a  Disable FusedFSDPA, bool, default=false"
    echo "h  Help info"
    echo
    echo "Note: set -j <sharegpt json path> will override -i, -o and -r"
    echo
}

model_path=""
num_hpu=1
module_ids="None"
dtype=bfloat16
input_len=1024
output_len=512
len_ratio=1.0
max_num_batched_tokens=8192
max_model_len=4096
max_num_seqs=128
num_prompts=1000
json_path=""
cache_path=""
skip_warmup=false
profile=false
disable_zero_padding=false
disable_fsdpa=false

block_size=128
scheduler_steps=1

# Get the options
while getopts hw:n:m:d:i:o:r:j:t:l:b:p:e:c:sfza flag; do
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
    d) # get data type
        dtype=$OPTARG ;;
    i) # input range
        input_len=$OPTARG ;;
    o) # output range
        output_len=$OPTARG ;;
    r) # ratio of min length
        len_ratio=$OPTARG ;;
    j) # json path
        json_path=$OPTARG ;;
    t) # max-num-batched-tokens
        max_num_batched_tokens=$OPTARG ;;
    l) # max-model-len
        max_model_len=$OPTARG ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    p) # number of prompts
        num_prompts=$OPTARG ;;
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

device=$(hl-smi -Q name -f csv | tail -n 1)
if [ "$json_path" != "" ]; then
    input_min=4
    input_max=1024
    output_min=4
    output_max=2048
    IO_FLAGS=(--dataset "$json_path")
    echo "Benchmarking throughput for ${model_name} from ${model_path} using ${num_prompts} random prompts from ${json_path} with max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"
    case_name="benchmark_throughput_${model_name}_${dtype}_${device}_sharegpt_bs${max_num_seqs}_tp${num_hpu}_step${scheduler_steps}_$(date +%F-%H-%M-%S)"
elif [ "$len_ratio" == "1.0" ]; then
    input_min=$input_len
    input_max=$input_len
    output_min=$output_len
    output_max=$output_len
    disable_zero_padding=true
    IO_FLAGS=(--input-len "$input_len" --output-len "$output_len")
    echo "Benchmarking throughput for ${model_name} from ${model_path} using ${num_prompts} fixed-length prompts with input_len=${input_len}, output_len=${output_len}, max_num_seqs=${max_num_seqs}, max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"
    case_name="benchmark_throughput_${model_name}_${dtype}_${device}_in${input_len}_out${output_len}_bs${max_num_seqs}_tp${num_hpu}_step${scheduler_steps}_$(date +%F-%H-%M-%S)"
else
    input_min=$(bc <<< "($input_len * $len_ratio + 0.5) / 1")
    input_max=$input_len
    output_min=$(bc <<< "($output_len * $len_ratio + 0.5) / 1")
    output_max=$output_len
    IO_FLAGS=(--dataset random --random-input-len "$input_len" --random-output-len "$output_len" --random-range-ratio "$len_ratio")
    echo "Benchmarking throughput for ${model_name} from ${model_path} using ${num_prompts} random-length prompts with input_range=[${input_min}, ${input_max}], output_range=[${output_min}, ${output_max}], max_num_seqs=${max_num_seqs}, max_num_batched_tokens=${max_num_batched_tokens}, max_model_len=${max_model_len} using ${num_hpu} HPUs with module_ids=${module_ids}"
    case_name="benchmark_throughput_${model_name}_${dtype}_${device}_in${input_min}-${input_max}_out${output_min}-${output_max}_bs${max_num_seqs}_tp${num_hpu}_step${scheduler_steps}_$(date +%F-%H-%M-%S)"
fi

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
export VLLM_GRAPH_PROMPT_RATIO=${VLLM_GRAPH_PROMPT_RATIO:-"0.8"}
export VLLM_EP_SIZE=${VLLM_EP_SIZE:-"1"}
export VLLM_MOE_N_SLICE=${VLLM_MOE_N_SLICE:-"4"}

gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-"0.9"}

set_numactl
set_bucketing

${NUMA_CTL} \
python "$BASH_DIR/../benchmarks/benchmark_throughput.py" \
    --backend vllm \
    --model "${model_path}" \
    --trust-remote-code \
    --tensor-parallel-size "${num_hpu}" \
    "${IO_FLAGS[@]}" \
    --device hpu \
    --dtype "${dtype}" \
    "${QUANT_FLAGS[@]}" \
    --seed 0 \
    --block-size "${block_size}" \
    --max-num-seqs "${max_num_seqs}" \
    --max-num-batched-tokens "${max_num_batched_tokens}" \
    --max-model-len "${max_model_len}" \
    --num-prompts "${num_prompts}" \
    --disable-log-requests \
    --use-v2-block-manager \
    --use-padding-aware-scheduling \
    --num-scheduler-steps "${scheduler_steps}" \
    --distributed_executor_backend mp \
    --gpu-memory-utilization "${gpu_memory_utilization}" \
    |& tee "${case_name}".log
