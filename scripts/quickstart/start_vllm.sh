#! /bin/bash

# set -x

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

Help() {
    # Display Help
    echo "Start vllm server for a huggingface model on Gaudi."
    echo
    echo "Syntax: bash  start_vllm.sh <-w> [-u:p:l:b:c:s] [-h]"
    echo "options:"
    echo "w  Weights of the model, could be model id in huggingface or local path"
    echo "u  URL of the server, str, default=0.0.0.0"
    echo "p  Port number for the server, int, default=8688"
    echo "l  max_model_len for vllm, int, default=16384, maximal value for single node: 32768"
    echo "b  max_num_seqs for vllm, int, default=128"
    echo "c  Cache HPU recipe to the specified path, str, default=None"
    echo "s  Skip warmup or not, bool, default=false"
    echo "h  Help info"
    echo
}

#Default values for parameters
model_path=/data/hf_models/DeepSeek-R1-Gaudi
vllm_port=8688
warmup_cache_path=/data/warmup_cache
max_num_seqs=128
host=0.0.0.0
max_model_len=16384


while getopts hw:u:p:l:b:c:s flag; do
    case $flag in
    h) # display Help
        Help
        exit
        ;;
    w) # get model path
        model_path=$OPTARG ;;
    u) # get the URL of the server
        host=$OPTARG ;;
    p) # get the port of the server
        vllm_port=$OPTARG ;;
    l) # max-model-len
        max_model_len=$OPTARG ;;
    b) # batch size
        max_num_seqs=$OPTARG ;;
    c) # use_recipe_cache
        warmup_cache_path=$OPTARG ;;
    s) # skip_warmup
        skip_warmup=true ;;
    \?) # Invalid option
        echo "Error: Invalid option"
        Help
        exit
        ;;
    esac
done


if [ "$warmup_cache_path" != "" ]; then
    echo "HPU recipe cache will be saved to $warmup_cache_path"
    export PT_HPU_RECIPE_CACHE_CONFIG=${warmup_cache_path},false,16384
    mkdir -p "${warmup_cache_path}"
fi

if [ "$skip_warmup" = "true" ]; then
    echo "VLLM_SKIP_WARMUP is set to True"
    export VLLM_SKIP_WARMUP=True
fi


ray stop --force

# check platform
if hl-smi 2>/dev/null | grep -q HL-225; then
    echo "Gaudi2 OAM platform"
    default_decode_bs_step=8
elif hl-smi 2>/dev/null | grep -q HL-288; then
    echo "Gaudi2 PCIe platform"
    default_decode_bs_step=2
else
    echo "Unknown platform and exit..."
    exit 1
fi

# DO NOT change unless you fully undersand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export PT_HPU_LAZY_MODE=1

export VLLM_MOE_N_SLICE=8
export VLLM_EP_SIZE=8

block_size=128
# DO NOT change ends...

# memory footprint tunning params
if (( max_model_len <= 16384 )); then
	export VLLM_GPU_MEMORY_UTILIZATION=0.85
else
	export VLLM_GPU_MEMORY_UTILIZATION=0.75
fi
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0
export VLLM_MLA_DISABLE_REQUANTIZATION=0
export VLLM_DELAYED_SAMPLING="true"
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
#export VLLM_MOE_SLICE_LENGTH=20480

# params
max_num_batched_tokens=$max_model_len
input_min=1
input_max=$max_model_len
output_max=$max_model_len


unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

#export VLLM_SKIP_WARMUP=True



# !!!!!!!!!!!!!!!!!!!! set bucketing !!!!!!!!!!!!!
prompt_bs_min=1
prompt_bs_step=$(( $max_num_seqs > 32 ? 32 : $max_num_seqs ))
prompt_bs_max=$(( $max_num_seqs > 64 ? 64 : $max_num_seqs ))
export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

prompt_seq_step=128
prompt_seq_min=128
prompt_seq_max=$max_num_batched_tokens
export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

decode_bs_min=1
decode_bs_step=$(( $max_num_seqs > $default_decode_bs_step ? $default_decode_bs_step : $max_num_seqs ))
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


echo " environments are reseted "

env | grep VLLM


python3 -m vllm.entrypoints.openai.api_server --host $host --port $vllm_port \
--block-size 128 \
--model $model_path \
--device hpu \
--dtype bfloat16 \
--kv-cache-dtype fp8_inc \
--tensor-parallel-size 8 \
--trust-remote-code  \
--max-model-len $max_model_len \
--max-num-seqs $max_num_seqs \
--max-num-batched-tokens $max_num_batched_tokens  \
--use-padding-aware-scheduling \
--use-v2-block-manager \
--distributed_executor_backend ray \
--gpu_memory_utilization $VLLM_GPU_MEMORY_UTILIZATION \
--disable-log-requests \
--enable-reasoning \
--reasoning-parser deepseek_r1
