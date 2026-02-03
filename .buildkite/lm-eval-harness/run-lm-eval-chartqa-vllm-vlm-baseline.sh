#!/bin/bash
# We can use this script to compute baseline accuracy on chartqa for vllm.
#
# Make sure you have lm-eval-harness installed:
#   pip install "lm-eval[api]>=0.4.9.2"

usage() {
    echo``
    echo "Runs lm eval harness on ChartQA using multimodal vllm."
    echo "This pathway is intended to be used to create baselines for "
    echo "our correctness tests in vllm's CI."
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -m    - huggingface stub or local directory of the model"
    echo "  -l    - limit number of samples to run"
    echo "  -t    - tensor parallel size to run at"
    echo
}

while getopts "m:l:t:" OPT; do
  case ${OPT} in
    m ) 
        MODEL="$OPTARG"
        ;;
    l ) 
        LIMIT="$OPTARG"
        ;;
    t ) 
        TP_SIZE="$OPTARG"
        ;;
    \? ) 
        usage
        exit 1
        ;;
  esac
done

lm_eval --model vllm-vlm \
  --model_args "pretrained=$MODEL,tensor_parallel_size=$TP_SIZE" \
  --tasks chartqa \
  --batch_size auto \
  --apply_chat_template \
  --limit $LIMIT
