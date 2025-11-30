#!/bin/bash
# We can use this script to compute baseline accuracy on aime25 for vllm.
#
# Make sure you have lm-eval-harness installed:
#   pip install lm-eval==0.4.9

usage() {
    echo``
    echo "Runs lm eval harness on aime25 using vllm."
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


lm_eval --model vllm \
  --model_args "pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,trust_remote_code" \
  --tasks aime25 \
  --batch_size auto \
  --apply_chat_template \
  --limit "$LIMIT" \
  --gen_kwargs temperature=1.0,top_p=1.0,top_k=0
