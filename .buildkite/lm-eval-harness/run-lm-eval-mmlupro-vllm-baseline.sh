#!/bin/bash
# We can use this script to compute baseline accuracy on MMLUPRO for vllm.
# We use this for fp8, which HF does not support.
#
# Make sure you have lm-eval-harness installed:
#   pip install "lm-eval[api]>=0.4.10"

usage() {
    echo``
    echo "Runs lm eval harness on MMLU Pro using huggingface transformers."
    echo "This pathway is intended to be used to create baselines for "
    echo "our automated nm-test-accuracy workflow"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -m    - huggingface stub or local directory of the model"
    echo "  -l    - limit number of samples to run"
    echo "  -f    - number of fewshot samples to use"
    echo "  -t    - tensor parallel size to run at"
    echo
}

while getopts "m:b:l:f:t:" OPT; do
  case ${OPT} in
    m )
        MODEL="$OPTARG"
        ;;
    b )
        BATCH_SIZE="$OPTARG"
        ;;
    l )
        LIMIT="$OPTARG"
        ;;
    f )
        FEWSHOT="$OPTARG"
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
  --model_args "pretrained=$MODEL,tensor_parallel_size=$TP_SIZE,add_bos_token=true,trust_remote_code=true,max_model_len=4096" \
  --tasks mmlu_pro --num_fewshot "$FEWSHOT" --limit "$LIMIT" \
  --batch_size auto
