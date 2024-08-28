#!/bin/bash
# We can use this script to compute baseline accuracy on GSM for transformers.
#
# Make sure you have lm-eval-harness installed:
#   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@9516087b81a61d0e220b22cc1b75be76de23bc10

usage() {
    echo``
    echo "Runs lm eval harness on GSM8k using huggingface transformers."
    echo "This pathway is intended to be used to create baselines for "
    echo "our automated nm-test-accuracy workflow"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -m    - huggingface stub or local directory of the model"
    echo "  -b    - batch size to run the evaluation at"
    echo "  -l    - limit number of samples to run"
    echo "  -f    - number of fewshot samples to use"
    echo
}

while getopts "m:b:l:f:" OPT; do
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
    \? ) 
        usage
        exit 1
        ;;
  esac
done

lm_eval --model hf \
  --model_args pretrained=$MODEL,parallelize=True \
  --tasks gsm8k --num_fewshot $FEWSHOT --limit $LIMIT \
  --batch_size $BATCH_SIZE
