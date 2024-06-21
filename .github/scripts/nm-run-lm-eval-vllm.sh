#!/bin/bash
# We can use this script to compute baseline accuracy on GSM for transformers.
#
# Make sure you have lm-eval-harness installed:
#   pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@9516087b81a61d0e220b22cc1b75be76de23bc10

usage() {
    echo``
    echo "Runs lm eval harness on GSM8k using vllm server and compares to "
    echo "precomputed baseline (measured by HF transformers."
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -c    - path to the test data config (e.g. neuralmagic/lm-eval/YOUR_CONFIG.yaml)"
    echo
}

while getopts "c:" OPT; do
  case ${OPT} in
    c ) 
        CONFIG="$OPTARG"
        ;;
    \? ) 
        usage
        exit 1
        ;;
  esac
done

LM_EVAL_TEST_DATA_FILE=$CONFIG pytest -v tests/accuracy/test_lm_eval_correctness.py
