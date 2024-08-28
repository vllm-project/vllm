#!/bin/bash

usage() {
    echo``
    echo "Runs lm eval harness on GSM8k using vllm and compares to "
    echo "precomputed baseline (measured by HF transformers.)"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -c    - path to the test data config (e.g. configs/small-models.txt)"
    echo "  -t    - tensor parallel size"
    echo
}

SUCCESS=0

while getopts "c:t:" OPT; do
  case ${OPT} in
    c ) 
        CONFIG="$OPTARG"
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

# Parse list of configs.
IFS=$'\n' read -d '' -r -a MODEL_CONFIGS < $CONFIG

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"
do
    LOCAL_SUCCESS=0
    
    echo "=== RUNNING MODEL: $MODEL_CONFIG WITH TP SIZE: $TP_SIZE==="

    export LM_EVAL_TEST_DATA_FILE=$PWD/configs/${MODEL_CONFIG}
    export LM_EVAL_TP_SIZE=$TP_SIZE
    pytest -s test_lm_eval_correctness.py || LOCAL_SUCCESS=$?

    if [[ $LOCAL_SUCCESS == 0 ]]; then
        echo "=== PASSED MODEL: ${MODEL_CONFIG} ==="
    else
        echo "=== FAILED MODEL: ${MODEL_CONFIG} ==="
    fi

    SUCCESS=$((SUCCESS + LOCAL_SUCCESS))

done

if [ "${SUCCESS}" -eq "0" ]; then
    exit 0
else
    exit 1
fi
