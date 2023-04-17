#!/bin/bash
# This script is used to test the performance of prefix translation for Orca.

for num_prefix in 1 8 2 4; do
    for request_rate in 1 8 16 24 32 40 48; do
        python benchmark/benchmark_prefix_translation_orca.py \
            --model ~/hf-llama/llama-13b/ \
            --num-prefix-examples "$num_prefix" \
            --request-rate "$request_rate" \
            --duration 1200 \
            --n1 1.0 \
            --len-estimator oracle \
            &>> log.txt
    done
done
