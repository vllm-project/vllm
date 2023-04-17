#!/bin/bash
# This script is used to test the performance of prefix translation for Orca.

#!/bin/bash
num_prefix=1
for request_rate in 1 24 26 28 29 30 31 32; do
    python benchmark/benchmark_prefix_translation_orca.py \
        --model ~/hf-llama/llama-13b/ \
        --num-prefix-examples "$num_prefix" \
        --request-rate "$request_rate" \
        --duration 1200 \
        --n1 1.0 \
        --len-estimator oracle \
        &>> log.txt
done

num_prefix=5
for request_rate in 1 4 8 9 10 11 12 13; do
    python benchmark/benchmark_prefix_translation_orca.py \
        --model ~/hf-llama/llama-13b/ \
        --num-prefix-examples "$num_prefix" \
        --request-rate "$request_rate" \
        --duration 1200 \
        --n1 1.0 \
        --len-estimator oracle \
        &>> log.txt
done
