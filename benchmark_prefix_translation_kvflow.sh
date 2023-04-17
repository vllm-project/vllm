#!/bin/bash

# Set the initial value of num_prefix
num_prefix=1

# Loop over request_rate values for the first num_prefix
for request_rate in 1 24 32 40 48 50 51 52; do
    # Execute the Python script and append the output to log.txt
    python benchmark/benchmark_prefix_translation.py \
        --model ~/hf-llama/llama-13b/ \
        --num-prefix-examples "$num_prefix" \
        --request-rate "$request_rate" \
        --duration 1200 \
        --n1 1.0 \
        &>> log.txt
done

# Set the updated value of num_prefix
num_prefix=5

# Loop over request_rate values for the updated num_prefix
for request_rate in 1 24 36 40 41 42 43 44 45 46 47 48; do
    # Execute the Python script and append the output to log.txt
    python benchmark/benchmark_prefix_translation.py \
        --model ~/hf-llama/llama-13b/ \
        --num-prefix-examples "$num_prefix" \
        --request-rate "$request_rate" \
        --duration 1200 \
        --n1 1.0 \
        &>> log.txt
done
