#!/bin/bash

rm -rf local_storage/
rm -f prefill_output.txt
rm -f decode_output.txt
rm -f decode_recovered_output.txt

VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 prefill_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 decode_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 decode_example.py --simulate-failure

# Compare outputs
if cmp -s decode_output.txt decode_recovered_output.txt; then
    echo "✅ Outputs match: recovery successful."
else
    echo "❌ Outputs differ: recovery failed."
    diff decode_output.txt decode_recovered_output.txt
    exit 1
fi
