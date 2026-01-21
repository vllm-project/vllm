#!/bin/bash

# Constants
SHARED_STORAGE_DIR="local_storage"
PREFILL_OUTPUT="prefill_output.txt"
DECODE_OUTPUT="decode_output.txt"
SEGMENTED_PREFILL_OUTPUT="segmented_prefill_decode_output.txt"

# Cleanup
rm -rf "$SHARED_STORAGE_DIR"
rm -f "$PREFILL_OUTPUT" "$DECODE_OUTPUT" "$SEGMENTED_PREFILL_OUTPUT"

# Run inference examples
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 prefill_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 decode_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 decode_example.py --segmented-prefill

# Compare outputs
if ! cmp -s "$DECODE_OUTPUT" "$SEGMENTED_PREFILL_OUTPUT"; then
    echo "❌ Outputs differ: segmented prefill output differs from regular prefill."
    diff -u "$DECODE_OUTPUT" "$SEGMENTED_PREFILL_OUTPUT"
    exit 1
fi


echo "✅ Outputs match: segmented prefill test successful."
