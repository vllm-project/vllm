#!/bin/bash

# Constants
SHARED_STORAGE_DIR="local_storage"
PREFILL_OUTPUT="prefill_output.txt"
DECODE_OUTPUT="decode_output.txt"
SYNC_DECODE_RECOVERED_OUTPUT="sync_decode_recovered_output.txt"
ASYNC_DECODE_RECOVERED_OUTPUT="async_decode_recovered_output.txt"

# Cleanup
rm -rf "$SHARED_STORAGE_DIR"
rm -f "$PREFILL_OUTPUT" "$DECODE_OUTPUT" "$SYNC_DECODE_RECOVERED_OUTPUT" "$ASYNC_DECODE_RECOVERED_OUTPUT"

# Run inference examples
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 prefill_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 decode_example.py
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 decode_example.py --simulate-failure
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 python3 decode_example.py --simulate-failure --async-load

# Compare outputs
if ! cmp -s "$DECODE_OUTPUT" "$SYNC_DECODE_RECOVERED_OUTPUT"; then
    echo "❌ Outputs differ: sync recovery failed."
    diff -u "$DECODE_OUTPUT" "$SYNC_DECODE_RECOVERED_OUTPUT"
    exit 1
fi

if ! cmp -s "$DECODE_OUTPUT" "$ASYNC_DECODE_RECOVERED_OUTPUT"; then
    echo "❌ Outputs differ: async recovery failed."
    diff -u "$DECODE_OUTPUT" "$ASYNC_DECODE_RECOVERED_OUTPUT"
    exit 1
fi

echo "✅ Outputs match: recovery successful."
