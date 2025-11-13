#!/bin/bash
# Run with 2 GPUs (or 2 CPU processes if no GPUs)
LD_PRELOAD="/usr/local/fbcode/platform010/lib/libcublasLt.so:/usr/local/fbcode/platform010/lib/libcublas.so" \
torchrun --nproc_per_node=2 test_real_parallelism.py
