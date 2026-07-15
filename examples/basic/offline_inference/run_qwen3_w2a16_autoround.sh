#!/usr/bin/env bash
set -euo pipefail

python examples/basic/offline_inference/generate.py \
  --model /home/yiliu7/workspace/llmc-ds/examples/autoround/quantization_wNa16/Qwen3-30B-A3B-Instruct-2507-W2A16-G128-AutoRound-10iters/
