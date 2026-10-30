#!/bin/bash
# Boot Qwen3.8-27B-AWQ-INT4 (cyankiwi, GS-32, BF16 MTP head) on GPU 2 for the #35 re-test.
# CoursIA slot window; STRICT GPU 2 pinning — never GPUs 0,1 (prod :5002 untouched).
# Levers validated in prod 2026-09-11 carried over: fp16 mamba state, prefix-match-unit 16,
# VLLM_WSL2_ENABLE_PIN_MEMORY=1 (V2-runner gate under WSL2, proven on this exact digest).
# KV = fp8 (NOT TQ: TQ x MTP degenerate on stock, vllm#53180).
# NO MTP on this slot: the cyankiwi checkpoint's BF16 MTP head wants a fixed
# 2.37 GiB embedding (qwen3_5_mtp.py:244) on top of 21.55 GiB of weights+overhead
# = 23.9 GiB, the whole card with zero KV. Structurally impossible on ONE 4090.
# MTP acceptance was already characterized on TP=2 (08-20: 0.75-0.79 solo, /1.6 load).
# NOTE mount: the WSL-side hub nests one extra level (.../huggingface/hub/hub/models--...);
# the real model store is that nested dir. Prod avoids the trap by mounting the parent
# onto /root/.cache/huggingface (HF_HOME); here we mount the nested dir straight onto
# /root/.cache/huggingface/hub.
# ISOLATION (measured 2026-10-30): under WSL2 `--gpus device=2` NO LONGER isolates —
# the container enumerates all 3 GPUs and CUDA index 0 is PROD GPU 0 (pci 01:00.0).
# CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=2 below are what pin GPU 2
# (pci 72:00.0). On 2026-09-12 the flag alone did isolate; what changed is not established.
# Run from Git Bash in d:/vllm:
#   MSYS_NO_PATHCONV=1 bash myia_vllm/_iteration_log/qwen38_27b_gpu2/serve_qwen38.sh
set -e
docker rm -f qwen38-eval 2>/dev/null || true
sleep 10  # let the killed engine's CUDA context release VRAM before the new one checks free memory
MSYS_NO_PATHCONV=1 docker run -d --name qwen38-eval \
  --gpus '"device=2"' \
  -p 127.0.0.1:5022:5022 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES=2 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -v '\\wsl.localhost\Ubuntu\home\user\vllm\.cache\huggingface\hub\hub:/root/.cache/huggingface/hub:ro' \
  -v vllm-compile-cache-qwen38-27b-eval-v0290:/root/.cache/vllm \
  vllm/vllm-openai:v0.29.0 \
  --model cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --served-model-name qwen3.8-27b \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --skip-mm-profiling \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256 \
  --prefix-match-unit 16 \
  --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --dtype auto \
  --trust-remote-code \
  --no-enable-flashinfer-autotune \
  --port 5022
echo "qwen38-eval started. Cold boot ~10 min (weights + torch.compile on fresh volume)."
echo "watch: docker logs -f qwen38-eval   (wait: Application startup complete)"
