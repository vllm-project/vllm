# JIT Decompression Serving Test

Compares baseline GPU serving vs JIT decompression serving for a quantized model.
Measures GPU VRAM, CPU RAM, and tok/s for each scenario.

---

## Prerequisites

vLLM built with nvCOMP. Verify with:

```bash
PYTHONPATH=/workspace python -c "
import torch, vllm._C
print('nvCOMP available:', bool(torch.ops._C.is_gpu_decompress_available()))
"
```

If it prints `False`, rebuild with `-DVLLM_NVCOMP_PATH=...` (see below).

---

## Step 0 — Rebuild the kernel (if needed or after csrc/ changes)

```bash
cd /workspace
cmake --preset release -DVLLM_NVCOMP_PATH=/path/to/nvcomp/install
cmake --build --preset release --target install
```

---

## Step 1 — Compress the model (one-time)

```bash
PYTHONPATH=/workspace python tools/compress_weights.py \
    --model-path /models/your-model-awq \
    --output-path /models/your-model-awq-compressed \
    --algorithm gdeflate \
    --verify
```

Expected output: `All N tensors verified OK.` and a ~10–20% reduction.

---

## Step 2A — Serve baseline (uncompressed, normal GPU load)

**Terminal 1:**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/your-model-awq \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --dtype float16
```

**Terminal 2 — monitor memory while it loads:**
```bash
watch -n1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; free -h | grep Mem'
```

Note down peak GPU VRAM and CPU RAM once the server prints `Application startup complete`.

---

## Step 2B — Serve with JIT decompression

Stop the baseline server (`Ctrl+C`), then:

**Terminal 1:**
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /models/your-model-awq-compressed \
    --load-format compressed \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.7 \
    --cpu-offload-gb 2 \
    --enforce-eager \
    --dtype float16 \
    --model-loader-extra-config '{"enable_jit_decompress":true,"gpu_decompress":true,"prefetch_layers":1}'
```

**Terminal 2 — monitor memory:**
```bash
watch -n1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; free -h | grep Mem'
```

Note down peak GPU VRAM and CPU RAM.

---

## Step 3 — Benchmark throughput

Run this against whichever server is currently running. Repeat for both scenarios.

**Throughput benchmark (50 requests, random 128→128 tokens):**
```bash
PYTHONPATH=/workspace python benchmarks/benchmark_serving.py \
    --backend openai \
    --base-url http://localhost:8000 \
    --model /models/your-model-awq \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 50 \
    --request-rate 4
```

Key metrics to record:
- `Output token throughput (tok/s)`
- `Median TTFT (ms)` — time to first token
- `Median ITL (ms)` — inter-token latency

**Single prompt test (quick sanity check):**
```bash
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/models/your-model-awq",
        "prompt": "Explain the difference between LZ4 and GDeflate compression in detail.",
        "max_tokens": 200,
        "temperature": 0
    }' | python3 -c "
import json, sys
d = json.load(sys.stdin)
u = d['usage']
print(f'Prompt: {u[\"prompt_tokens\"]} tokens')
print(f'Output: {u[\"completion_tokens\"]} tokens')
print(d['choices'][0]['text'][:300])
"
```

---

## Results table

Fill in after running both scenarios:

| Metric | Baseline | JIT (gdeflate + GPU decompress) |
|--------|----------|----------------------------------|
| GPU VRAM at startup | _ GB | _ GB |
| CPU RAM at startup | _ GB | _ GB |
| Output tok/s | _ | _ |
| Median TTFT (ms) | _ | _ |
| Median ITL (ms) | _ | _ |
