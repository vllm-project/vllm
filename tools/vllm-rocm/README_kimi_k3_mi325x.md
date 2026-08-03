# Serving Kimi-K3 on 8x MI325X (gfx942)

Notes from bringing `moonshotai/Kimi-K3` up on CDNA3. The model is
~2.75 T parameters, MXFP4 (`mxfp4-pack-quantized`), ~1.5 TB on disk.

## Parallelism is forced

K3 has 96 attention heads and 1.5 TB of weights against 2 TB of HBM.

| TP | heads/rank | weights/GPU | viable |
| -- | ---------- | ----------- | ------ |
| 8  | 12         | ~195 GB     | yes    |
| 6  | 16         | ~260 GB     | no, exceeds 256 GB (and 7168 % 6 != 0) |
| 4  | 24         | ~390 GB     | no     |

TP=8 is the only configuration that fits, and it puts fewer than 16 heads
per rank, which is why the AITER MLA small-head padding path matters here.

## Serving

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve moonshotai/Kimi-K3 \
  --served-model-name kimi-k3 \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 512 \
  --limit-mm-per-prompt.image 1 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --enable-auto-tool-choice
```

`--max-num-seqs` must stay under the number of available Mamba/KDA state
blocks. K3 has 69 KDA layers and every concurrent decode sequence needs one
state block, so the default 1024 fails CUDA graph capture:

```
ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (854)
```

At `--gpu-memory-utilization 0.97` this yields ~1.39 M tokens of KV cache,
about 10.6x concurrency at 128 k context.

## Client-side settings that matter for quality

K3 is a reasoning model. The `kimi_k3` reasoning parser *strips* the think
channel rather than surfacing it, and it strips on the closing marker. If
`max_tokens` truncates generation before that marker is emitted, the raw
chain-of-thought is returned as `content` instead of an answer.

The template defaults to `thinking_effort=max`, which is both the most
expensive and the least reliable setting for tool calls. Measured tool-call
emission, 6 trials per cell:

| schema  | default (max) | low | high | thinking off |
| ------- | ------------- | --- | ---- | ------------ |
| 1-param | 6/6           | 6/6 | 6/6  | 0/6          |
| 2-param | 3/6           | 6/6 | 6/6  | 3/6          |
| 3-param | 6/6           | 6/6 | 6/6  | 0/6          |

Recommended: `chat_template_kwargs={"thinking_effort": "low"}`. It is 18/18
on tool calls and ~3.5x cheaper in tokens than the default (44 vs 167 tokens
for `17 * 23`; 115 vs 432 for a two-sentence comparison).

Do **not** set `{"thinking": false}` if you use tools -- it is the worst
configuration measured. Valid efforts are `low`, `high`, `max`; the template
advertises `medium` in its own system message but the validator rejects it.

## CDNA3 MoE tile shapes

AITER's a16w4 tile selection is architecture-blind and sized for CDNA4's
160 KiB LDS. On gfx942 (64 KiB) that costs ~1.9x end-to-end prefill. See
`patch_aiter_cdna3_moe_tiles.py` in this directory and
`benchmarks/kernels/benchmark_moe_a16w4_tiles.py`.
