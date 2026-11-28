# Multimodal input preprocessing

> Community sprint note 024/120 · dependency-free design note · vLLM community working group

## Scope

Images become token sequences with their own preprocessing cost, cache behaviour and memory shape. Serving vision-language models is a scheduling problem as much as a model problem.

## Mechanism

1. Images are resized to a patch grid and encoded by a vision tower into embeddings.
2. Embeddings are spliced into the prompt at placeholder positions.
3. Encoded embeddings are cached by content hash so repeated images skip the encoder.
4. Resolution decides both encoder time and the number of tokens the KV cache must hold.

## Knobs

| knob | default | what it changes |
| --- | --- | --- |
| `image_tokens` | `576.0` | Prompt tokens one image expands to. |
| `patch_size` | `14.0` | Pixel size of one patch. |
| `encoder_ms` | `45.0` | Vision tower time per image on the serving GPU. |
| `cache_mb` | `512.0` | Embedding cache budget. |

## Failure modes

- **Sizing the KV pool for text only, then OOMing on the first multi-image request.**
- **Re-encoding the same product image on every request.**
- **Resolution chosen for quality without measuring the encoder's share of TTFT.**

## Instrumentation

Watch `encoder_time_pct`, `embedding_cache_hit_rate` and `image_tokens_per_request` before and after every change in this area. If a change
moves throughput without moving one of those three, the measurement is probably noise.

## Reproduction shape

```bash
# illustrative only: substitute your own model, hardware and workload shape
vllm serve MODEL --image_tokens 576.0 --gpu-memory-utilization 0.90
```

## Discussion

The model side of this note lives in `community/src/vllm_community/multimodal-preprocessing.py`, its tests in
`community/tests/test_multimodal-preprocessing.py`, and a micro-benchmark in
`community/benchmarks/bench_multimodal-preprocessing.py`. Treat the numbers there as ratios to be re-measured
on your own hardware, never as absolute throughput claims.
