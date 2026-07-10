# Qwen3.6-27B model comparison — benchmarks

Setup: 1× RTX 5090 (32 GB) + 2× RTX 3080 (20 GB), TP=4 with
`--rank-gpu-id 0,0,1,2` (2 ranks on the 5090, CUDA MPS active),
`--rank-gpu-memory-mib 15500`*, `--max-model-len -1` (auto-fit),
`--max-num-seqs 8 --max-num-batched-tokens 8192`,
`--mamba-cache-mode align --enable-prefix-caching --dtype bfloat16`,
MTP speculative decoding (`num_speculative_tokens: 3`).
vLLM fork `qwen35-gguf-rankgpu-test` + vllm-gguf-plugin `qwen35-support`.

\* Exception Q4_K_M: ran with 12000 MiB/rank (an already-loaded instance was
reused); speed numbers are unaffected, the max-KV value was re-measured
separately with 15500 MiB.

Prefill tests: unique German text, gen=60.
Multiturn: 5 rounds of ~13.2k new prompt tokens each, min. 300 generated.

KV cache dtype: from Q8_0 onwards with `--kv-cache-dtype fp8`; the first two
runs (Q4_K_M, Q6_K) still used auto/bf16 KV — their KV capacity is halved
accordingly, speed numbers are barely affected.

`--max-num-seqs`: the Q4/Q6/Q8/AWQ runs used `--max-num-seqs 8`; the intended
value for these single-stream tests is `1` — only the FP8 AEON run uses
`--max-num-seqs 1`. Since the suite only ever issues one request at a time,
this mainly affects CUDA-graph capture sizes/scheduler overhead, not the
measurement methodology.

## Launch commands

Common base (worktree `/spinning/shvllm-gguf-mtp`, venv python, MPS via
`nvidia-cuda-mps-control -d`):

```bash
python -m vllm.entrypoints.openai.api_server \
  --tensor-parallel-size 4 --rank-gpu-id 0,0,1,2 \
  --max-model-len -1 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --mamba-cache-mode align --enable-prefix-caching --dtype bfloat16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8003 <model-specific arguments>
```

Model-specific:

| Model | `--model` | `--rank-gpu-memory-mib` | KV dtype |
|---|---|---|---|
| GGUF Q4_K_M | `.../Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/...Q4_K_M.gguf` | 12000* | auto (bf16) |
| GGUF Q6_K | `.../Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/...Q6_K.gguf` | 15500 | auto (bf16) |
| GGUF Q8_0 | `.../huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF/...Q8_0.gguf` | 14500** | `--kv-cache-dtype fp8` |
| AWQ INT4 | `.../Qwen3.6-27B-AWQ-BF16-INT4` + `--quantization compressed-tensors` | 15500 | `--kv-cache-dtype fp8` |
| FP8 AEON | `.../Qwen3.6-27B-AEON-Ultimate-Uncensored-FP8-MTP` | 14500** | `--kv-cache-dtype fp8` |

\* already-loaded instance reused (see above).
\** 2×15500 MiB on the 5090 (31.34 GiB usable) left no runtime headroom at
all for the large models (~7.3 GiB weights/rank) → OOM on the first large
prefill chunk (15000 was still too tight); 14500 runs stably (~2.3 GiB
slack).
Local GGUF files need `config.json` + tokenizer/processor files in the same
directory (taken from the AWQ repo here).

Vision encoder: Q4/Q6 (mmproj-BF16 in the directory), AWQ and FP8 ran as
multimodal `Qwen3_5ForConditionalGeneration` incl. loaded vision tower;
**only Q8_0 ran text-only** (no mmproj file in the Huihui directory). This
has practically no effect on the text speed numbers (the vision tower is
idle for text requests), but the Q8 run's KV capacity is slightly favored
by it (~0.5–1 GB/rank more budget free).

## GGUF Q4_K_M (heretic-v2) — KV cache bf16

- max_model_len: **182400** (auto-fit: 182400)
- KV cache: 3.16 GiB/rank, capacity 182,400 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 0.12 s | 382 | 57.8 |
| Code 800 | 48 | 800 | 0.13 s | 374 | 81.2 |
| Prefill 5k | 5175 | 60 | 3.72 s | 1390 | 85.7 |
| Prefill 15k | 15993 | 60 | 11.72 s | 1365 | 62.5 |
| Prefill 50k | 53743 | 60 | 40.50 s | 1327 | 30.2 |
| Prefill 100k | 109724 | 60 | 91.23 s | 1203 | 15.1 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13588 | 10.17 s | 1337 | 54.8 |
| Turn 2 | 27172 | 16.27 s | 835 | 13.9 |
| Turn 3 | 40759 | 12.60 s | 1079 | 16.7 |
| Turn 4 | 54348 | 12.56 s | 1082 | 14.1 |
| Turn 5 | 67939 | 13.17 s | 1032 | 12.8 |

## GGUF Q6_K (heretic-v2) — KV cache bf16

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 5.96 GiB/rank, capacity 337,806 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 1.55 s | 30 | 51.8 |
| Code 800 | 48 | 800 | 0.10 s | 470 | 80.0 |
| Prefill 5k | 5110 | 60 | 3.80 s | 1344 | 78.2 |
| Prefill 15k | 15798 | 60 | 11.07 s | 1427 | 54.0 |
| Prefill 50k | 53098 | 60 | 39.77 s | 1335 | 29.4 |
| Prefill 100k | 108434 | 60 | 89.53 s | 1211 | 15.3 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13418 | 9.79 s | 1370 | 54.3 |
| Turn 2 | 26832 | 11.11 s | 1208 | 37.0 |
| Turn 3 | 40249 | 11.48 s | 1168 | 31.5 |
| Turn 4 | 53668 | 11.90 s | 1128 | 27.3 |
| Turn 5 | 67089 | 12.26 s | 1094 | 22.2 |

## GGUF Q8_0 (Huihui abliterated) — KV fp8, 14500 MiB/rank

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 3.78 GiB/rank, capacity 434,002 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 1.82 s | 25 | 43.4 |
| Code 800 | 48 | 800 | 0.14 s | 336 | 72.4 |
| Prefill 5k | 6280 | 60 | 4.67 s | 1345 | 73.4 |
| Prefill 15k | 19308 | 60 | 13.85 s | 1394 | 72.5 |
| Prefill 50k | 64708 | 60 | 50.88 s | 1272 | 63.5 |
| Prefill 100k | 131654 | 60 | 117.36 s | 1122 | 69.2 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 16478 | 11.79 s | 1398 | 70.6 |
| Turn 2 | 32952 | 13.75 s | 1198 | 71.7 |
| Turn 3 | 49429 | 14.58 s | 1130 | 72.2 |
| Turn 4 | 65908 | 15.49 s | 1064 | 65.0 |
| Turn 5 | 82389 | 16.26 s | 1013 | 70.6 |

## AWQ BF16-INT4 (compressed-tensors) — KV fp8

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 5.66 GiB/rank, capacity 619,479 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 2.21 s | 21 | 65.7 |
| Code 800 | 48 | 800 | 0.09 s | 526 | 115.6 |
| Prefill 5k | 5565 | 60 | 4.15 s | 1342 | 125.6 |
| Prefill 15k | 17163 | 60 | 12.03 s | 1427 | 123.4 |
| Prefill 50k | 57613 | 60 | 44.11 s | 1306 | 114.1 |
| Prefill 100k | 117464 | 60 | 101.42 s | 1158 | 110.3 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 14608 | 10.39 s | 1406 | 111.2 |
| Turn 2 | 29212 | 12.32 s | 1185 | 114.7 |
| Turn 3 | 43819 | 13.26 s | 1101 | 106.8 |
| Turn 4 | 58428 | 14.21 s | 1028 | 107.1 |
| Turn 5 | 73039 | 15.07 s | 969 | 112.8 |

### AWQ BF16-INT4 — 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt total | Gen total | Aggregate gen t/s | TTFT avg | Decode/req avg |
|---|---|---|---|---|---|---|---|
| Prose 8×400 | 8 | 11.2 s | 384 | 3200 | **286.7** | 2.17 s | 47.5 |
| Code 8×400 | 8 | 7.9 s | 336 | 3200 | **407.3** | 0.31 s | 55.8 |
| Prefill 8×10k | 8 | 60.2 s | 83824 | 480 | **8.0** | 37.25 s | 12.8 |

## FP8 AEON Ultimate (max-num-seqs 1)

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 4.12 GiB/rank, capacity 464,243 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 2.22 s | 21 | 59.5 |
| Code 800 | 48 | 800 | 13.33 s | 4 | 97.0 |
| Prefill 5k | 5240 | 60 | 12.14 s | 432 | 105.4 |
| Prefill 15k | 16188 | 60 | 13.48 s | 1201 | 104.7 |
| Prefill 50k | 54388 | 60 | 43.58 s | 1248 | 95.9 |
| Prefill 100k | 111014 | 60 | 98.83 s | 1123 | 93.4 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13758 | 12.94 s | 1063 | 96.6 |
| Turn 2 | 27512 | 18.30 s | 751 | 94.4 |
| Turn 3 | 41269 | 18.32 s | 751 | 93.5 |
| Turn 4 | 55028 | 20.46 s | 673 | 87.3 |
| Turn 5 | 68789 | 20.67 s | 666 | 89.2 |

## FP8 AEON Ultimate (max-num-seqs 1) — repeat run

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 4.12 GiB/rank, capacity 464,243 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 14.43 s | 3 | 60.9 |
| Code 800 | 48 | 800 | 8.45 s | 6 | 98.9 |
| Prefill 5k | 5240 | 60 | 5.97 s | 878 | 105.7 |
| Prefill 15k | 16188 | 60 | 13.38 s | 1210 | 104.5 |
| Prefill 50k | 54388 | 60 | 45.03 s | 1208 | 95.7 |
| Prefill 100k | 111014 | 60 | 98.56 s | 1126 | 93.4 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13758 | 15.85 s | 868 | 96.4 |
| Turn 2 | 27512 | 18.03 s | 763 | 91.3 |
| Turn 3 | 41269 | 19.03 s | 723 | 96.2 |
| Turn 4 | 55028 | 20.54 s | 670 | 87.8 |
| Turn 5 | 68789 | 21.55 s | 639 | 91.0 |

## AWQ BF16-INT4 — AFTER the heterogeneous-TP fix (af798f32f)

**Important context for all results above:** all earlier runs were affected
by a correctness bug found later: with mixed GPU architectures in one TP
group (5090 + 3080s via `--rank-gpu-id`), the redundant per-rank sampling
sporadically diverged (~1/1000 tokens), which escalated into repetition
loops in long outputs. The **speed** numbers above remain valid (the bug
cost no time), but the **output quality** of those runs was degraded.
Fix: broadcast of the sampled and MTP-draft tokens from rank 0 (commit
af798f32f), overhead <1 ms/step.

Setup as above, but: 14000 MiB/rank, `--max-num-batched-tokens 4096`, and a
Docker image build was running IN PARALLEL on the CPU (TTFT/prefill slightly
conservative as a result). max_model_len 262144 (no auto-fit limit),
KV cache 4.3 GiB/rank = 456,921 tokens capacity.

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 0.09 s | 514 | 73.4 |
| Code 800 | 48 | 800 | 0.09 s | 525 | 108.1 |
| Prefill 5k | 6215 | 60 | 4.45 s | 1396 | 124.2 |
| Prefill 15k | 19113 | 60 | 13.51 s | 1414 | 122.8 |
| Prefill 50k | 64063 | 60 | 49.25 s | 1301 | 105.4 |
| Prefill 100k | 130364 | 60 | 114.50 s | 1139 | 102.7 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 16308 | 11.87 s | 1374 | 109.2 |
| Turn 2 | 32612 | 13.62 s | 1197 | 118.4 |
| Turn 3 | 48919 | 15.61 s | 1045 | 93.7 |
| Turn 4 | 65228 | 16.89 s | 965 | 91.0 |
| Turn 5 | 81539 | 18.21 s | 896 | 96.5 |

Quality validation (systematic for the first time): 8/8 (2B) + 4/4 (27B LXC)
+ 4/4 (27B Docker/host) long generations without loops/doubled words;
2245-word prose clean; generated 164-line Python code compiles without
errors.

---

# Benchmark marathon (Docker image with hetero-TP fix, 2026-07-10)

All models: TP=4, `--rank-gpu-id 0,0,1,2` @ 14000 MiB/rank,
`--max-num-seqs 8`, `--max-num-batched-tokens 4096`, fp8 KV cache, MTP k=3,
prefix caching, `--mamba-cache-mode align`, `--max-model-len -1`.
Containers on the host, measured from the LXC (local bridge, TTFT
overhead <1 ms).

> **Note:** The "Prose 800" row of every model shows a TTFT of ~47 s — that
> is the JIT warmup of the freshly started container on its very first
> request (Triton/FlashInfer kernel compilation; cache volume not mounted),
> not a real prefill value. The decode rate of that row remains valid.

## Marathon summary

| Model | Decode Code | Decode Prose | Decode @100k | 8× Prose agg. | 8× Code agg. | KV capacity |
|---|---|---|---|---|---|---|
| AWQ BF16-INT4 | 108.8 | 71.2 | 111.4 | 301 | 426 | 453k |
| FP8 AEON | 112.5 | 67.4 | 108.6 | 303 | 414 | 378k |
| FP8 Original | 110.9 | 66.9 | 108.6 | 291 | 405 | 373k |
| GGUF Q4_K_M | 81.1 | 51.3 | 70.4 | 199 | 244 | 592k |
| GGUF Q8_0 \* | 80.4 | 48.5 | 75.7 | 187 | 240 | 303k |
| GGUF Q6_K | 69.3 | 45.3 | 63.7 | 197 | 246 | 485k |

\* Q8_0 ran WITH the vision tower this time (mmproj), hence less free KV
than in the earlier text-only run. All models ran multimodal, all with fp8
KV — which is why the GGUFs now hold their decode rate even at 100k context
(previously ~15 t/s with bf16 KV).

## GGUF Q4_K_M (heretic-v2)

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 5.25 GiB/rank, capacity 591,654 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 48.52 s | – | 51.3 |
| Code 800 | 48 | 800 | 0.13 s | 368 | 81.1 |
| Prefill 5k | 5175 | 60 | 3.70 s | 1398 | 90.1 |
| Prefill 15k | 15993 | 60 | 11.40 s | 1403 | 89.6 |
| Prefill 50k | 53743 | 60 | 41.40 s | 1298 | 82.0 |
| Prefill 100k | 109724 | 60 | 94.98 s | 1155 | 70.4 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13588 | 9.77 s | 1391 | 81.1 |
| Turn 2 | 27172 | 12.00 s | 1132 | 72.6 |
| Turn 3 | 40759 | 13.44 s | 1011 | 79.4 |
| Turn 4 | 54348 | 13.48 s | 1008 | 77.1 |
| Turn 5 | 67939 | 15.10 s | 900 | 79.0 |

### 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt total | Gen total | Aggregate gen t/s | TTFT avg | Decode/req avg |
|---|---|---|---|---|---|---|---|
| Prose 8×400 | 8 | 16.0 s | 384 | 3200 | **199.4** | 1.71 s | 29.5 |
| Code 8×400 | 8 | 13.1 s | 336 | 3200 | **244.1** | 0.35 s | 32.9 |
| Prefill 8×10k | 8 | 68.3 s | 93184 | 480 | **7.0** | 39.01 s | 8.4 |

## GGUF Q6_K (heretic-v2)

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 4.36 GiB/rank, capacity 484,746 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 48.08 s | – | 45.3 |
| Code 800 | 48 | 800 | 0.10 s | 489 | 69.3 |
| Prefill 5k | 5110 | 60 | 3.66 s | 1398 | 76.8 |
| Prefill 15k | 15798 | 60 | 11.21 s | 1409 | 67.6 |
| Prefill 50k | 53098 | 60 | 40.63 s | 1307 | 70.0 |
| Prefill 100k | 108434 | 60 | 92.90 s | 1167 | 63.7 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13418 | 9.64 s | 1392 | 69.5 |
| Turn 2 | 26832 | 11.69 s | 1148 | 67.4 |
| Turn 3 | 40249 | 13.02 s | 1030 | 68.6 |
| Turn 4 | 53668 | 12.88 s | 1042 | 67.3 |
| Turn 5 | 67089 | 14.15 s | 949 | 66.6 |

### 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt total | Gen total | Aggregate gen t/s | TTFT avg | Decode/req avg |
|---|---|---|---|---|---|---|---|
| Prose 8×400 | 8 | 16.3 s | 384 | 3200 | **196.6** | 1.29 s | 28.6 |
| Code 8×400 | 8 | 13.0 s | 336 | 3200 | **245.5** | 0.36 s | 32.8 |
| Prefill 8×10k | 8 | 67.5 s | 92144 | 480 | **7.1** | 38.53 s | 7.9 |

## GGUF Q8_0 (Huihui)

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 2.86 GiB/rank, capacity 303,149 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 47.24 s | – | 48.5 |
| Code 800 | 48 | 800 | 0.10 s | 484 | 80.4 |
| Prefill 5k | 5110 | 60 | 3.65 s | 1398 | 85.2 |
| Prefill 15k | 15798 | 60 | 11.24 s | 1406 | 79.4 |
| Prefill 50k | 53098 | 60 | 40.70 s | 1305 | 82.7 |
| Prefill 100k | 108434 | 60 | 93.09 s | 1165 | 75.7 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13418 | 9.64 s | 1392 | 76.7 |
| Turn 2 | 26832 | 11.67 s | 1149 | 81.6 |
| Turn 3 | 40249 | 13.03 s | 1030 | 78.1 |
| Turn 4 | 53668 | 12.87 s | 1042 | 73.4 |
| Turn 5 | 67089 | 14.15 s | 948 | 72.9 |

### 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt total | Gen total | Aggregate gen t/s | TTFT avg | Decode/req avg |
|---|---|---|---|---|---|---|---|
| Prose 8×400 | 8 | 17.1 s | 384 | 3200 | **187.3** | 1.74 s | 28.2 |
| Code 8×400 | 8 | 13.4 s | 336 | 3200 | **239.5** | 0.40 s | 31.5 |
| Prefill 8×10k | 8 | 67.4 s | 92144 | 480 | **7.1** | 38.60 s | 8.8 |

## AWQ BF16-INT4

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 4.23 GiB/rank, capacity 452,527 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 47.78 s | – | 71.2 |
| Code 800 | 48 | 800 | 0.07 s | 735 | 108.8 |
| Prefill 5k | 4980 | 60 | 3.48 s | 1431 | 126.9 |
| Prefill 15k | 15408 | 60 | 10.79 s | 1428 | 111.1 |
| Prefill 50k | 51808 | 60 | 39.29 s | 1319 | 124.1 |
| Prefill 100k | 105854 | 60 | 89.85 s | 1178 | 111.4 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13078 | 9.24 s | 1415 | 114.0 |
| Turn 2 | 26152 | 11.05 s | 1183 | 114.9 |
| Turn 3 | 39229 | 11.96 s | 1094 | 110.3 |
| Turn 4 | 52308 | 12.88 s | 1016 | 113.9 |
| Turn 5 | 65389 | 13.86 s | 944 | 105.6 |

### 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt total | Gen total | Aggregate gen t/s | TTFT avg | Decode/req avg |
|---|---|---|---|---|---|---|---|
| Prose 8×400 | 8 | 10.6 s | 384 | 3200 | **301.2** | 1.66 s | 48.4 |
| Code 8×400 | 8 | 7.5 s | 336 | 3200 | **425.6** | 0.30 s | 58.4 |
| Prefill 8×10k | 8 | 63.8 s | 89024 | 480 | **7.5** | 35.96 s | 13.9 |

## FP8 AEON Ultimate

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 3.36 GiB/rank, capacity 377,838 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 47.07 s | – | 67.4 |
| Code 800 | 48 | 800 | 0.09 s | 534 | 112.5 |
| Prefill 5k | 4785 | 60 | 3.37 s | 1419 | 121.7 |
| Prefill 15k | 14823 | 60 | 10.39 s | 1427 | 121.2 |
| Prefill 50k | 49873 | 60 | 37.65 s | 1325 | 126.5 |
| Prefill 100k | 101984 | 60 | 86.07 s | 1185 | 108.6 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 12568 | 8.85 s | 1421 | 114.3 |
| Turn 2 | 25132 | 11.45 s | 1098 | 112.2 |
| Turn 3 | 37699 | 11.99 s | 1048 | 107.9 |
| Turn 4 | 50268 | 12.50 s | 1006 | 103.6 |
| Turn 5 | 62839 | 12.91 s | 973 | 105.4 |

### 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt total | Gen total | Aggregate gen t/s | TTFT avg | Decode/req avg |
|---|---|---|---|---|---|---|---|
| Prose 8×400 | 8 | 10.6 s | 384 | 3200 | **303.1** | 1.65 s | 48.0 |
| Code 8×400 | 8 | 7.7 s | 336 | 3200 | **413.7** | 0.31 s | 55.9 |
| Prefill 8×10k | 8 | 61.5 s | 85904 | 480 | **7.8** | 34.67 s | 13.0 |

## FP8 Original (Qwen)

- max_model_len: **262144** (auto-fit: no limit, full 262144)
- KV cache: 3.53 GiB/rank, capacity 373,445 tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prose 800 | 46 | 800 | 46.17 s | – | 66.9 |
| Code 800 | 48 | 800 | 0.09 s | 526 | 110.9 |
| Prefill 5k | 4850 | 60 | 3.42 s | 1420 | 121.8 |
| Prefill 15k | 15018 | 60 | 10.57 s | 1421 | 114.5 |
| Prefill 50k | 50518 | 60 | 38.21 s | 1322 | 111.5 |
| Prefill 100k | 103274 | 60 | 87.72 s | 1177 | 108.6 |

| Multiturn | Context | TTFT | New-prefill t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 12738 | 8.98 s | 1419 | 107.7 |
| Turn 2 | 25472 | 11.72 s | 1087 | 106.5 |
| Turn 3 | 38209 | 12.42 s | 1026 | 107.8 |
| Turn 4 | 50948 | 13.10 s | 972 | 111.9 |
| Turn 5 | 63689 | 13.65 s | 934 | 105.4 |

### 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt total | Gen total | Aggregate gen t/s | TTFT avg | Decode/req avg |
|---|---|---|---|---|---|---|---|
| Prose 8×400 | 8 | 11.0 s | 384 | 3200 | **291.0** | 1.69 s | 47.7 |
| Code 8×400 | 8 | 7.9 s | 336 | 3200 | **405.3** | 0.31 s | 54.7 |
| Prefill 8×10k | 8 | 63.1 s | 87984 | 480 | **7.6** | 35.54 s | 12.2 |
