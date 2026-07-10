# Qwen3.6-27B model comparison — benchmarks

Hardware: 1× RTX 5090 (32 GB) + 2× RTX 3080 (20 GB). All runs use the
prebuilt Docker image `ghcr.io/efschu/shvllm-qwen35-gguf:cu129` (vLLM fork
incl. the heterogeneous-TP fix `af798f32f` + vllm-gguf-plugin
`qwen35-support`).

> Earlier (pre-fix) measurements were removed from this file — they were
> taken while a correctness bug degraded long outputs on mixed-architecture
> TP groups (redundant per-rank sampling diverging ~1/1000 tokens). Their
> speed numbers were valid but are superseded by the marathon below; see
> git history if needed.

Quality validation of the fixed build: 8/8 (2B) + 4/4 (27B LXC) + 4/4
(27B Docker/host) long generations without loops or doubled words;
2245-word prose clean; generated 164-line Python code compiles.

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

### MTP acceptance (from surviving server logs)

The marathon logs were captured before the benchmark requests ran, so they
contain no SpecDecoding metrics. Two post-fix logs survived:

| Model | Acceptance | Mean acceptance length | Drafted tokens | Workload |
|---|---|---|---|---|
| AWQ BF16-INT4 | 58.0 % | 2.74 | 29,829 | full benchmark suite (prose/code/prefill/multiturn) |
| GGUF Q6_K | 98.1 % | 3.94 | 3,999 | short interactive session (small sample, not comparable) |

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
