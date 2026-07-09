# Qwen3.6-27B Modellvergleich — Benchmarks

Setup: 1× RTX 5090 (32 GB) + 2× RTX 3080 (20 GB), TP=4 mit
`--rank-gpu-id 0,0,1,2` (2 Ranks auf der 5090, CUDA MPS aktiv),
`--rank-gpu-memory-mib 15500`*, `--max-model-len -1` (Auto-Fit),
`--max-num-seqs 8 --max-num-batched-tokens 8192`,
`--mamba-cache-mode align --enable-prefix-caching --dtype bfloat16`,
MTP-Speculative-Decoding (`num_speculative_tokens: 3`).
vLLM-Fork `qwen35-gguf-rankgpu-test` + vllm-gguf-plugin `qwen35-support`.

\* Ausnahme Q4_K_M: lief mit 12000 MiB/Rank (bereits geladene Instanz
weiterverwendet); Speed-Werte sind davon unabhängig, der KV-Max-Wert
wurde separat mit 15500 MiB nachgemessen.

Prefill-Tests: einzigartiger deutscher Text, gen=60.
Multiturn: 5 Runden à ~13.2k neue Prompt-Tokens, min. 300 gen.

KV-Cache-Dtype: ab Q8_0 mit `--kv-cache-dtype fp8`; die ersten beiden
Läufe (Q4_K_M, Q6_K) liefen noch mit auto/bf16-KV — dort ist die
KV-Kapazität entsprechend halbiert, Speed-Werte sind kaum betroffen.

`--max-num-seqs`: Die Läufe Q4/Q6/Q8/AWQ liefen mit `--max-num-seqs 8`;
eigentlich vorgesehen für diese Single-Stream-Tests ist `1` — erst der
FP8-AEON-Lauf nutzt `--max-num-seqs 1`. Da die Suite immer nur einen
Request gleichzeitig stellt, wirkt sich das primär auf CUDA-Graph-
Capture-Größen/Scheduler-Overhead aus, nicht auf die Messmethodik.

## Startkommandos

Gemeinsame Basis (Worktree `/spinning/shvllm-gguf-mtp`, venv-Python,
MPS via `nvidia-cuda-mps-control -d`):

```bash
python -m vllm.entrypoints.openai.api_server \
  --tensor-parallel-size 4 --rank-gpu-id 0,0,1,2 \
  --max-model-len -1 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --mamba-cache-mode align --enable-prefix-caching --dtype bfloat16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8003 <modellspezifische Argumente>
```

Modellspezifisch:

| Modell | `--model` | `--rank-gpu-memory-mib` | KV-Dtype |
|---|---|---|---|
| GGUF Q4_K_M | `.../Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/...Q4_K_M.gguf` | 12000* | auto (bf16) |
| GGUF Q6_K | `.../Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF/...Q6_K.gguf` | 15500 | auto (bf16) |
| GGUF Q8_0 | `.../huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF/...Q8_0.gguf` | 14500** | `--kv-cache-dtype fp8` |
| AWQ INT4 | `.../Qwen3.6-27B-AWQ-BF16-INT4` + `--quantization compressed-tensors` | 15500 | `--kv-cache-dtype fp8` |
| FP8 AEON | `.../Qwen3.6-27B-AEON-Ultimate-Uncensored-FP8-MTP` | 14500** | `--kv-cache-dtype fp8` |

\* bereits geladene Instanz weiterverwendet (siehe oben).
\** 2×15500 MiB auf der 5090 (31.34 GiB nutzbar) ließ bei den großen
Modellen (~7.3 GiB Weights/Rank) keinerlei Runtime-Headroom → OOM beim
ersten großen Prefill-Chunk (auch 15000 war noch zu knapp); 14500 läuft
stabil (~2.3 GiB Luft).
Lokale GGUF-Dateien brauchen `config.json` + Tokenizer-/Prozessor-Dateien
im selben Verzeichnis (hier aus dem AWQ-Repo übernommen).

Vision-Encoder: Q4/Q6 (mmproj-BF16 im Verzeichnis), AWQ und FP8 liefen
als multimodale `Qwen3_5ForConditionalGeneration` inkl. geladenem
Vision-Tower; **nur Q8_0 lief text-only** (keine mmproj-Datei im
Huihui-Verzeichnis). Auf die Text-Speed-Werte hat das praktisch keinen
Einfluss (Vision-Tower ist bei Text-Requests inaktiv), aber die
KV-Kapazität des Q8-Laufs ist dadurch leicht begünstigt
(~0.5–1 GB/Rank mehr Budget frei).

## GGUF Q4_K_M (heretic-v2) — KV-Cache bf16

- max_model_len: **182400** (Auto-Fit: 182400)
- KV-Cache: 3.16 GiB/Rank, Kapazität 182,400 Tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prosa 800 | 46 | 800 | 0.12 s | 382 | 57.8 |
| Code 800 | 48 | 800 | 0.13 s | 374 | 81.2 |
| Prefill 5k | 5175 | 60 | 3.72 s | 1390 | 85.7 |
| Prefill 15k | 15993 | 60 | 11.72 s | 1365 | 62.5 |
| Prefill 50k | 53743 | 60 | 40.50 s | 1327 | 30.2 |
| Prefill 100k | 109724 | 60 | 91.23 s | 1203 | 15.1 |

| Multiturn | Kontext | TTFT | Prefill-neu t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13588 | 10.17 s | 1337 | 54.8 |
| Turn 2 | 27172 | 16.27 s | 835 | 13.9 |
| Turn 3 | 40759 | 12.60 s | 1079 | 16.7 |
| Turn 4 | 54348 | 12.56 s | 1082 | 14.1 |
| Turn 5 | 67939 | 13.17 s | 1032 | 12.8 |

## GGUF Q6_K (heretic-v2) — KV-Cache bf16

- max_model_len: **262144** (Auto-Fit: kein Auto-Fit-Limit (volle 262144))
- KV-Cache: 5.96 GiB/Rank, Kapazität 337,806 Tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prosa 800 | 46 | 800 | 1.55 s | 30 | 51.8 |
| Code 800 | 48 | 800 | 0.10 s | 470 | 80.0 |
| Prefill 5k | 5110 | 60 | 3.80 s | 1344 | 78.2 |
| Prefill 15k | 15798 | 60 | 11.07 s | 1427 | 54.0 |
| Prefill 50k | 53098 | 60 | 39.77 s | 1335 | 29.4 |
| Prefill 100k | 108434 | 60 | 89.53 s | 1211 | 15.3 |

| Multiturn | Kontext | TTFT | Prefill-neu t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13418 | 9.79 s | 1370 | 54.3 |
| Turn 2 | 26832 | 11.11 s | 1208 | 37.0 |
| Turn 3 | 40249 | 11.48 s | 1168 | 31.5 |
| Turn 4 | 53668 | 11.90 s | 1128 | 27.3 |
| Turn 5 | 67089 | 12.26 s | 1094 | 22.2 |

## GGUF Q8_0 (Huihui abliterated) — KV fp8, 14500 MiB/Rank

- max_model_len: **262144** (Auto-Fit: kein Auto-Fit-Limit (volle 262144))
- KV-Cache: 3.78 GiB/Rank, Kapazität 434,002 Tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prosa 800 | 46 | 800 | 1.82 s | 25 | 43.4 |
| Code 800 | 48 | 800 | 0.14 s | 336 | 72.4 |
| Prefill 5k | 6280 | 60 | 4.67 s | 1345 | 73.4 |
| Prefill 15k | 19308 | 60 | 13.85 s | 1394 | 72.5 |
| Prefill 50k | 64708 | 60 | 50.88 s | 1272 | 63.5 |
| Prefill 100k | 131654 | 60 | 117.36 s | 1122 | 69.2 |

| Multiturn | Kontext | TTFT | Prefill-neu t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 16478 | 11.79 s | 1398 | 70.6 |
| Turn 2 | 32952 | 13.75 s | 1198 | 71.7 |
| Turn 3 | 49429 | 14.58 s | 1130 | 72.2 |
| Turn 4 | 65908 | 15.49 s | 1064 | 65.0 |
| Turn 5 | 82389 | 16.26 s | 1013 | 70.6 |

## AWQ BF16-INT4 (compressed-tensors) — KV fp8

- max_model_len: **262144** (Auto-Fit: kein Auto-Fit-Limit (volle 262144))
- KV-Cache: 5.66 GiB/Rank, Kapazität 619,479 Tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prosa 800 | 46 | 800 | 2.21 s | 21 | 65.7 |
| Code 800 | 48 | 800 | 0.09 s | 526 | 115.6 |
| Prefill 5k | 5565 | 60 | 4.15 s | 1342 | 125.6 |
| Prefill 15k | 17163 | 60 | 12.03 s | 1427 | 123.4 |
| Prefill 50k | 57613 | 60 | 44.11 s | 1306 | 114.1 |
| Prefill 100k | 117464 | 60 | 101.42 s | 1158 | 110.3 |

| Multiturn | Kontext | TTFT | Prefill-neu t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 14608 | 10.39 s | 1406 | 111.2 |
| Turn 2 | 29212 | 12.32 s | 1185 | 114.7 |
| Turn 3 | 43819 | 13.26 s | 1101 | 106.8 |
| Turn 4 | 58428 | 14.21 s | 1028 | 107.1 |
| Turn 5 | 73039 | 15.07 s | 969 | 112.8 |

### AWQ BF16-INT4 — 8× parallel (`--max-num-seqs 8`)

| Test | n | Wall | Prompt ges. | Gen ges. | Aggregat gen t/s | TTFT ⌀ | Decode/Req ⌀ |
|---|---|---|---|---|---|---|---|
| Prosa 8x400 | 8 | 11.2 s | 384 | 3200 | **286.7** | 2.17 s | 47.5 |
| Code 8x400 | 8 | 7.9 s | 336 | 3200 | **407.3** | 0.31 s | 55.8 |
| Prefill 8x10k | 8 | 60.2 s | 83824 | 480 | **8.0** | 37.25 s | 12.8 |

## FP8 AEON Ultimate (max-num-seqs 1)

- max_model_len: **262144** (Auto-Fit: kein Auto-Fit-Limit (volle 262144))
- KV-Cache: 4.12 GiB/Rank, Kapazität 464,243 Tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prosa 800 | 46 | 800 | 2.22 s | 21 | 59.5 |
| Code 800 | 48 | 800 | 13.33 s | 4 | 97.0 |
| Prefill 5k | 5240 | 60 | 12.14 s | 432 | 105.4 |
| Prefill 15k | 16188 | 60 | 13.48 s | 1201 | 104.7 |
| Prefill 50k | 54388 | 60 | 43.58 s | 1248 | 95.9 |
| Prefill 100k | 111014 | 60 | 98.83 s | 1123 | 93.4 |

| Multiturn | Kontext | TTFT | Prefill-neu t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13758 | 12.94 s | 1063 | 96.6 |
| Turn 2 | 27512 | 18.30 s | 751 | 94.4 |
| Turn 3 | 41269 | 18.32 s | 751 | 93.5 |
| Turn 4 | 55028 | 20.46 s | 673 | 87.3 |
| Turn 5 | 68789 | 20.67 s | 666 | 89.2 |

## FP8 AEON Ultimate (max-num-seqs 1)

- max_model_len: **262144** (Auto-Fit: kein Auto-Fit-Limit (volle 262144))
- KV-Cache: 4.12 GiB/Rank, Kapazität 464,243 Tokens

| Test | Prompt | Gen | TTFT | Prefill t/s | Decode t/s |
|---|---|---|---|---|---|
| Prosa 800 | 46 | 800 | 14.43 s | 3 | 60.9 |
| Code 800 | 48 | 800 | 8.45 s | 6 | 98.9 |
| Prefill 5k | 5240 | 60 | 5.97 s | 878 | 105.7 |
| Prefill 15k | 16188 | 60 | 13.38 s | 1210 | 104.5 |
| Prefill 50k | 54388 | 60 | 45.03 s | 1208 | 95.7 |
| Prefill 100k | 111014 | 60 | 98.56 s | 1126 | 93.4 |

| Multiturn | Kontext | TTFT | Prefill-neu t/s | Decode t/s |
|---|---|---|---|---|
| Turn 1 | 13758 | 15.85 s | 868 | 96.4 |
| Turn 2 | 27512 | 18.03 s | 763 | 91.3 |
| Turn 3 | 41269 | 19.03 s | 723 | 96.2 |
| Turn 4 | 55028 | 20.54 s | 670 | 87.8 |
| Turn 5 | 68789 | 21.55 s | 639 | 91.0 |
