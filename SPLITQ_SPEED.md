# ROCM_SPLITQ: speed

Measured 2026-10-01 on the local box: 4× RX 7900 XTX (gfx1100), Qwen3.8-27B
W4A16 (`Q38V4`), tensor parallel 4, 8 GB of KV cache per GPU
(`--kv-cache-memory 8000000000`), `--max-num-seqs 6`,
`--max-num-batched-tokens 2048`, CUDA graphs on. MTP k=3 unless the row says
"no MTP". Same image and flags for every row; only `--kv-cache-dtype` /
`--attention-backend` (and the MTP flag) change.

SplitQ rows are the final format: `splitq_k3v3` (216 B per token and KV
head) and `splitq_k3v4` (248 B), codebook scale `x·x̂ = |x|²`.

## Capacity (tokens of KV cache that fit)

| KV cache | Bytes / token / KV head | Tokens (MTP) | Tokens (no MTP) |
| --- | --- | --- | --- |
| fp16 | 1024 | 445,825 | 481,237 |
| `turboquant_3bit_nc` | 198 | 2,053,571 | — |
| `turboquant_k3v4_nc` | 230 | 1,790,625 | 2,059,504 |
| `turboquant_4bit_nc` | 262 | — | 1,813,138 |
| **`splitq_k3v3`** | **216** | **1,897,520** | 2,185,964 |
| **`splitq_k3v4`** | **248** | **1,678,832** | — |

With MTP the draft layer also takes KV cache, hence the smaller counts.

## TTFT, `vllm bench serve`

Cold time to first token: `--dataset-name random` with a different seed for
each length (with one seed, prompts of different lengths share their prefix
and prefix caching hides part of the prefill), `--max-concurrency 1`, a 4k
warm-up request first, temperature 0.6, MTP on.

| KV cache | 100k (2 requests) | 380k |
| --- | --- | --- |
| fp16 (`TRITON_ATTN`) | 61.4 / 61.5 s | 371.0 s |
| `turboquant_k3v4_nc` | 154.8 / 59.9 s * | 620.6 s |
| **`splitq_k3v3_compact`** | **57.2 / 57.2 s** | **297.4 s** |
| `splitq_k3v3` | 56.7 / 56.7 s | 300.5 s |
| `splitq_k3v4` | 56.9 / 57.0 s | 300.4 s |

\* TurboQuant's first long request is slow, likely a Triton compile for new
shapes (not verified); its second request is the fair number.

At 380k SplitQ reaches the first token 2.1× faster than TurboQuant and 20%
faster than fp16. At 100k the prefill is bound by the model's GEMMs and
all-reduce, and the gap narrows to 5-7% over fp16.

Earlier rows in this file (100k 42.7 s, 380k 287 s) used one seed for every
length and were partly served from the prefix cache; they were dropped.

## Long document: prefill and decode at 380k

`long.py`: one 380k-token document, prefilled once with prompt logprobs
(for the NLL), then 400 tokens generated with the document cached.

| KV cache | Prefill 380k | Decode at 380k |
| --- | --- | --- |
| fp16 | 399.7 s | 2.7 tok/s |
| `turboquant_k3v4_nc` | 379.3 s | 177.5 tok/s * |
| `turboquant_3bit_nc` | 405.6 s | 176.8 tok/s * |
| `turboquant_4bit_nc`, no MTP | 811.8 s | 28.4 tok/s |
| **`splitq_k3v3`** | **375.8 s** | **129.6 tok/s** |
| **`splitq_k3v4`** | **375.5 s** | **151.4 tok/s** |

\* TurboQuant with MTP generates garbage ("…add 11111110…",
"Here\n\nHere\n\n…", GSM8K 0.5-1.2%), so the drafts are accepted at an
abnormal rate: these tok/s are not usable output. Without MTP TurboQuant is
correct but decodes at 28 tok/s at 380k. SplitQ decodes **4.6-5.3× faster
than usable TurboQuant** and **~50× faster than fp16** at this length
(fp16 has 446k tokens of cache for a 380k request and decodes from a
nearly full cache with one split per request).

Decode rows are one 400-token generation each (n=1): with MTP the tok/s
depends on the acceptance of that text, so treat them as ±15%.

## Per layer: `benchmarks/attention_benchmarks`

One attention layer of one TP4 shard of Qwen3.8-27B (6 query heads, 1 KV
head, head size 256, block size 16, fp16), one GPU. Decode rows use CUDA
graphs (`do_bench_cudagraph`); the MTP-verify and prefill rows run without
graphs for every backend (`--no-cuda-graphs`), because TurboQuant's
multi-token path copies between host and device and cannot be captured.
Times are per layer.

| Batch | fp16 (`TRITON_ATTN`) | `turboquant_k3v4_nc` | **`splitq_k3v3`** | **`splitq_k3v4`** |
| --- | --- | --- | --- | --- |
| decode, 8k | 63 µs | 92 µs | **25 µs** | 26 µs |
| decode, 32k | 238 µs | 273 µs | **63 µs** | 67 µs |
| decode, 100k | 639 µs | 744 µs | **173 µs** | 195 µs |
| decode, 380k | 2370 µs | 2565 µs | **614 µs** | 691 µs |
| decode, 6 × 100k | 2344 µs | 2408 µs | **694 µs** | 753 µs |
| MTP verify (4 tokens), 100k | 12.87 ms | 1.94 ms | **0.18 ms** | 0.20 ms |
| MTP verify (4 tokens), 380k | 48.77 ms | 6.65 ms | **0.62 ms** | 0.70 ms |
| prefill 2048 over 32k | 9.87 ms | 6.87 ms | **6.17 ms** | **6.17 ms** |
| prefill 2048 over 100k | 32.54 ms | 22.66 ms | **17.24 ms** | 17.26 ms |
| prefill 2048 over 380k | 125.6 ms | 86.6 ms | **63.9 ms** | 64.2 ms |

- Decode: SplitQ is **4.2× faster than TurboQuant and 3.9× faster than
  fp16** per layer at 380k: it reads 216-248 B per token instead of 230-1024
  and its WMMA kernel turns the codes into int8 with one `v_perm_b32`.
- MTP verification: **10.7× faster than TurboQuant** and 78× faster than
  `TRITON_ATTN`, whose 4-token path falls back to a kernel without KV
  splits (that is the 992 ms TPOT of fp16 at 380k with MTP).
- Prefill: **26% faster than TurboQuant and 49% faster than fp16** per layer
  at 380k (10-24% faster than TurboQuant from 32k up). TurboQuant expands the
  cached prefix to fp16 and runs flash attention; SplitQ reads the packed
  cache directly: 8 loader waves unpack 16-token tiles into LDS while 16
  compute waves run int8 WMMA for QK and fp16 WMMA for PV. On RDNA3 a WMMA
  and a VALU instruction share the SIMD, so the kernel is built to spend as
  little VALU per tile as possible (one block-table walk per tile, masking
  only on the last tile, the score scale and log2(e) in one FMA).
  The previous version of this kernel (2 loader waves) took 92.8 ms here.

Reproduce (one shard of the model above):

```bash
cd benchmarks/attention_benchmarks
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python benchmark.py \
    --backends ROCM_SPLITQ --kv-cache-dtype splitq_k3v3 --model <model> \
    --head-dim 256 --num-q-heads 6 --num-kv-heads 1 --block-size 16 \
    --num-layers 4 --batch-specs q1s8k q1s32k q1s100k q1s380k 6q1s100k
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python benchmark.py --no-cuda-graphs \
    --backends ROCM_SPLITQ --kv-cache-dtype splitq_k3v3 --model <model> \
    --head-dim 256 --num-q-heads 6 --num-kv-heads 1 --block-size 16 \
    --num-layers 4 --batch-specs q4s100k q4s380k q2ks32k q2ks100k q2ks380k
```

`vllm bench serve` TPOT with MTP depends on draft acceptance on random
tokens with `--ignore-eos` and moves a lot between requests, so it is not
used for kernel comparisons; the table above is.

## Kernel fidelity on real data

Real Q/K/V of every full-attention layer of the rank-0 shard, captured from
a 32k-token prefill. Error of the attention output against exact fp32
attention:

|  | Format error | Kernel vs format reference |
| --- | --- | --- |
| Prefill (2048-token chunk over 30k prefix), k3v3 | 6.31% | 0.20% |
| Prefill, k3v4 | 5.12% | 0.20% |
| Decode (last token over 32k), k3v3 | 13.43% | 0.02% |
| Decode, k3v4 | 9.84% | 0.02% |

The kernels add nothing measurable on top of the format.

## Reproducing

```bash
vllm serve <model> -tp 4 --kv-cache-dtype splitq_k3v3 \
    --kv-cache-memory 8000000000 --max-num-seqs 6 \
    --max-num-batched-tokens 2048 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}'
vllm bench serve --backend vllm --dataset-name random \
    --random-input-len 100000 --random-output-len 256 --random-range-ratio 0 \
    --num-prompts 2 --max-concurrency 1 --num-warmups 0 --ignore-eos --seed 7 \
    --temperature 0.6 --top-p 0.95 --top-k 20
```
