# GLM-5.3 latency-specialized decode (NVIDIA)

GLM-5.3 (`GlmMoeDsaForCausalLM`) serves low-batch decode at low latency on 8xB200
(SM100) and 8xB300 (SM103): steps of B requests of 1 + K tokens, B (1 + K) <= 16 rows
(up to 16 requests with MTP off, 4 at MTP K = 3). CuTe DSL kernels (`ops/cute_dsl`)
replace vLLM's FULL uniform decode graph of the 78 target layers. The MTP drafter
(layer 78) is vLLM's DeepSeek-V3.2 MTP model on vLLM's speculator. Every other
forward (prefill, mixed steps, eager steps) stays on vLLM's stock layers.

## Terms

- `q4`, `q8`: the four-row build (one request of up to four rows) and the eight-row
  W8A16 MoE; entry names carry them (`*_q4*`, `*_q8*`).
- `seam`: the MoE of a layer whose next layer reuses an earlier layer's index selection;
  it also writes the next layer's normalized input (`*_seam*` entries).
- `probe`: the attention worker of a layer with its own index selection, on a rank that
  also runs the selector; its weight forms are named `probe.*`.
- `ckpt_shared` (MoE entries): the MoE reads the checkpoint's FP8 experts in place and
  includes the shared expert.

## Layout

| Module | Role |
| --- | --- |
| `model.py` | The DSA decoder and `GlmMoeDsaForCausalLM` with the segment hooks, plus `Glm53ModelState` (Model Runner V2), which tracks requests and steps for the segment. |
| `switch.py` | Reads `decode_backend` once, in `GlmMoeDsaForCausalLM.__init__`, into a frozen `LowLatencyDecode` record; refuses unsupported deployments. |
| `decode.py` | `SpecializedDecode`, segment 1 (the target layers): build, launch, capture and the model hook. |
| `weights.py` | The segment's weight forms, built from the checkpoint during load. |
| `kv.py` | KV-cache hygiene between the segment and vLLM's stock forwards (index-key rank completion). |
| `arena.py` | The cross-rank peer arena and its placement (plain memory where the optional locality placement is unavailable). |
| `providers.py` | The segment's launch providers (prologue, selector, worker, FFN): each allocates its results and protocol state, names its peer-mailbox slices and fixes its launch's arguments. |
| `ops/cute_dsl/` | CuTe DSL kernel families (`registry.py`: served entry to kernel class, host sizes, compiled host wrapper and its object cache; `fused_moe.py` also owns the expert layout). |
| `ops/launch.py` | `Glm53Geometry`, the model's shapes and layer layout read from the config, and binding and running the compiled launches. |

The model class (`model.py`) owns the hooks: the target model's
forward offers each step to the installed segment, and `process_weights_after_loading`
builds the weight forms and installs the segment.
Runner-side hooks that are not model-specific stay in `vllm/v1/worker/gpu/`.

## Switch

`--additional-config '{"decode_backend": "latency_specialized"}'` turns on the
whole path, as Kimi K3's specialized decode does: the kernels, the weight forms,
and segment 1 (`auto`, the default, keeps vLLM's decode). With
the path on, startup refuses, on every TP rank (a MIN all-reduce over the TP group),
unless: the GPU is SM100 or SM103; tensor parallel size is 8 without pipeline
parallelism; the KV cache is FlashInfer MLA's 576-byte fp8 rows
(`--attention-backend FLASHINFER_MLA_SPARSE --kv-cache-dtype fp8`, vLLM's SM10x
default); the segment runs under Model Runner V2; decode steps run in FULL CUDA graphs
(not `--enforce-eager`).

## Deployment

Set a persistent `VLLM_CACHE_ROOT` (node-local, reused across restarts). It holds
the compiled CuTe host wrappers (`glm53_kernels/<entry>.<arch>.<key>.o`; a cold build is about 40 s per rank) and
DeepGEMM's JIT cache (`deep_gemm`, for the dense FP8 linears). A cold
node also needs `VLLM_ENGINE_READY_TIMEOUT_S=3600`, past vLLM's default 600 s.

```bash
export VLLM_CACHE_ROOT=/local/ssd/vllm-cache VLLM_ENGINE_READY_TIMEOUT_S=3600
vllm serve zai-org/GLM-5.3 --tensor-parallel-size 8 --block-size 64 \
  --attention-backend FLASHINFER_MLA_SPARSE --kv-cache-dtype fp8 \
  --max-num-seqs 16 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

- With MTP, add `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3,
  "use_local_argmax_reduction": true}'` and `--max-num-seqs 4` (K = 3: four rows per
  request). The drafter is vLLM's MTP; `use_local_argmax_reduction` takes its greedy drafts
  from a per-rank argmax. Benchmarks at a fixed acceptance length add vLLM's
  synthetic acceptance:
  `"rejection_sample_method": "synthetic", "synthetic_acceptance_length": 2.99`.
- The path serves the MoE from vLLM's `flashinfer_trtllm` experts (FlashInfer trtllm-gen's
  DeepSeekFp8 BlockMajorK form, stock's default on SM100), which the decode MoE kernels
  read in place (`fused_moe.LAYOUT`): vLLM's routed and shared expert tensors and its FP32
  routed scales stay where vLLM allocated them, and the segment adds only a 1152-byte header
  per layer (the shared expert's scales decoded to FP32). `--moe-backend`
  is not needed: the switch turns vLLM's `auto` into `flashinfer_trtllm` and refuses any
  other explicit backend, and vLLM's MTP drafter inherits it. The steps vLLM's stock
  forward runs (Batch shapes) read the same experts with stock's kernels.
- Without `--speculative-config` (MTP off) the segment claims vLLM's one-token
  decode graphs. One request runs as a T = 4 step: rows 1-3 repeat the token at
  p+1..p+3 as padding, vLLM's head samples row 0, and padding rows past p's block
  write vLLM's null block. Several requests run packed (Batch shapes).
- The KV pool needs no `--kv-cache-memory-bytes`: the segments built during
  vLLM's CUDA graph memory profiling are rebuilt over the pool vLLM binds after it.
- The startup log shows `GLM-5.3 latency-specialized decode: MTP K=1: B = [1]: the q4
  build; B = [2]: packed rows on tier 8 (q4 MoE, at most 4 live rows); B = [3, 4]:
  packed rows on tier 8 (q8 MoE, at most 8 live rows)` (one clause per CUDA graph;
  `--max-num-seqs 4` at K = 1) and `GLM-5.3 segment 1 installed (row tiers [4, 8, 16],
  claims [2, 4, 8, 16]-token forwards)`
  (K = 1, `--max-num-seqs` 8: the row tiers and the forward sizes the segment claims).

## Batch shapes

`switch.shape_path` claims every (MTP K, B requests) shape exactly once; the build, its
kernels, the CUDA graph sizes (one graph per run of request counts on the same build,
derived when `--cudagraph-capture-sizes` is left out) and the steps left to vLLM's stock
forward all follow from it. A request owns `1 + K` contiguous rows; a step has
`R = B (1 + K)` live rows.

| K | B = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9-16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Q4 | P8q4 | P8q4 | P8q4 | P8q8 | P8q8 | P8q8 | P8q8 | P16 |
| 1 | Q4 | P8q4 | P8q8 | P8q8 | P16 | P16 | P16 | P16 | S |
| 2 | Q4 | P8q8 | P16 | P16 | P16 | S | S | S | S |
| 3 | Q4 | A8 | A16 | A16 | S | S | S | S | S |
| 4 | L8 | P16 | P16 | S | S | S | S | S | S |
| 5 | L8 | P16 | S | S | S | S | S | S | S |
| 6 | L8 | P16 | S | S | S | S | S | S | S |
| 7 | L8 | A16 | S | S | S | S | S | S | S |

- `Q4`: one request of at most four rows on the one-request build (four rows). Its
  tokens come first, then dead rows that repeat the last token at the next positions;
  nothing reads their results.
- `L8`: one request of five to eight rows on the 8-row build, with the same dead rows
  below eight rows (K = 4 to 6). Its two row groups select in parallel, rank g for group g
  (the second group's rank first inserts the first group's new index keys); every rank
  then holds all ids, as in the request-parallel tiers.
- `A<n>`: several requests of a whole number of row groups (K = 3, 7) on tier `n`, the
  smallest that holds `R`; no dead rows.
- `P<n>`: packed rows on tier `n` (8 or 16, the smallest that holds `R` and a selector
  slot per segment: tier 8 has eight slots, tier 16 sixteen): the true rows of all
  requests lie back to back with no dead rows, a group of four rows may hold several
  requests, and the index selection runs one segment of at most four rows of one request
  per slot, request-parallel over all eight ranks (rank `r` serves requests `r` and
  `r + 8`).
- `S`: vLLM's stock forward, in the same deployment. The segment's steps hold at most
  16 requests and `R = B (1 + K) <= 16` rows (16 requests for K = 0, 8 for K = 1, 5 at K = 2,
  4 at K = 3, 3 at K = 4, and 2 at K = 5 to 7); any `--max-num-seqs` starts, and a
  larger step replays vLLM's own CUDA graph: the capture sizes keep vLLM's sizes above
  the segment's largest graph, up to `--max-num-seqs` x (1 + K). Before such a step the
  step hook completes the index keys the segment wrote on the selector ranks only (a
  replay skips the forward hook). No tier above 16 rows is built. Every prefill or
  mixed step runs vLLM's path. The log names each step size once: `GLM-5.3 uniform
  decode steps: 41, vLLM's stock forward 12 (this step: 32 tokens, stock)` (or
  `tier 16`). The stock forward's MoE runs stock's `flashinfer_trtllm` kernels on the
  same experts, as stock vLLM's `auto` backend does.
  The stock graphs of steps above the tiers take more memory than vLLM's CUDA graph
  estimate (B200 TP8, `--gpu-memory-utilization` 0.92: 4.44 vs 0.93 GiB at K = 3 and
  `--max-num-seqs` 16, 7.28 vs 0.69 GiB at K = 1), leaving about 6.5 GiB free per GPU.

The row tiers are 8 and 16 plus the one-request build of four rows.

Precision. A step of at most eight live rows runs a W8A16 MoE: the one-request build's
four-row MoE for every `Q4` step and the packed `P8q4` steps (at most four rows), the
eight-row `q8` MoE for the packed `P8q8` steps (five to eight rows), for `A8` (K = 3,
B = 2) and for every `L8` step, seam layers included. A `P8q4` / `P8q8` step runs tier 8
with one / two live row groups and its MoE on rows 0..3 / 0..7 of the hidden buffer;
tail rows past `R` reach the MoE with stale data and their results are not read. The q8
MoE routes only its live rows (the row table's `R`; for `L8` at K = 4 to 6 the request's
`1 + K` tokens, entries `q8d8x<live>`): rows past them add no expert to the union. Every
step of nine or more live rows uses the batched W8A8 MoE.
K >= 4 multi-group requests are not bitwise reproducible run to run, because the
accumulation order is free.

Device memory per rank at the largest `--max-num-seqs` (MiB; peer arena, results and
protocol state of every build of the deployment, from the build plans):

| K | `--max-num-seqs` | MiB |
| --- | --- | --- |
| 0 | 16 | 5724 |
| 1 | 8 | 5724 |
| 2 | 5 | 4444 |
| 3 | 4 | 2836 |
| 4 | 3 | 3490 |
| 5 | 2 | 3490 |
| 6 | 2 | 3490 |
| 7 | 2 | 2423 |

The segment allocates this at startup, so at a fixed memory budget every MiB is a MiB
less KV pool. The packed tier-16 build is shared by every K and its selector has 16
slots (K = 0 at B = 16 needs them), 2662 MiB of the totals; K >= 1 fills at most eight. K = 0 and 1 keep two builds on tier 8 (the q4 MoE and the q8 MoE), each
with its own arena.
