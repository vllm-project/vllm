# MiniMax-M3.1 specialized decode (NVIDIA)

A decode executor for small decode batches of MiniMax-M3.1-Flash on GB300
(and B200/GB200), enabled with
`--additional-config '{"decode_backend": "latency_specialized"}'`. The default,
`auto`, keeps the stock decode: the model then builds no executor, defines no
model state of its own and runs exactly as without this package. The executor
runs the target model's decoder layers; the DSpark drafter keeps its stock
model but, with the switch on, samples its Markov chain in one kernel per
position (`drafter.py`), fuses its norms with the next MXFP8 quantize
(`draft_norm.py`) and reads both LM heads' logits through peer memory
(`peer_logits.py`). The fused kernels are based on NVIDIA's low-latency
MiniMax-M3 decode kernels.

## Scope

`specialized_decode_config` (`__init__.py`) reads the setting once, when the
target model is built. `unsupported()` lists every requirement the deployment
misses, and every TP rank raises with that list:

- GPU: SM100 (B200/GB200) or SM103 (B300/GB300).
- Kernels: every kernel is CuTe DSL (`nvidia-cutlass-dsl`, a vLLM
  dependency), compiled just in time for the live device during warm-up,
  before memory profiling and CUDA graph capture. No nvcc is needed.
- Runner: Model Runner V2, async scheduling.
- Parallelism: TP 4 (`SUPPORTED_TP_SIZES`) over the whole world; no pipeline,
  data (DP attention), expert or context parallelism, no micro-batching.
- Speculative decoding: none, or DSpark with at most 7 draft tokens and
  without adaptive verification.
- Precision: fp8 (E4M3) KV cache (`--kv-cache-dtype fp8`) and fp8 (E4M3)
  index-K cache (`--attention-config '{"indexer_kv_dtype": "fp8"}'`); NVFP4
  and BF16 caches are refused. The checkpoint's MXFP8 attention and dense
  linears and NVFP4 routed experts (`modelopt_mxfp8`, `moe_quant_algo: NVFP4`)
  on the default TRTLLM-gen MoE backend (not MegaMoE).
- CUDA graphs: FULL graphs for decode batches only (`FULL_AND_PIECEWISE` or
  `FULL_DECODE_ONLY`) with at least one capture size within the row cap.
- No `torch.compile` of the model, no sleep mode, no LoRA or batch
  invariance. No KV transfer except `SimpleCPUOffloadConnector` (CPU KV
  offload): its per-layer hooks are no-ops and it loads and saves around the
  forward, so the executor skipping the stock attention layers does not affect
  it (`STEP_LEVEL_KV_CONNECTORS`).

The GPU check is rank-local, so the ranks' results are combined with a MIN
all-reduce over the TP group: if any rank misses a requirement, every rank
raises, rather than some ranks running the executor and others the stock
layers, which would issue different collectives and hang.

The stock decode to compare against is the same deployment with the switch
off (fp8 KV + fp8 index cache), not the recipe's NVFP4 default.

## Structure

| File | Contents |
| --- | --- |
| `__init__.py` | Row caps, `DecodeConfig` (query length, cache dtypes, autotune/JIT flags), `unsupported()`, `specialized_decode_config()` |
| `stages.py` | The stage interface: one ABC per stage, with tensor shapes and dtypes; `Stages` (implementations per stage) and `StepStages` (one per stage, for a step) |
| `stock.py` | Every stage on the ops the stock layers run (the stock-op chain) |
| `kernel_stages.py` | The `kernels/` registry entries behind the stage interfaces (`Kernel*` adapters) and `KernelContext` (peer transport, bound buffers, kernel instances, per-step page rows and sparse lengths) |
| `drafter.py` | The DSpark drafter under the switch: the Markov sampling kernel (`sample_sequential`), the TP-sharded context projection, the row-norm kernel |
| `draft_norm.py` | The draft norms fused with the next linear's MXFP8 quantize (after the stock all-reduce) |
| `peer_logits.py` | Target and draft LM-head logits pushed through CUDA IPC memory instead of the NCCL all-gather |
| `runtime/decode.py` | `SpecializedDecode` (per-layer plans, `plan()`, `run()`), `DecodeStep`, `default_stages()`, `captures_pure_decode()` |
| `runtime/model_state.py` | `MiniMaxM3SpecializedDecodeModelState`: builds the executor after the weights are loaded and warms it up, before memory profiling |
| `runtime/warmup.py` | Calls each stage implementation's `warm_up` for the step rows it supports |
| `kernels/` | The kernels and their registry (stage -> implementation + arch/TP/rows envelope); `kernels/wide.py` holds the front and o_proj GEMVs above the fused kernels' rows; each kernel family is a host module plus its CuTe DSL kernel module (`front_kernel.py`, `page_topk_kernel.py`, `sparse_decode_kernel.py`, `post/`, `o_proj/`, `moe/`) |

Outside the package, `../model.py` reads the switch in
`MiniMaxM3SparseForCausalLM.__init__`, adds `build_specialized_decode()`,
sets `get_model_state_cls` only when the switch is on (the VL wrapper forwards
it), and `MiniMaxM3Model.forward(..., decode_step=...)` hands the layers to the
step after the embedding. `../../common/dspark.py` calls
`drafter.attach_drafter` only with the switch on. No shared vLLM file
changes: the model runner, the CUDA graph manager and the DSpark speculator
run as for any model. The executor returns the aux hidden states as the stock
forward does (clones of the residual stream, one `[rows, hidden]` tensor per
aux layer), and the drafter's FLASHINFER_TS metadata is built eagerly before
each draft step, as upstream.

Per layer the executor runs:

```text
dense layers 0-2:  allreduce_norm.input -> front -> dense_decode -> o_proj
                   -> allreduce_norm.post_attention -> dense_mlp
sparse layers:     allreduce_norm.input -> front -> index_score -> page_topk
                   -> sparse_decode -> o_proj
                   -> allreduce_norm.post_attention (+ router) -> moe
after the layers:  allreduce_norm.final
```

With the kernel stages (sparse MoE layers only; every adapter runs its stock
stage elsewhere) the chain fuses at three places:

```text
KernelFront.run_fused   = allreduce_norm.input + front   (AR land, residual,
                          input norm, MXFP8, QKV/index GEMV, norms, RoPE,
                          FP8 KV / index-K insert)
KernelOProj + KernelAllReduceNorm.post_attention = one pair (o_proj GEMV +
                          router CTAs push -> post: AR land, residual, norm,
                          NVFP4 FC1 input, top-4 routes); the o_proj launches
                          from post_attention, which has the residual
KernelMoE (push)        -> the next layer's KernelFront polls the push
```

The router exchange (`RouterExchangeSlot`) has two halves by tag parity: a
peer's o_proj(k + 1) may store before this post(k) reads, never two calls
ahead. Keep o_proj/post paired and both halves.

`rows = c * (K + 1)` for `c` requests and `K` draft tokens, padded rows
included; at most `MAX_ROWS` = 64 (8 requests x 8 rows). The MoE leaves its
top-k finalize to the next input boundary exactly when the stock MoE does
(`defers_finalize`), unless the next front fuses that boundary (a kernel
front takes a finalized BF16 partial or a push). With the switch on, the stock
model defers up to 64 rows instead of 32 (`fused_finalize_max_tokens`): at
33..64 rows the next input boundary is then the stock fused finalize +
all-reduce + norm + MXFP8 (two launches) instead of the MoE's finalize, the
shared add, FlashInfer's all-reduce + norm and an MXFP8 quantize.

Each kernel stage has its own rows (registry `max_rows`): index_score,
page_topk and sparse_decode up to 64; the front up to 16 (bucket-8 and
bucket-16 builds); the o_proj pair and the MoE 1, 2, 4 or 8. Above those rows
the "wide" variants (`kernels/wide.py`, registry `variant="wide"`, adapters
`Kernel*Wide`) run up to 64:

```text
front (17..64):  stock input boundary -> GEMV QKV (rows = the MMA's N) on
                 its MXFP8 rows (after a deferring MoE: the fused finalize's
                 output; else FlashInfer AR + norm -> mxfp8_rows) -> stock
                 q/k norm + RoPE + cache insert
o_proj (9..64):  GEMV on the kernel sparse decode's MXFP8 output (BF16 partial)
```

Above 8 rows the post-attention boundary and the MoE run stock: at 32-64
rows x top-4 most experts are hit and the stock grouped GEMMs stream each
expert's weights once near the HBM roofline, and the push + post is slower
than FlashInfer's AR + norm there. The build logs the remaining per-rows
fallbacks (`KernelContext.row_fallbacks`).

Which (c, K) steps reach which kernels (rows = c x (K + 1)):

| rows | front | index / top-k / sparse decode | o_proj | post + MoE |
| --- | --- | --- | --- | --- |
| 1, 2, 4, 8 | kernel | kernel | kernel (pair) | kernel (pair, fused engine) |
| 3, 5, 6, 7 | kernel | kernel | stock | stock |
| 9..16 | kernel (bucket 16) | kernel | wide GEMV | stock |
| 17..64 | wide GEMV | kernel | wide GEMV | stock |

So at K = 7, c = 1 (8 rows) runs every kernel; c = 2..8 (16..64 rows) run the
kernel attention trio and the wide front / o_proj GEMVs with the stock
post-attention boundary and MoE.

Multi-request steps (more than 8 rows) at long context take different launch
builds of the attention trio; one-request steps keep the default ones:

- `index_score`: 768 CTAs per request instead of upstream's 256 (the grid is
  batch-fastest, so a long request's CTAs otherwise land on a fraction of the
  SMs). The scores do not depend on the split.
- `page_topk`: a row of more than `SINGLE_MAX` = 4096 valid pages is cut into
  1024-page chunk CTAs; the last chunk CTA to arrive merges their exact local
  top pages (a per-row counter it re-arms, no CTA waits for another). The
  table is the same, bit for bit.
- `sparse_decode`: reads the sequence lengths (and q, when the front wrote it
  two launches back) before its PDL grid wait; row counts where a
  `__launch_bounds__` min-blocks-3 build saves a wave of clusters use it
  (bit-identical output).

## Dispatch and fallback

`plan()` returns a `DecodeStep` only for a FULL CUDA graph capture of a pure
decode step of `0 < rows <= max_rows`; MRV2 replays those graphs. It returns
None, and the forward runs the stock layers, for:

- pipeline-parallel intermediate tensors, or no forward context;
- eager forwards, including the eager run before each capture (which compiles
  the stock kernels) and the memory-profiling run;
- PIECEWISE captures (prefill and mixed batches);
- any attention or index-cache metadata with prefill rows;
- more rows than `max_rows`.

Per step, each stage takes the first of its implementations that `supports`
the rows and is `compatible` with the others (`Stages.select`: an
implementation whose partner is not selected is demoted, down to the stock
one); the last is always the stock stage. At build, implementations that
do not run over the deployment's cache dtypes (`Stage.cache_dtypes`) are
dropped. Every decision reads host state all TP ranks share (rows, module
attributes, metadata host fields); nothing reads GPU values back.

Mix rules (every mix `select` returns is correct):

| kernel stage | needs | otherwise |
| --- | --- | --- |
| `KernelFront` | nothing (takes a push or a BF16 partial) | - |
| `KernelIndexScore` | nothing (stock contract) | - |
| `KernelPageTopK` (physical pages) | `KernelSparseDecode` | `KernelPageTopKLogical` (stock buffer) |
| `KernelSparseDecode` | nothing (gathers physical pages from the stock buffer) | - |
| `KernelOProj` | `KernelAllReduceNorm` | stock |
| `KernelAllReduceNorm` | `KernelOProj` | stock |
| `KernelMoE` | `KernelAllReduceNorm` (routes, NVFP4 input) | stock |
| `KernelFrontWide`, `KernelOProjWide` | nothing (stock contracts) | - |

The kernel build (`KernelContext.build`, from the first kernel stage's
`warm_up`) is collective and fail closed: peer memory first in a fixed order,
then the CuTe DSL kernel modules, then the instances; after each phase the ranks agree
(MIN over the TP CPU group), and a stage that failed anywhere runs stock
everywhere (logged). The agreement on the stages to build comes before any
collective: a rank-local failure (a failed kernel compile) drops those stages
everywhere instead of sizing the collectives differently.

## Adding a kernel stage

1. Register the kernel in `kernels/__init__.py` (arch, TP, max rows) and
   write its adapter in `kernel_stages.py`, implementing the stage's ABC from
   `stages.py`. Keep the input/output shapes, dtypes and side effects (cache
   writes, the shared `topk_indices_buffer`) its docstring states, so that it
   composes with the stock stages on either side; where it cannot (a private
   layout, a fused boundary), declare the partner in `compatible`.
2. `supports` comes from the registry envelope and the build; larger steps
   fall back to the stock stage.
3. Build in `KernelContext.build` (peer memory in the fixed-order phase,
   compiles in the local phase), never inside a capture. Size its grid from
   the live SM count (`kernels/geometry.py` holds every SM-count dependent
   choice); a grid whose CTAs spin on each
   other or on a co-running grid must pass `_arch.check_persistent_grid` /
   `check_co_resident` with the runtime's occupancy (clusters:
   `cudaOccupancyMaxActiveClusters`) at build, or it can deadlock.
4. Add it to `kernel_stage_lists` (ahead of the stock stage). Weight
   addresses may be bound at build (sleep mode is refused).
5. Write it in CuTe DSL and compile every variant the stage uses at build,
   never inside a capture.

Swapping a stage never changes the stock path: with the switch off nothing in
this package runs.

## Serving

GB300, TP4:

```bash
vllm serve MiniMaxAI/MiniMax-M3.1-Flash -tp 4 --block-size 128 \
  --disable-hybrid-kv-cache-manager --async-scheduling \
  --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype": "fp8"}' \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "attention_backend": "FLASHINFER_TS"}' \
  --additional-config '{"decode_backend": "latency_specialized"}'
```

The log shows `MiniMax-M3.1 specialized decode on, for FULL-graph decode
steps of up to 64 rows`.

## Not yet

- Attention-metadata CUDA graphs (DeepSeek-V4.1's `metadata_graphs`): the
  sparse and indexer metadata builds still run eagerly each step; the CUTLASS
  sparse decode's plan cache must be audited for host reads first.
- A low-batch target LM head / fused rejection sampler.
- The fused post + MoE engine above 8 rows (the c = 2..8 steps at K = 7).
- TP 8/16 (`unsupported()` accepts TP 4 only).
