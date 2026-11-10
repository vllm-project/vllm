# DeepSeek-V4.1 specialized decode on Hopper

`HopperSpecializedDecode` (`decode.py`) is the specialized decode executor on
H100/H200 (SM90). `SpecializedDecode.for_target` builds it on SM90 instead of
the GB300 executor. It reuses the parent executor's `DecodeStep` and final
collapse, with its own plan, layer loop, attention step and boundaries over
the Hopper stock layers. The DSpark draft model keeps its stock paths.

It runs no DeepGEMM kernel and needs none of the model's mHC overlap path
(`supports_mhc_overlap`, the model's `mhc_stream`, `mhc_pre_delayed_overlap`):
the coefficients come from its own mHC pre (`mhc.py`) on its own side stream,
and the stock layers keep reducing their own outputs outside the executor.

## Scope

The requirements are the parent package's, except:

- GPU: H100 or H200 (SM90).
- Attention: the FlashMLA sparse attention (the default backend on SM90) with
  its `fp8_ds_mla` KV cache.

Only FULL CUDA graph captures of pure decode steps of up to 192 rows (32
DSpark requests) run the executor. Every other forward, and every step larger
than a kernel's row cap, runs the stock layers. Each Hopper kernel reads the
stock layer's weights and caches in place: Humming W8A16 dense weights,
Humming W4A8 experts, and the FlashMLA KV and index caches. The one exception
is the MLA chain's `wo_a` (below).

## Structure

| Module | Contents |
| --- | --- |
| `decode.py` | The executor: plan, layer loop, warm-up of every kernel before graph capture |
| `steps.py` | `HopperAttentionStep`, `HopperMoEStep`, `HopperBoundaries` |
| `mhc.py`, `mhc_cute.py` | The mHC pre: TF32 mixes in CuTe DSL in place of the DeepGEMM prenorm GEMM, the stock epilogue |
| `kernels.py` | One function per kernel choice (dense linears, `wo_b`, MoE) |
| `dense.py`, `dense_gemm_cute.py` | Dense FP8 linears on the CuTe DSL GEMV or tuned Humming, per seam and row count (`DISPATCH`); the FP8 `wo_a` copy |
| `moe.py`, `moe_cute.py` | The low-batch W4A8 MoE (one persistent launch), finalized by the next boundary |
| `sparse_mla.py` | FlashMLA decode on plans and global top-k tables shared across layers |
| `mla_small.py`, `mla_cute.py`, `attn_chain.py`, `attn_cute.py` | The small-batch MLA chain: q/kv norm with KV insert, cluster decode to the `wo_a` input |
| `ptx.py` | Inline PTX helpers of the MoE, MLA and mHC CuTe DSL kernels (the dense GEMV and the q/kv norm keep their own `inline_ptx` helpers) |

## Paths

| Path | Rows |
| --- | --- |
| Fused all-reduce + mHC at every boundary | up to `FUSED_MAX_ROWS` (144); above that the TP all-reduce and mHC post |
| The next coefficients from the executor's mHC pre (`mhc.SeamPre`) on its side stream, in chunks of 64 rows | all executor steps |
| Dense FP8 linears (`fused_wqa_wkv`, `wq_b`, the indexer's `wq_b`, `wo_b`, the shared expert's) on the CuTe DSL GEMV or tuned Humming, chained with PDL | up to 192, per `dense.DISPATCH` |
| Low-batch MoE: both expert GEMMs and the shared expert in one launch, the top-k reduction left to the next boundary (layers with a W8A16 shared expert, routed by score, image sentinel tokens with `bias_vl` as the stock router; hash-routed layers run the stock MoE) | up to 144 |
| MoE entry mode: consumers finalize whole tiles right after GEMM1 | 17 to 144 |
| FlashMLA with one plan per layer type and one global top-k table per index source | steps off the MLA chain |
| Small-batch MLA chain (`ChainDecode`): q/kv norm writing the step's keys into fp16 SWA tile images, one cluster per row group, query RoPE and inverse RoPE fused; `wo_a` on an exact FP8 copy of the stock BF16 weight (8.65 MB a layer: 346 MB, 0.32 GiB, a GPU over the 40 layers at TP4) | uniform steps up to 48 |

The GB300 low-batch kernels, MXFP8 layer input, row-sharded indexer and
candidate overlap do not run on Hopper; nor do the Engram WKV prefetch,
`wo_b` mailbox publishing and recorded metadata builds, which measured no
gain here.

## Ordering across kernels

Every Hopper kernel launches with programmatic dependent launch (PDL) and may
read, before its `griddepcontrol.wait`, only data that no kernel of the
current step writes (weights, tensor maps, the previous layer's outputs once
an earlier wait covered them). The boundary's fused finalize directly follows
the low-batch MoE kernel, which writes the route map and routing weights the
finalize reads, so the executor builds its `BoundaryAllReduce` with
`finalize_metadata_before_pdl=False`: the finalize stages them after its
wait. The GB300 executor keeps the default (its maps come from an earlier
kernel).

## Stock internals it relies on

The executor calls the stock layers' private members where no public hook
exists. A refactor of any of these breaks it (and `test_hopper.py` does not
cover `kernels.moe`):

- `kernels.SharedMlp.installed` assigns the shared expert MLP's `forward` on
  the instance for the call, so the MoE runner's shared-expert stream runs
  the low-batch dense calls.
- `kernels.partial_output` sets the runner's
  `moe_config.skip_final_all_reduce` for the call, so the stock MoE above 144
  rows leaves its output TP-partial for the boundary.
- `kernels.moe` calls the runner's `gate` for the router logits; `moe.HopperLowBatchMoE.of`
  reads `routed_experts`' Humming parameters, the gate's
  `e_score_correction_bias`, `tid2eid` and `bias_vl`, and the shared
  expert's Humming weights and scales.
- `kernels._wo_b_partial` calls `wo_b.quant_method.apply` to skip the
  reduction of a `wo_b` that reduces its own output; the attention step calls
  `DeepseekV4Attention._fused_wqa_wkv_gemm`, `_wq_b_proj`, `_wo_b_proj`,
  `_split_qkv_and_norm`, `_fused_qnorm_rope_kv_insert`, `_compressed_kv_cache` and
  `_alloc_attn_out`, and the indexer's `_wq_b_proj` and `_produce_k`.
- `dense.DenseFp8Linear.of` takes the `HummingMxfp8LinearKernel` and its
  `layer_config`, `compute_config` and `locks`; `sparse_mla` calls
  `get_mla_metadata` and `flash_mla_with_kvcache` directly.

## Editing

- Every kernel is specialized to the shapes of DeepSeek-V4.1-Flash at TP4.
  Each host module's `applies`/`serves` checks gate it, and its `_compiled`
  arguments are the only switches. Edit the CuTe DSL kernels in place.
- The CuTe DSL compiler re-reads kernel sources when it compiles, at warm-up.
  Don't edit a tree while a server built from it is still warming up.
- Collectives: the executor issues the fused all-reduce on every rank. Each
  per-step choice reads only host state that all TP ranks share (row and
  request counts, metadata host fields), as in the parent package.
- Tests: `tests/models/deepseek_v41/nvidia/specialized_decode/test_hopper.py`
  (one SM90 GPU) compares each kernel with the stock op it replaces.
