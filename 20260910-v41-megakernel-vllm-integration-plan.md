# Integrating the FlashMLA V4.1 megakernel and V4.1 KV formats into `vllm/models/deepseek_v4_1/attention.py`

| | |
| :-- | :-- |
| Source doc | `20260910-upstream-pr221-v41-megakernel.md` (FlashMLA PR #221) |
| Target | `vllm/models/deepseek_v4_1/attention.py` + `nvidia/flashmla.py` (CUDA / SM100 only) |
| Hardware here | GB200 (SM100), torch 2.13, vLLM builds `_flashmla_C` against the torch 2.11 stable ABI |
| Checkpoint | `ckpt20260903`: 64 heads, `head_dim=512`, `o_groups=8` (8 heads per group), `q_lora_rank=1280`, `o_lora_rank=1024`, `sliding_window=128`, `index_topk=512`, MXFP8 weights (`[32,32]` ue8m0 blocks) |
| Status | In progress on branch `v41-megakernel` (worktree). Done: FlashMLA pinned to upstream `07a1089` and built (Task 2), op wrappers + equivalence tests (Tasks 3-4), Phase 0 spike (Task 5, section 4b), layout helpers, Q padding kernel, permuted O quant, config flags, int32 positions, weight-permutation hook, base-class forward hooks and the fused attention layer (Tasks 1, 6-12). Pending: TP4 end-to-end parity/latency (Task 13), `nvfp4_ds_mla` (Part 3). |

---

## 0. Recommendation in one paragraph

Integrate the megakernel as a **new attention subclass** `DeepseekV4FlashMLAFusedAttention` selected by a config flag, keep the existing FlashMLA class untouched as the fallback, and do the weight permutations **in torch at load time** (the MXFP8 checkpoint makes both permutations plain row/column reorders, so upstream's `permute_q_b_proj` / `permute_wv_proj` kernels are not needed). Land it in two functional steps: first the megakernel on today's `fp8_ds_mla` (V4, 584 B) caches, then a new `nvfp4_ds_mla` mode that stores the sliding-window cache as V4.1 fp8 (528 B) and the compressed cache as V4.1 fp4 (288 B). The single biggest prerequisite is finishing the FlashMLA fork merge on `sync/upstream-pr-221`: it currently has unresolved conflict markers, upstream's pybind registration instead of the fork's `STABLE_TORCH_LIBRARY`, and it dropped the fork's V3.2 NVFP4 format that vLLM's V3.2 backend still builds. The main performance risk is bs1 decode: the fused decode kernel has no split-KV, so one query token runs on one CTA; Phase 0 measures this before any vLLM code is written.

---

## 1. Facts the plan rests on

### 1.1 Current attention pipeline (per layer, FlashMLA subclass)

```text
hidden --fused_wqa_wkv--> qr | kv
qr --q_norm + MXFP8 quant (fused_q_kv_rmsnorm_quant)--> QuantizedActivation
qr --wq_b (FlashInfer CUTLASS MXFP8 GEMM)--> q [N, 16 local heads, 512] bf16      (TP4)
q, kv --_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert-->                     (attention.py:867)
        q_padded [N, 64, 512] (Q RoPE, zero-filled heads 16..63)  +  SWA cache insert (V4 584 B rows)
compressor.insert_cache --rope_quant_insert (Triton)--> compressed cache (V4 584 B rows)
forward_mqa:
   prefill: dequantize_and_gather_k_cache -> bf16 workspace; combine indices; flash_mla_sparse_fwd  (flashmla.py:_forward_prefill)
   decode:  flash_mla_with_kvcache(k_cache=SWA, extra_k_cache=compressed, tile_sched, split-KV)   (flashmla.py:_forward_decode)
o_padded[:, :16] --fused_inv_rope_fp8_quant (Triton, per-32 ue8m0, TMA-aligned SF)--> (o_fp8 [N, 2, 4096], sf)
--deep_gemm fp8_einsum("bhr,hdr->bhd", recipe (1,1,32)) with wo_a [2, 1024, 4096]--> z [N, 2, 1024]
--wo_b--> hidden
```

The megakernel replaces the boxed middle: Q RoPE + padding kernel, the sparse attention kernel, and the inverse-RoPE + FP8 quant kernel, and emits `(o_fp8, sf)` directly in the layout the `wo_a` einsum consumes (after `wo_a` is permuted once).

### 1.2 Weights (why no permute kernels are needed)

| Tensor | Checkpoint | After vLLM loading (per rank, TP4) | Post-load processing that must run *after* our permutation |
| :-- | :-- | :-- | :-- |
| `wq_b.weight` / `.scale` | `[32768,1280]` e4m3, `[1024,40]` e8m0 | `[8192,1280]` fp8, `weight_scale [8192,40]` uint8 (per-row; `KMxfp8Static` repeat-interleaves the 32-row block) | `FlashInferCutlassMxfp8LinearKernel.process_weights_after_loading` swizzles the scale (`mxfp8/flashinfer.py:50`) |
| `wo_a.weight` / `.scale` | `[8192,4096]`, `[256,128]` | `[2048,4096]` fp8, `weight_scale [2048,128]` uint8 | `DeepGemmMxfp8BmmLinearKernel.process_weights_after_loading` views as `[2,1024,4096]` and runs `transform_sf_into_required_layout` (`mxfp8/deep_gemm.py:44`) |

Because both scales are per-row-per-32 after loading, upstream's permutations reduce to:

- `wq_b`: row `(h, d)` moves to row `(d // 16) * (h_local * 16) + h * 16 + (d % 16)`; apply the same row permutation to `weight_scale`. `h_local` is the *local* head count the GEMM produces (16 at TP4), not the padded 64.
- `wo_a`: within each group's 4096 input columns, the 32-wide chunk `(h, c)` (`h` head in group, `c = d // 32`) moves to chunk position `c * 8 + h`; apply the same chunk permutation to the 128 scale columns. All rows share this column permutation, so it is one `index_select` on the 2-D weight and one on the 2-D scale.

Ordering is already right: `DeepseekV41ForCausalLM.load_weights` calls `self.process_weights_after_loading()` (`nvidia/model.py:1103`) *before* `model_loader/base_loader.py:81` runs the quant methods' `process_weights_after_loading`. The DSpark draft model (`nvidia/dspark.py:522`) and `vl_model.py` have their own hooks and must get the same treatment (MTP layers `mtp.{0-2}.attn` carry `wq_b`/`wo_a` too).

### 1.3 Megakernel contract that vLLM must satisfy (decode)

- `q [s_q, h_q, 512]` bf16 with `h_q in {64, 128}`, permuted layout, `q.stride(1) == 512` (whole token contiguous). Batch is flattened into `s_q`.
- `k_cache [num_blocks, page, 1, bytes]` uint8, `stride(1) == bytes`, `stride(0)` a multiple of 576 (V4), 512 (V4.1) or 256 (fp4). vLLM's `alignment=` on the spec already pads pages; `unsqueeze(-2)` keeps the strides, exactly as today.
- `indices_in_kvcache [s_q, topk]` int32, `block * page + offset`, `-1` invalid; `topk_length [s_q]` int32. `topk % 8 == 0` (contiguous). SWA width 128, DSpark width padded to 64, `index_topk` 512: all satisfy it.
- `token_positions [s_q]` int32 contiguous (vLLM positions are int64: one copy per step, see 3.9).
- `cos_sin_cache [*, 64]` fp32: `self.rotary_emb.cos_sin_cache` already is (`deepseek_scaling_rope.py:247`).
- `attn_sink [h_q]` fp32: `self.attn_sink` is already padded with `-inf`.
- `enable_q_norm=False` for V4.1 (today's call passes `apply_q_norm=False`), `num_per_channels=32`, the three DeepGEMM flags `True`.
- Returns `out_fp8 [s_q, n_wv_group=h_q/8, 4096]`, `out_sf [s_q, n_wv_group, 32]` int32 with `stride(0) == 1`, `lse`. No `out=` parameters yet, no split-KV, no scheduler metadata.
- `extra_k_cache` must be the same format as `k_cache`, or V4.1 fp4 when `k_cache` is V4.1 fp8.

### 1.4 DeepGEMM einsum accepts a sliced group dimension

`fp8_einsum("bhr,hdr->bhd")` permutes A to `[h, b, r]` and builds its TMA descriptors from `a.stride(1)` and `a.stride(0)` (`DeepGEMM/csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:418`), so `out_fp8[:, :n_local_groups]` (batch stride 4096, row stride 8*4096) and `out_sf[:, :n_local_groups]` can be fed without a copy. This avoids padding `wo_a` to 8 groups (which would cost 4x einsum weight traffic at TP4). Must be covered by a unit test (3.10 T8) because the SF path goes through `transform_sf_pair_into_required_layout`.

### 1.5 FlashMLA fork state (`/home/yongye/FlashMLA`, branch `sync/upstream-pr-221`, commit `aa9f304`)

- Conflict markers remain in `csrc/api/{api.cpp, common.h, sparse_decode.cpp, sparse_prefill.cpp, dense_decode.cpp}`.
- `csrc/api/api.cpp` is upstream's `PYBIND11_MODULE`; vLLM's cmake (`USE_SABI 3`, `TORCH_TARGET_VERSION=0x020B...`) and the vendored `flash_mla_interface.py` (`torch.ops._flashmla_C`) require the fork's `STABLE_TORCH_LIBRARY(_flashmla_C)` with `torch::stable::Tensor` implementations.
- `ModelType::V32_NVFP4_FP8ROPE` (fork PR #18, 352 B/token) is gone from `csrc/kernels/params.h` / `kv_cache_format.h`; `vllm/v1/attention/backends/mla/flashmla_sparse.py` and `cmake/external_projects/flashmla.cmake` still depend on it.
- The megakernel API (`fused_norm_rope_attn_rope_cast_fwd.cpp`) uses `at::Tensor`, `torch::empty`, `TORCH_CHECK`; it has to be ported to the stable ABI like `sparse_decode.cpp` was.
- `cmake/external_projects/flashmla.cmake` pins `vllm-project/FlashMLA@6bc4941` and lists the *old* source tree (`csrc/sm90/...`, `csrc/smxx/...`); every path changes to `csrc/kernels/...`.
- Open PRs on `vllm-project/FlashMLA` and `Inferact/sra` (checked 2026-09-10): none overlap.
- Update (2026-09-10 06:20): the main checkout `/home/yongye/FlashMLA` now carries a staged, uncommitted stable-ABI port of the merged tree (`STABLE_TORCH_LIBRARY` in `csrc/api/api.cpp`, `interfaces.h`, the fused API in `torch::stable`, ops `fused_norm_rope_attn_rope_cast_fwd` / `_decode` / `permute_q_b_proj` / `permute_wv_proj` all returning `Tensor[]`). Another session owns that work; this plan consumes it through `FLASH_MLA_SRC_DIR` and does not edit it. Two gaps observed there: the Python `flash_mla_interface.py` had not yet restored the fork's `out=` parameters that vLLM's decode/prefill calls use, and `V32_NVFP4_FP8ROPE` is still absent.

---

## 2. Scope

### In scope

1. Fused prefill and fused decode for every DeepSeek V4.1 layer type (cr = 0 SWA-only, cr = 1, cr = 2), on SM100, FlashMLA backend, `h_q` padded to 64 (TP >= 1 head counts of 64 or fewer; 128 needs no code change but is untested here).
2. Weight permutation at load for `wq_b` and `wo_a` (main and MTP layers), idempotent under weight reload.
3. `--kv-cache-dtype nvfp4_ds_mla` for DSv4.1: SWA cache in V4.1 fp8 (528 B/token, RoPE quantized, per-32 ue8m0), compressed cache in V4.1 fp4 (288 B/token, e2m1 with per-16 e4m3 scales). Insert kernels, prefill gather/dequant, KV spec plumbing, SM100 gating. This also works with the non-fused decode kernel (PR #221 added V41 / V41+fp4 to `flash_mla_with_kvcache` head64), so the format and the megakernel can be enabled independently.
4. Decode fallback to the split-KV kernel when the batch is too small for the persistent megakernel to fill the GPU (policy set by Phase 0).
5. Tests, gsm8k / gpqa parity, bs1 and high-concurrency latency numbers.

### Out of scope (stated assumptions)

- FlashInfer backends, SM90, SM120, ROCm: unchanged. `nvfp4_ds_mla` is rejected outside SM100 for DSv4.1.
- DeepSeek V4.0 (`vllm/models/deepseek_v4/`): not wired (its `enable_q_norm=True` variants exist upstream; the cmake list can add them later).
- The `fp8_ds_mla` string keeps today's V4 584 B layout for both caches. A V4.1-fp8-for-both mode (528/528) is not added; it would need a new `CacheDType` literal and is only useful for isolating fp4 accuracy effects.
- Upstream's `permute_q_b_proj` / `permute_wv_proj` kernels are not built into vLLM.

---

## 3. Design

### 3.1 Selection and class layout

- New `AttentionConfig.dsv4_fused_attention: bool | None = None`. `None` resolves to `True` when: CUDA SM100, backend resolves to FlashMLA, `torch.ops._flashmla_C.fused_sparse_decode_fwd` exists, `n_heads // o_groups == 8`, and the padded head count is 64 or 128. Anything else resolves to `False` with an `info_once` reason.
- New `AttentionConfig.dsv4_fused_decode_min_tokens: int` (default from Phase 0; `0` means always fused). Decode steps with fewer query tokens than this take the split-KV fallback.
- `_select_dsv4_attn_cls` (`nvidia/model.py:114`) returns `DeepseekV4FlashMLAFusedAttention` (new file `nvidia/flashmla_fused.py`, subclass of `DeepseekV4FlashMLAAttention`) when the flag is on. `backend_cls` / `swa_backend_cls` are unchanged: metadata and KV specs are the same objects; only the layer's forward changes.

Alternatives considered: (a) a new `AttentionBackendEnum` member: rejected, the KV-cache group machinery would treat it as a different backend for no reason. (b) an instance flag on `DeepseekV4FlashMLAAttention`: rejected, the fused path changes the forward's output contract (see 3.2) and the class is already the one that must keep working unchanged as the fallback.

### 3.2 Forward restructure in `DeepseekV4Attention` (attention.py)

`eager_break_during_capture` requires in-place outputs (`breakable_cudagraph.py:77`). Today the eager region writes `o_padded [N, 64, 512]` and the graph runs `_o_proj`. The fused path's eager region must instead end at `z [N, n_local_groups, o_lora_rank]` bf16 (post-`wo_a`), because `out_fp8`/`out_sf` are allocated by the kernel and consumed by the einsum inside the same eager region. Two small hooks make both shapes fit one `forward`:

```python
def forward(self, positions, hidden_states, llama_4_scaling=None):
    num_tokens = hidden_states.shape[0]
    attn_out = self._alloc_attn_out(num_tokens, hidden_states)   # base: o_padded; fused: z
    qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
    qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
    self._prepare_and_attn_fn(hidden_states, qr, kv, qr_scale, kv_score, indexer_weights, positions, attn_out)
    return self._finish_o_proj(attn_out, positions)               # base: slice heads -> _o_proj; fused: wo_b(z.flatten(1))
```

`_prepare_and_attn` is unchanged except that `_fused_qnorm_rope_kv_insert` becomes an overridable `_prepare_q_and_insert_kv(q, kv, positions, attn_metadata) -> q_for_attention` and `forward_mqa(q, kv, positions, out)` keeps its signature (`out` is `z` for the fused class). No change for FlashInfer / ROCm subclasses beyond the two default hooks.

### 3.3 Q side in the fused class

`wq_b` produces `q_perm_local [N, 16*512]` in the permuted layout for 16 heads (32 chunks of 256). The kernel wants 32 chunks of 1024 (64 heads) with heads 16..63 zero.

- `n_local_heads == padded_heads` (TP1): pass `q.view(N, 64, 512)` straight to the kernel. No Q kernel at all.
- otherwise: Triton kernel `dsv41_q_layout` (new, `common/ops/q_layout.py`) with a `MODE` constexpr:
    - `PERMUTED_PADDED`: scatter each 256-element chunk to offset `c * 1024` of a persistent `[max_num_batched_tokens, 32768]` bf16 buffer whose padding regions are zeroed once at init (never rewritten; graph-stable address). Traffic: 16 KB read + 16 KB write per token, less than today's Q RoPE kernel (16 KB read, 64 KB write).
    - `STANDARD_PADDED_ROPE`: for the fallback, un-permute to `[N, 64, 512]`, apply GPT-J RoPE on the last 64 dims with the same `cos_sin_cache`, zero padding heads. Mirrors the Q half of `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (`processDeepseekV4Slot`, no Q norm).

Rejected: padding `wq_b` to 64 heads (4x GEMM weight bytes per layer at TP4, roughly 0.1-0.2 ms per step at bs1); a 32-way batched GEMM writing a strided output (no MXFP8 kernel does broadcast-A bmm). A follow-up fork change (`n_real_heads` argument so WG0 zero-fills padding heads while loading Q) removes the scatter entirely; listed in Phase 4.

### 3.4 KV insert (both classes need it; the fused class needs it split from Q)

Today `_C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` does Q + KV in one launch. The fused class needs KV-only, layout-selectable inserts:

| Cache | Layout | Kernel |
| :-- | :-- | :-- |
| SWA, `fp8_ds_mla` | V4 584 B: 448 e4m3 + 64 bf16 RoPE + 7 ue8m0 (per-64) in a segregated 8 B scale row | existing CUDA kernel with `num_heads_q = 0` (Q half skipped) or the Triton `quantize_and_insert_k_cache` + RoPE. Prefer adding a `q_out == nullptr` early-out to the CUDA kernel: one launch, already tested. |
| SWA, `nvfp4_ds_mla` | V4.1 fp8 528 B: RoPE'd row, all 512 quantized e4m3 in 16 per-32 tiles, `sf = ceil_pow2(amax/448)` (clamp amax >= 1e-4), 16 B ue8m0 scale row at `page + block*512 + slot*16` | new store path in the same CUDA kernel (`KV_LAYOUT` template: V4 / V41). RoPE result is already in registers. |
| compressed, `fp8_ds_mla` | V4 584 B | existing Triton `_rope_quant_insert_kernel` |
| compressed, `nvfp4_ds_mla` | V4.1 fp4 288 B: RoPE'd row, 32 tiles of 16, `scale = clamp(amax/6, 2^-9, 448)` as e4m3, codes e2m1 RN packed 2/byte (even index low nibble), 256 B data at `page + slot*256`, 32 B scales at `page + block*256 + slot*32` | new Triton branch in `rope_quant_insert` reusing `_fp32x2_to_fp4x2` from `fused_indexer_q`. Reference: FlashMLA `tests/quant.py::quantize_k_cache(V41_FP4)` (NaN element poisons the tile scale; match it). |

The DSpark context insert (`nvidia/dspark.py:_insert_context_kv`) calls the same CUDA op and gets the same layout switch.

### 3.5 Attention dispatch in the fused class

```text
forward_mqa(q_perm_local, kv, positions, z):
  attn_metadata is None (dummy run): reserve workspaces (prefill gather as today, Q pad buffer), z.zero_(), return
  prefill tokens (per chunk of the existing chunk plan):
      gather+dequant compressed (V4|fp4) and SWA (V4|V41) into the bf16 workspace   (3.6)
      combine_topk_swa_indices (unchanged)
      out_fp8, out_sf, _, _ = fused_sparse_attn_prefill(q_pad[qs:qe], kv_ws, indices, sm_scale, attn_sink, topk_length,
                                                        token_positions=pos32[qs:qe], cos_sin_cache, n_wv_group=8)
      fp8_einsum("bhr,hdr->bhd", (out_fp8[:, :G], out_sf[:, :G]), (wo_a.weight, wo_a.weight_scale), z[qs:qe], recipe=(1,1,32))
  decode tokens:
      if num_decode_tokens >= dsv4_fused_decode_min_tokens:
          out_fp8, out_sf, _ = fused_sparse_attn_decode(q_pad, swa_cache, swa_indices, ..., extra_k_cache=compressed, extra_indices, extra_topk_length,
                                                        token_positions=pos32, cos_sin_cache)
          fp8_einsum(... z[:num_decode_tokens])
      else:   # split-KV fallback, same z contract
          q_std = dsv41_q_layout(q_perm_local, MODE=STANDARD_PADDED_ROPE)
          flash_mla_with_kvcache(... as today ..., out=o_padded_ws)
          o_fp8, sf = fused_inv_rope_fp8_quant(o_padded_ws[:, :16], ..., permuted_output=True)   # 3.7
          fp8_einsum(... z[:num_decode_tokens])
```

`z[qs:qe]` slices are valid einsum outputs (DeepGEMM reads `d.stride(0)`/`d.stride(1)`). The kernel-allocated `out_fp8`/`out_sf` live only inside the eager region. The tile-scheduler metadata (`tile_sched_*`) is only consulted on the fallback path.

### 3.6 Prefill gather / dequant for the new formats

`dequantize_and_gather_k_cache` (`common/ops/cache_utils.py:394`) dispatches on the 584 B layout only. Add `layout` selection by `k_cache.shape[-1]` (584 / 528 / 288) with Triton branches: V41 (16 tiles x 32, ue8m0 from the segregated 16 B row) and fp4 (nibble unpack via a 16-entry e2m1 LUT, times the e4m3 scale; exact in bf16 as upstream notes). The CuteDSL variant (`has_cutedsl`) stays V4-only and the dispatcher falls back to Triton for the new formats. The fused prefill kernel itself only ever sees bf16, so this is the entire prefill-side change for the new formats.

### 3.7 Output side

- `fused_inv_rope_fp8_quant` gains `permuted_output: bool`: chunk `(h, c)` is stored at chunk position `c * 8 + h` and its ue8m0 byte goes to byte `(c*8 + h)` of the token's 128 scale bytes (store through a `uint8` view of the int32 SF buffer; DeepGEMM packs byte `k` of int32 `i` as chunk `4i + k`, little-endian, which matches the existing `<< (k*8)` packing). Only the fallback path uses this; the fused kernel emits the same layout natively.
- `deep_gemm_fp8_o_proj` (`deepseek_v4/nvidia/ops/o_proj.py`) is not used by the fused class; `wo_b(z.flatten(1))` runs in the graph after the eager region.

### 3.8 KV cache format plumbing (`nvfp4_ds_mla` for DSv4.1)

One table, one place (`attention.py`, next to `_resolve_dsv4_kv_cache_dtype`):

| `--kv-cache-dtype` | SWA bytes / alignment | compressed bytes / alignment | requires |
| :-- | :-- | :-- | :-- |
| `fp8_ds_mla` (and `fp8`, `auto`) | V4 584 / 576 | V4 584 / 576 | SM90 or SM100 (today) |
| `nvfp4_ds_mla` | V4.1 fp8 528 / 512 | V4.1 fp4 288 / 256 | SM100, FlashMLA backend |

- `_resolve_dsv4_kv_cache_dtype` returns the row above (a small frozen dataclass `DSv4KVLayout(swa_bytes, swa_alignment, compressed_bytes, compressed_alignment, name)`) instead of `(str, torch.dtype)`; both caches stay `torch.uint8`.
- `DeepseekV4SWACache` takes the layout (bytes, alignment) instead of testing `cache_dtype == "fp8_ds_mla"` (`sparse_swa.py:110`); `DeepseekV4Attention.get_kv_cache_spec` likewise (`attention.py:936`). `model_version` stays `"deepseek_v4"` (asserted at `kv_cache_interface.py:810`); `kv_quant_mode=get_kv_quant_mode("nvfp4_ds_mla")` already yields `NVFP4_DS_MLA`; `kv_cache_dtype_str_to_dtype` already maps the string to `uint8`.
- `DeepseekV4SparseMLABackend.supported_kv_cache_dtypes` adds `"nvfp4_ds_mla"`; `supports_combination` rejects it off SM100. `DeepseekV4IndexerCache` is unaffected (its layout comes from `indexer_kv_dtype`).
- Page geometry check: SWA block 32 x 544 B = 17408 = 34 x 512; compressed kernel block 128 (cr 1) or 64 (cr 2) x 288 B = 36864 / 18432, both multiples of 256. `alignment=` still pads defensively.
- Capacity effect: SWA -9.6 %, compressed -50.7 % bytes per token.

### 3.9 Metadata

`DeepseekSparseSWAMetadata` gains `positions_int32: torch.Tensor | None`, written by the builder from `common_attn_metadata.positions` into a preallocated `[max_num_batched_tokens]` int32 buffer (graph-stable), once per step, shared by all layers. Nothing else changes: `decode_swa_indices` / `decode_swa_lens` and `compute_global_topk_indices_and_lens` already produce the `block * page + offset` / `-1` convention the fused kernel wants, and the prefill chunk plan is reused verbatim.

### 3.10 Tests (all under `tests/kernels/`, GPU-marked, SM100-skipped elsewhere)

| # | Test | Guards against |
| :-- | :-- | :-- |
| T1 | `permute_wq_b_rows` / `permute_wo_a_cols` vs the index formulas in 1.2, CPU only, including idempotency guard | wrong permutation, double permutation on reload |
| T2 | `dsv41_q_layout` both modes vs torch reference; `STANDARD_PADDED_ROPE` vs the existing CUDA op's Q output | layout / RoPE mismatch |
| T3 | fused decode on V4 caches (`s_q in {1, 7, 64, 300}`, with and without `extra_k_cache`, `topk_length`, all-invalid rows) vs `flash_mla_with_kvcache` + `fused_inv_rope_fp8_quant` + einsum, compared at `z` | end-to-end numerics of the fused path, sliced-group einsum |
| T4 | fused prefill vs `flash_mla_sparse_fwd` + o_proj at `z`, chunked | prefill path, `z[qs:qe]` slicing |
| T5 | V4.1 fp8 SWA insert round trip vs a torch port of FlashMLA `tests/quant.py` (V41 branch) | scale rounding, scale-row addressing |
| T6 | fp4 compressed insert round trip vs the V41_FP4 reference, including NaN poisoning and the 2^-9 clamp | e2m1 rounding, nibble order |
| T7 | `dequantize_and_gather_k_cache` for 528 / 288 vs T5/T6 references with `gather_lens`, `offset` | prefill dequant |
| T8 | `fp8_einsum` with `out_fp8[:, :2]` of an 8-group buffer vs contiguous copy | DeepGEMM strided batch + SF layout |
| T9 | fused decode against both `flash_mla_with_kvcache` and a bf16 torch reference on V41 + fp4 caches | format detection, extra-cache boundary block (`topk` 128 with `B_TOPK` 64) |
| E1 | `recipe/dsv41/sra_vigil_tp4_gsm8k.yaml` with fused on/off, `fp8_ds_mla` | parity |
| E2 | gsm8k + gpqa with `nvfp4_ds_mla`; plus a 32k-context task (fp4 only affects the compressed, long-range path; gsm8k barely exercises it) | accuracy of quantized RoPE and fp4 |
| P1 | `sra_vigil_tp4_8k1k_32k1k_bs1.yaml` and `..._dspark_synth.yaml`: TPOT must not regress (baseline 2026-09-10: no-spec ~6.6 ms, DSpark ~3.35 ms) | bs1 regression from the persistent kernel |
| P2 | same model at concurrency 32 / 128, and a prefill-heavy run | the actual win |

---

## 4. Phased delivery

### Phase 0: measurement spike (no vLLM code)

Build upstream `pr-221` (`07a1089`, pybind, builds standalone in `/home/yongye/FlashMLA/.venv`) and run `tests/test_fused_norm_rope_attn_rope_cast.py` plus a small script comparing, at `h_q = 64`, `topk = 128 + 512`, V4 caches:

- fused decode vs `flash_mla_with_kvcache` (split-KV) at `s_q in {1, 6, 16, 64, 256, 1024}`;
- fused prefill vs `flash_mla_sparse_fwd` at `s_q in {184, 2123}`;
- kernel time under CUDA graph capture (the kernel is CLC-driven persistent; confirm it captures and replays).

Output: the `dsv4_fused_decode_min_tokens` default and whether the fallback path (3.3 mode 2, 3.7) is needed at all. Estimate before measuring: one CTA walks about 10 blocks of 64 rows per query token, so bs1 decode is likely 2-4x slower per layer than split-KV, while the two removed launches save roughly 0.1 ms per step; expect fallback to be needed below a few dozen tokens.

### Phase 1: FlashMLA fork (prerequisite, largest chunk)

1. Finish the `sync/upstream-pr-221` merge: resolve the five conflicted API files on the fork's `STABLE_TORCH_LIBRARY` side; keep optional `out_` buffers and `num_sm_parts` clamping.
2. Port `fused_norm_rope_attn_rope_cast_fwd.cpp` to `torch::stable` and register two ops with optional output buffers (so vLLM can later pass graph-stable buffers), e.g. `fused_sparse_prefill_fwd(...) -> (Tensor, Tensor, Tensor, Tensor)` and `fused_sparse_decode_fwd(...) -> (Tensor, Tensor, Tensor)`; expose them from `flash_mla/flash_mla_interface.py` (that file is what vLLM's cmake vendors and rewrites).
3. Re-add `V32_NVFP4_FP8ROPE` (352 B, `d_qk = 576`) to `params.h` / `kv_cache_format.h`, its head64 decode instantiation and detection, so vLLM's V3.2 `nvfp4_ds_mla` keeps working.
4. Build both ways: upstream `setup.py` and vLLM's cmake flags (`10.0f` family gencode). If the megakernel sources need `sm_100a` (upstream switched to `100a`/`103a` "for better SASS"), give those files their own `set_gencode_flags_for_srcs` entry in `flashmla.cmake`.
5. vLLM `flashmla.cmake`: new `FlashMLA_SOURCES` paths (`csrc/kernels/...`), add the 8 `*_nonorm.cu` megakernel instantiations (v4 h64/h128 prefill + decode, v41 h64/h128 decode, v41fp4 h64/h128 decode), include dir `csrc/kernels`, bump `GIT_TAG` after the fork PR merges (use `FLASH_MLA_SRC_DIR=/home/yongye/FlashMLA` meanwhile; see memory note on incremental kernel builds).
6. `vllm/v1/attention/ops/flashmla.py`: `fused_sparse_attn_prefill`, `fused_sparse_attn_decode`, `is_flashmla_fused_supported()`.

Exit: existing FlashMLA tests pass in vLLM on GB200 (V3.2 fp8 and nvfp4, V4.1 fp8_ds_mla), the fused ops are callable from `torch.ops._flashmla_C`.

### Phase 2: megakernel on `fp8_ds_mla` (V4 caches)

1. `attention.py`: forward restructure (3.2), `_prepare_q_and_insert_kv` hook, KV-only mode of the CUDA insert kernel (3.4 row 1).
2. `common/ops/q_layout.py` (3.3), `fused_inv_rope_fp8_quant(permuted_output=True)` (3.7).
3. Weight permutation hook in `nvidia/model.py`, `nvidia/dspark.py` (`vl_model` inherits), filtered by `loaded_params` so partial reloads stay correct.
4. `nvidia/flashmla_fused.py` (3.5) with the fallback threshold; `positions_int32` in the SWA metadata (3.9); JIT warmup registration for the new Triton kernels (`enable_jit_warmup` path in `attention.py:490`); dummy-run workspace reservation.
5. Config flag (3.1) and `_select_dsv4_attn_cls`.
6. T1-T4, T8, E1, P1, P2.

Exit: gsm8k parity with fused on/off; P1 within noise of baseline; P2 shows the win (or the plan stops here and the flag defaults off).

### Phase 3: `nvfp4_ds_mla` (V4.1 fp8 SWA + fp4 compressed)

1. Layout table and spec plumbing (3.8), backend support lists and SM100 gating.
2. V41 SWA store path in the CUDA insert kernel; fp4 branch in `rope_quant_insert`; DSpark context insert (3.4).
3. Gather/dequant branches (3.6).
4. T5-T7, T9, E2, P1/P2 repeated with `nvfp4_ds_mla` (both fused and non-fused decode, since `flash_mla_with_kvcache` also supports the new formats).

Exit: accuracy report on gsm8k, gpqa and one long-context task; KV capacity and TPOT deltas.

### Phase 4: follow-ups (optional)

- Fork: `n_real_heads` argument so the kernel zero-fills padding heads while loading Q (removes the Phase 2 scatter for TP > 1); pass vLLM-owned `out_fp8` / `out_sf` buffers.
- Prune norm variants from the build if compile time hurts; add them back for DeepSeek V4.0.
- Drop the fallback path if Phase 0 shows the fused decode is never slower in the regimes we serve.

---

## 4b. Phase 0 results (2026-09-10, GB200, upstream FlashMLA @ 07a1089 built into vLLM)

`benchmarks/kernels/benchmark_dsv41_fused_attention.py`, `h_q = 64`, decode over
`topk_swa = 128` (V4 584 B SWA cache) + `topk_extra = 512` (V4 compressed cache),
prefill over `topk = 640`; medians of 50 iterations in microseconds. "split-KV total"
is Q RoPE (torch) + `flash_mla_with_kvcache` + `fused_inv_rope_fp8_quant`.

| decode `s_q` | fused (graph) | split-KV attn only (graph) | split-KV total (graph) | fused (eager) | split-KV attn (eager) |
| --: | --: | --: | --: | --: | --: |
| 1 | 22.7 | 22.7 | 43.2 | 27.7 | 41.4 |
| 6 | 24.0 | 22.7 | 47.2 | 27.0 | 38.1 |
| 16 | 24.7 | 23.4 | 47.3 | 27.2 | 36.9 |
| 64 | 24.7 | 23.9 | 59.6 | 28.2 | 36.2 |
| 256 | 48.9 | 47.3 | 126.8 | 50.4 | 48.0 |
| 1024 | 153.8 | 160.6 | 443.1 | 154.8 | 161.0 |

| prefill `s_q` | fused | `flash_mla_sparse_fwd` + Q RoPE + O quant |
| --: | --: | --: |
| 184 | 30.3 | 386.2 |
| 2123 | 163.0 | 785.0 |

Conclusions: the fused decode kernel matches the split-KV attention kernel even at
`s_q = 1` (the bs1 concern in section 5 did not materialize at this top-k), and it
halves the decode attention segment once the two surrounding kernels are counted.
CUDA-graph capture and replay of the fused decode works. `dsv4_fused_decode_min_tokens`
therefore defaults to 0 (fused everywhere); the split-KV fallback path stays as an
escape hatch.

## 4c. Implementation status (2026-09-10, branch `v41-megakernel`)

| Area | State |
| :-- | :-- |
| FlashMLA build | `flashmla.cmake` pins upstream `deepseek-ai/FlashMLA@07a1089…` (pybind11 module, `torch_python` linked); 51 sm_100 cubins; existing V3.2 fp8 smoke tests pass through the regenerated interface. `_flashmla_extension_C` and V3.2 `nvfp4_ds_mla` are unavailable with this pin. |
| Op wrappers | `flash_mla_fused_sparse_prefill/decode`, `is_flashmla_fused_sparse_supported`, `out=` emulation for the fork-only parameter. |
| Kernel-level tests | 59 passing: layout permutations, Q padding kernel, permuted O quant, fused decode/prefill vs the split-KV pipeline (lse tight, `z` within 2 %), sliced-group einsum, V4.1 fp8/fp4 insert + gather vs torch ports of FlashMLA's reference quantizer, fused decode over V4.1 fp8 + fp4 caches. |
| Layer | `DeepseekV4FlashMLAFusedAttention` selected by `attention_config.dsv4_fused_attention`; weight permutation at load keyed on `loaded_params`; split-KV fallback below `dsv4_fused_decode_min_tokens` (default 0 after Phase 0). |
| `nvfp4_ds_mla` | Layout table, spec plumbing, insert/gather kernels, SM100 gating, DSpark context insert. Needs the fused layer. |
| Pending | TP4 end-to-end parity (fused vs unfused, `fp8_ds_mla` and `nvfp4_ds_mla`), gsm8k/gpqa and bs1 latency via the `recipe/dsv41/*_fused.yaml` recipes (which use the config-shim checkpoint `/mnt/data/yongye/cache/ckpt20260903-v41cfg` because origin/sra-tracking #64 renamed the hf_config fields). |

## 5. Risks and open questions

| Risk | Mitigation |
| :-- | :-- |
| bs1 / DSpark decode slower (no split-KV, one CTA per query token) | Phase 0 measures; threshold fallback keeps today's kernel below it. |
| CLC persistent kernel under CUDA graph capture (`FULL_AND_PIECEWISE`) | Phase 0 captures a graph around the fused decode and replays with changed `s_q` padding. |
| `10.0f` family gencode vs upstream's `sm_100a` | Build both, compare fused-test throughput; per-file gencode if needed. |
| DeepGEMM sliced-batch einsum / SF layout | T8 before any layer code. |
| Weight permutation applied twice on reload / RL refit | Permute only params named in `loaded_params` of that `load_weights` call; T1 covers it. |
| Accuracy: RoPE quantized to e4m3 in V4.1 fp8; fp4 compressed cache | E2 with long-context task; keep `fp8_ds_mla` as the default. |
| Fork merge scope creep (five conflicted files, stable ABI port, NVFP4 V3.2 re-add) | Phase 1 has its own exit gate and can ship as a separate fork PR before any vLLM change. |
| Compile time: 8 more 1600-line kernel instantiations | Only `_nonorm` variants; measure `_flashmla_C` build time. |

Open questions for the user:

1. Is bs1 TPOT a hard constraint for this work (it decides how much effort the fallback path deserves), or is the target the throughput regime the upstream numbers describe?
2. Should `nvfp4_ds_mla` be the DSv4.1 default once E2 passes, or stay opt-in?
3. Which long-context accuracy task should gate the fp4 compressed cache?

---

## 6. PR checklist (AGENTS.md)

- Duplicate check done 2026-09-10: `gh pr list --repo vllm-project/FlashMLA --state open`, `gh pr list --repo Inferact/sra --state open --search "megakernel OR fused_norm_rope OR nvfp4"`; no overlap.
- Each PR description: why not a duplicate, exact test commands and results (T*, E*, P* above), eval numbers because output changes, statement that AI assistance was used.
- Commands: `.venv/bin/python -m pytest tests/kernels/test_dsv41_fused_attention.py -v`; `pre-commit run --all-files`; `pre-commit run mypy-3.12 --all-files --hook-stage manual`.
