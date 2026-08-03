# DeepSeek-V4-Flash on NVIDIA Turing (SM75) — Design

> Status: Approved by user.
> Date: 2026-08-03
> Repo: `/data/docker/vllm` at commit `8ba87e0181`, branch `main`.

## Goal

Make `deepseek-ai/DeepSeek-V4-Flash` (284B params, 13B activated, FP4 routed
experts + FP8 block dense) run *at all* on a cluster of 8× NVIDIA RTX 2080 Ti
22GB (compute capability 7.5 / Turing), with long-context (100K+) support.

Success criteria, in priority order:

1. Model loads, forward pass completes, produces coherent tokens.
2. Long-context (≥100K tokens) requests work without OOM or silent corruption.
3. Correctness validated against reference (FP8 path) for identical prompts.

Performance is a research-grade "get it to run" target: 1–5 tok/s decode is
acceptable. Correctness and memory fit matter more than speed.

## Hardware reality (Turing / SM75)

- FP16 tensor cores exist; **BF16 does not exist in hardware**, FP8/FP4/INT4
  tensor-core GEMM do not exist. All heavy GEMM must run FP16 on tensor cores.
- 2080 Ti ≈ 14 TFLOPS FP16. This project's cards are modded to 22GB each;
  total cluster VRAM 8×22GB = 176GB.
- Quantized weights ≈ 159GB (FP4 experts + FP8 dense). Dequantized-FP16 weights
  ≈ 568GB — impossible. Weights must stay quantized on-GPU; dequantization to
  FP16 happens in-kernel at compute time, weight-by-weight-tile.

## Strategy

Build a **Turing (SM75) backend** for the DeepSeek-V4 model: a new platform
branch in `vllm/models/deepseek_v4/__init__.py` that selects `turing/` modules
on CUDA + capability 7.5. The backend reuses as much as possible:

- **Portable Triton kernels** from the XPU backend (`xpu/model.py`,
  `xpu_sparse.py`, `xpu_sparse_decode_fp8.py`) and `common/ops/*` — all pure
  `tl.*`/torch, portable to CUDA. These are the blueprint.
- **Marlin kernels for quantized GEMM** where supported on SM75:
  - Dense FP8 linear layers → `MarlinFP8ScaledMMLinearKernel`
    (`vllm/model_executor/kernels/linear/scaled_mm/marlin.py:29`), which
    supports block (128×128) FP8 weights (`weight_quant_key in {kFp8Static128BlockSym}`, line 67)
    and requires only capability 7.5+ (line 43).
  - FP4 routed experts (MXFP4) → `MarlinMoeKernel`
    (`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`), which
    supports `kMxfp4Static` weights and requires only capability 7.5+
    (`_supports_current_device`, line 626).
- **torch/Triton fallbacks** for the remaining SM100-only ops
  (cutedsl MLA, DeepGEMM einsum/o_proj, DeepGEMM MegaMoE, tilelang MHC).

The design intentionally does **not** reimplement DeepSeek-V4 model code from
scratch; it ports the XPU implementation to CUDA-Triton with FP16 activations,
reusing the shared `DeepseekV4Attention`, `common/ops`, and the sparse-MLA
Triton kernels.

## Architecture

### Platform dispatch

`vllm/models/deepseek_v4/__init__.py` gains a CUDA branch that checks
`current_platform.get_device_capability()`:

```python
if current_platform.is_rocm():
    ...  # amd
elif current_platform.is_xpu():
    ...  # xpu
elif capability is not None and capability.major <= 7:
    from .turing.model import DeepseekV4ForCausalLM
    from .turing.dspark import DSparkDeepseekV4ForCausalLM
    from .turing.mtp import DeepSeekV4MTP
else:
    ...  # nvidia (unchanged default)
```

New directory: `vllm/models/deepseek_v4/turing/`.

### Compute paths (SM75)

| DSv4 component | NVIDIA (SM90+/100) | Turing (SM75) path |
|---|---|---|
| Dense FP8 linear | cutlass FP8 / FP8 block | `MarlinFP8ScaledMMLinearKernel` (block 128×128) |
| Routed experts (FP4) | DeepGEMM FP8×FP4 MegaMoE (SM100) | `FusedMoEFactory` → Marlin MXFP4 (`MarlinMoeKernel`) |
| Routed experts (FP8, Flash-Base) | FP8 MoE | Marlin FP8 block MoE |
| MLA forward | FlashMLA (SM90+) / FlashInfer (SM120) | Triton sparse MLA (port `xpu_mla_sparse`, FP16 KV) |
| MLA KV compress/cache | cutedsl `fused_kv_compress_norm_rope_insert_*` | Triton from `common/ops/fused_compress_quant_cache.py` |
| Indexer topk/logits | DeepGEMM `fp8_fp4_mqa_logits` | Triton `xpu_*` logits + `torch.ops._C.cooperative_topk` (CUDA native, portable) |
| o_proj (einsum + inv rope) | DeepGEMM `fp8_einsum` | dequant weights → FP16, torch `einsum`; `common/ops/fused_inv_rope_fp8_quant.py` Triton (fp16) |
| MHC | tilelang | torch/Triton fallbacks in `vllm/model_executor/layers/mhc.py` (fp16) |
| MTP | FlashMLA + DeepGEMM | shared `common/ops` Triton + Triton MLA; disable if blocked |
| DSPark (draft) | V2 runner only | torch/Triton only; disabled on SM75 (research goal) |

### KV cache layout

- NVIDIA FlashMLA uses FP8 uint8 KV with per-token scales
  (`fp8_ds_mla` layout). Turing keeps **FP16 KV cache** (`kv_cache_dtype=half`):
  simpler, no dequant at gather, fits research goal. Indexer `k` is FP16.
- This requires the FP16 `triton_bf16_mla_sparse_interface` variant with
  `kv_cache_dtype` FP16 (adapt the XPU BF16 kernel, see below).

### Dtype strategy

- Run the model in **FP16** end-to-end (`--dtype=half`).
- Bypass the SM80 BF16 gate: `vllm/platforms/cuda.py:614` rejects BF16 below
  SM80. The Turing backend forces `torch.float16` for all compute tensors
  (quantized weights stay uint8/FP8/FP4 as serialized).
- The XPU Triton kernels assert BF16 (`q.dtype == torch.bfloat16`); the ported
  CUDA copies use FP16 and remove those asserts.

### Attention backend selection

- `vllm/models/deepseek_v4/nvidia/model.py:_select_dsv4_attn_cls` is NVIDIA-only.
- Turing model overrides attention selection to a new `TuringMLAAttention`
  class using `TritonMLABackend`-style flow with the ported sparse MLA Triton
  kernels. Default `attention_config.backend` = `FLASHMLA_SPARSE_DSV4` must map
  to the Turing class on SM75 (via `vllm/v1/attention/backends` dispatch or a
  direct override in the turing model).

## Files

New (all under `vllm/models/deepseek_v4/turing/`):

- `__init__.py` — re-exports.
- `model.py` — `DeepseekV4ForCausalLM`, `DeepseekV4DecoderLayer`,
  `DeepseekV4MLP`; port of `xpu/model.py` adapted for FP16 CUDA Triton.
  Reuses shared `DeepseekV4Attention` from `..attention` and `common/ops`.
- `attention.py` — `TuringMLAAttention` (Triton sparse MLA, FP16 KV).
- `sparse.py` — Triton FP16 sparse-MLA kernels (ported from XPU).
- `mtp.py` — `DeepSeekV4MTP` (torch/Triton; may `raise NotImplementedError`
  on SM75 if MTP dense paths need SM90+).
- `dspark.py` — `DSparkDeepseekV4ForCausalLM` (torch-only, V2; kept stub
  raising on SM75 unless needed).

Modified:

- `vllm/models/deepseek_v4/__init__.py` — add `turing/` branch.
- `vllm/platforms/cuda.py` — relax BF16 gate only if the model path requests
  BF16 despite `--dtype=half`; primary fix is forcing FP16 throughout the
  Turing backend (no `torch.bfloat16` tensors created on SM75).
- `vllm/model_executor/layers/sparse_attn_indexer.py:773` — replace the
  `has_deep_gemm()` hard-raise with a portable fallback path for SM75.
- `vllm/model_executor/layers/mhc.py` — ensure torch/Triton fallback is used
  with FP16 (tilelang path is SM90+; forced off on SM75).
- `vllm/model_executor/kernels/linear/__init__.py` — confirm
  `MarlinFP8ScaledMMLinearKernel` is first-choice on SM75 for block FP8.
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` — confirm
  `select_deepseek_v4_mxfp4_moe_backend` falls through to `MARLIN` on SM75
  (TRTLLM/DeepGEMM capability checks reject; Marlin accepts 7.5+).

## Data flow (decode step)

1. `DeepseekV4ForCausalLM.forward` runs embedding → decoder layers → LM head.
2. Each `DeepseekV4DecoderLayer`: input RMSNorm → `DeepseekV4MLP` (Marlin FP8
   block) + `TuringMLAAttention` (Triton sparse MLA).
3. Attention: compress KV → Triton `fused_compress_quant_cache` → sparse
   indexer (Triton logits + `_C.cooperative_topk`) → Triton sparse MLA FP16 →
   dequant o_proj → FP16 einsum.
4. Routed experts: FusedMoEFactory → `MarlinMoeKernel` (MXFP4 w13/w2) with
   `RoutedExperts` loading (round-hidden/intermediate to Marlin tile sizes).
5. MHC layers use `mhc_pre_torch`/`mhc_post_torch`/`hc_head_triton` fallbacks.
6. Output projections: Marlin FP8 block (dense).

## Error handling

- Any SM90+/SM100-only kernel that has no SM75 fallback must raise a clear
  `NotImplementedError` at module init or first forward (fail fast), naming the
  op and the required capability, rather than silently mis-computing.
- The indexer fallback path on SM75 must log once which kernels it is using.

## Testing

- Unit: port `tests/models/decoder_only/vision_language/.../test_deepseek_v4*` —
  adapt for Turing path where kernels differ. New tests:
  - `test_marlin_fp8_block_linear.py` — block-FP8 linear matches FP8 reference
    on a synthetic layer (dequant + compare).
  - `test_marlin_mxfp4_moe.py` — MXFP4 expert round-trip on SM75.
  - `test_turing_sparse_mla.py` — Triton FP16 sparse MLA vs reference
    `DeepseekV4FlashMLAAttention` on CPU/GPU for a small case.
  - `test_sm75_indexer_fallback.py` — indexer logits+topk via portable path.
- Model-level (on the cluster): load with a small max-model-len, run a few
  prompt/completion pairs; then a 100K+ token context request.
- Eval: same logits compared against the reference vLLM build (if available)
  or the FP8 PyTorch reference path, for identical prompts.
- Commands (all via `uv`/`.venv/bin/python`, never bare python3):
  - Lint: `pre-commit run ruff-check --all-files`
  - Type: `pre-commit run mypy-3.12 --all-files --hook-stage manual`
  - Tests: `.venv/bin/python -m pytest tests/... -v`

## Known limitations / out of scope

- **Performance**: no FP8/FP4/BF16 HW on Turing → every heavy op is a dequant +
  FP16 GEMM; expect 1–5 tok/s decode.
- **DSPark** (speculative decoding draft model) is out of scope (V2-runner +
  DeepGEMM-only); disabled on SM75.
- **MTP** (Multi-Token Prediction) only if portable paths suffice; otherwise
  disabled on SM75. Not required for the research goal.
- No `deep_gemm_mega_moe` backend; `use_mega_moe=False` forced.
- Long context tight on memory: quantized weights 159GB of 176GB leaves ~17GB
  for KV + activations; small max-model-len chunks or reduced max context may be
  needed; the Triton FP16 KV is tuned for the actual head dims once the
  checkpoint `config.json` is available.

## Open questions to resolve during implementation

- Exact MLA head dims / `compress_ratios` from the checkpoint `config.json`
  (not yet downloaded; HF fetch timed out). Tune Triton kernel block sizes
  accordingly.
- Whether `Mxfp4MoEMethod` (Marlin) accepts e8m0fnu (ue8m0) FP8 scales — DSv4
  FP4 checkpoints use e8m0fnu scales (`quant_config.py:is_scale_e8m0`). If not,
  route FP4 experts through the Mxfp4 EMULATION kernel (torch, portable) or a
  custom Triton FP4-dequant path.
- Whether `hc_head` / fused-post-pre torch fallbacks cover every MHC call site
  used by `nvidia/model.py` (needs a full call-site audit in the Turing port).
- Whether FP16 Triton `tl.dot` accepts the compressed MTP/attention shapes used
  by DSv4 (block-size constraints); adjust tiles if needed.
