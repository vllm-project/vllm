# Ascend NPU Migration Log (vLLM 0.11.0)

## Scope
- Base branch: `v0.11.0`
- Working branch: `feat/ascend-npu-adapt-v0.11.0`
- Source reference (legacy): `E:\vllm_npu\vllm_0.2.7_ascend`
- Official reference: `E:\vllm_npu\vllm-ascend` on branch `v0.11.0-dev`

## What vllm_0.2.7_ascend changed
- Replaced CUDA-specific paths with NPU paths in worker/model/cache initialization.
- Added an Ascend attention backend (`ascend_attn.py`) and routed attention to it.
- Added NPU custom op wrappers for:
- `npu_swiglu`
- `npu_rms_norm` / `npu_add_rms_norm`
- `_npu_rotary_embedding`
- Adjusted distributed backend usage from `nccl` to `hccl`.
- Disabled or bypassed some CUDA-only attention/cache paths in old flash/paged attention files.

## Official vllm-ascend (v0.11.0-dev) pattern observed
- Uses out-of-tree platform plugin (`NPUPlatform`) rather than hard-forking all in-tree code.
- Uses `adapt_patch` split into:
- global/platform patch
- worker/runtime patch
- Registers Ascend attention backends through platform selection (`get_attn_backend_cls`).
- Keeps device-specific behavior mostly in plugin + patches + custom ops.

## Changes made in this branch
1. `vllm/_custom_ops.py`
- Added optional `torch_npu` import.
- Added NPU path for `rotary_embedding` using `torch_npu._npu_rotary_embedding`.
- Added NPU path for `rms_norm` using `torch_npu.npu_rms_norm` and `out.copy_`.
- Added NPU path for `fused_add_rms_norm` using `torch_npu.npu_add_rms_norm`.
- Fixed repetition-penalty CUDA dispatch guard from `logits.is_cuda` to `logits.device.type == "cuda"` to avoid wrong dispatch on non-CUDA backends.

2. `vllm/model_executor/layers/activation.py`
- Added optional `torch_npu` import.
- Added `SiluAndMul.forward_oot`:
- use `torch_npu.npu_swiglu` when tensor device is NPU
- fallback to native implementation otherwise

## Why these changes first
- These are low-coupling and high-reuse primitives used by many models.
- They map directly to legacy 0.2.7 Ascend optimizations.
- They are consistent with official vllm-ascend practice: keep NPU specialization in backend/custom-op paths first, avoid large invasive scheduler/engine rewrites in the first migration step.

## Not migrated yet (next phase)
- Full Ascend attention backend wiring for v1 scheduler/data path.
- HCCL-specific distributed coordinator customization.
- Full platform plugin registration and patch package in-tree/out-of-tree integration.
- KV-cache layout and graph-capture policy tuning specific to Ascend hardware.
