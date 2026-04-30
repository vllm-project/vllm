# Cursor AI Prompt: Extend vLLM IR Ops for Optional Output Support

## Context
I am working on the vLLM project to resolve Issue #40607. The goal is to eliminate duplicate quantization logic by replacing the legacy `per_token_group_quant_fp8` function with the standardized `ir.ops.dynamic_group_quant_fp8`.

## The Technical Blocker
The file:
`vllm/model_executor/layers/quantization/utils/deep_gemm_moe.py`

currently uses a pre-allocated tensor (`out_q`) to store quantization results.

Standard vLLM IR operations do not support passing an external output tensor—they always allocate a new one.

To migrate `deep_gemm_moe.py` without performance regressions, we must extend the IR Op infrastructure.

## Objective
Implement "Route 2":
Extend the `dynamic_group_quant_fp8` IR operation to support an optional `out` (output tensor) argument, following the `maybe_inplace` pattern.

## Instructions for Cursor

### Step 1: Analyze Existing Patterns
- Examine PR #36823
- Search the codebase for `maybe_inplace`
- Identify how IR ops like `fused_add_rms_norm` handle optional output tensors

### Step 2: Modify the IR Op Definition
- Locate the definition and registration of `dynamic_group_quant_fp8`
- Likely in:
  - `vllm/v1/attention/backends/rocm_attn.py`
  - or the general IR registry

- Update the operation signature to include:
  out: Optional[torch.Tensor] = None

- Update dispatch logic:
  - If `out` is provided → write results into it (in-place behavior)
  - If `out` is None → allocate and return a new tensor

### Step 3: Refactor the Caller
In:
`vllm/model_executor/layers/quantization/utils/deep_gemm_moe.py`

- Import `ir.ops.dynamic_group_quant_fp8`
- Replace:
  per_token_group_quant_fp8(...)
- With:
  dynamic_group_quant_fp8(..., out=out_q)

### Step 4: Cleanup
After confirming migration works:
- Delete `per_token_group_quant_fp8` from:
  `vllm/model_executor/layers/quantization/utils/fp8_utils.py`

## Files to Reference (use @ in Cursor)
- @vllm/model_executor/layers/quantization/utils/deep_gemm_moe.py
- @vllm/model_executor/layers/quantization/utils/fp8_utils.py
- @vllm/v1/attention/backends/rocm_attn.py

## Branch Strategy
- Branch Name: feat/ir-op-output-variant-quant
- Commit Message: feat(ir): add optional output tensor support to dynamic_group_quant_fp8

## How to Use in Cursor
1. Save as `cursor_instructions.md`
2. Open Cursor Chat (Cmd + L)
3. Run:
   Follow the instructions in @cursor_instructions.md to implement Route 2 for the IR Op migration.
