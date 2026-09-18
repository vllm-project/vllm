# Reload Loading Layout Audit

完整调用关系和模型级示例见 [Reload 流程导读](reload-flow-walkthrough.md)。

## Lifecycle

The first validated local arrival calls `policy.prepare_for_load(state)`.
Preparation creates canonical loader inputs for all roles in that state.
Subsequent arrivals reuse them. Nonlocal, ignored and invalid slots do not
trigger preparation. Completion still requires every expected slot and every
dependency. `finish()` converts inputs and writes the bound runtime outputs.

`ReloadState.prepare_sources(reuse_roles=...)` implements shared allocation:

- The policy explicitly selects roles whose storage may be reused.
- A dense runtime allocation can lend its physical-order prefix to a contiguous
  checkpoint-layout view, including transposed or padded runtime tensors.
- The original runtime object, address, shape and strides remain unchanged.
- Different dtypes, insufficient capacity, missing targets and incompatible
  layouts use staging. Encoded scales are not reinterpreted as FP32.
- `preserve_checkpoint=True` disables reuse; destructive conversion uses working
  copies through `state.work()`.

This restores a writable layout, not old checkpoint values. It assumes a complete
reload with inference paused and expert placement fixed for the round.
Conversions may still allocate scratch. Failure after a runtime write is not
rolled back; this design does not support partial checkpoint updates.

## Policy Coverage

| Policy | Reusable input roles | Exceptions |
| --- | --- | --- |
| `CopyReloadPolicy` | All owned roles | Shared aliases remain owned by another state |
| `TensorFP8LinearReloadPolicy` | Weight, bias | Merged weight scales and activation scales stage |
| `XPUTensorFP8LinearReloadPolicy` | Weight, bias | Inherits tensor policy; KN storage provides canonical NK view |
| `AiterTensorFP8LinearReloadPolicy` | Weight, bias | Shuffle runs again at finish; FNUZ dtype mismatch stages |
| `B12xTensorFP8LinearReloadPolicy` | Bias | Opaque packed dataclass leaves are not reused as inputs |
| `BlockFP8LinearReloadPolicy` | All roles | Shared capacity/layout/dtype checks apply |
| `CPUBlockFP8LinearReloadPolicy` | Scale, bias; all roles in direct mode | Packed weight stays staged |
| `AiterBlockFP8LinearReloadPolicy` | All roles | Re-shuffle at finish; encoded/FNUZ dtype mismatch stages |
| `XPUBlockFP8LinearReloadPolicy` | All roles | Ragged scale expansion and BMM caches still rebuilt at finish |
| `B12xBlockFP8LinearReloadPolicy` | All roles | Native block layout; encoded-to-FP32 scale conversion stages |
| `DeepGEMMReloadPolicy` | All roles | Packed scales reuse only with compatible dtype and layout |
| `PlainMoEReloadPolicy` | All roles | Reduced scales may have insufficient runtime capacity |
| `PlannedMoEReloadPolicy` | All roles except CPU packed weights | Covers Triton/CUTLASS batched variants, TRTLLM, HPC, AITER, XPU and CPU |
| `CutlassMoEReloadPolicy` | W13/W2 weights and both block scales | Per-tensor scales stage |

Marlin and Humming linear/MoE policies retain their existing destination paths,
as requested for the current review scope. Their packing is not migrated here.
No change to backend selection or accepted quantization configurations is implied.

## Verification Boundaries

Tests check dense reuse versus staging, preservation, once-per-round preparation,
runtime identities/layouts, repeated reload values, and available native forward
and CUDA Graph comparisons. Platform conversion tests with substitute kernels
do not establish native AMD/XPU/AMX/SM120 execution correctness.
No peak-memory reduction or full-model accuracy result is claimed by this audit.

Final H200 regression: job `b968e0b39220`, actual `status=ok rc=0`,
**249 passed, 2 skipped**. The two native PyTorch block-GEMM cases require a
CUDA >= 12.9 build; the fixed runtime is CUDA 12.8. Separate cold/reload conversion
cases for that backend passed. The initial broad run exposed this environment
restriction before reload; a later test-fixture correction constructed non-dense
meta tensors directly because `.to("meta")` compacts their strides.

The selected tests span `tests/quantization/test_fp8.py`,
`tests/model_executor/model_loader/test_reload.py`, and
`tests/model_executor/kernels/test_b12x_linear.py`. Final workload log:
`/inspire/hdd/global_user/wangtongyu-25057/vllm-reload-trace-20260915/all-policy-loading-layout-review-04.log`.
