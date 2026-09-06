# CUTLASS FP8 MoE reload: phases 1–3

Base: `5968e6b39f` (FP8 dense per-tensor selective reload).
Local worktree: `/home/tttony/workspace/vllm-fp8-moe-reload`.
Remote worktree: `/inspire/hdd/global_user/wangtongyu-25057/vllm-fp8-moe-reload`.
The original local and remote workspaces are not used for this implementation.

## Scope and phases

1. **Source/runtime contracts — implemented.** Include serialized per-tensor
   and block-wise FP8 from the first phase. Use the selectable
   `FLASHINFER_CUTLASS` backend and the real `RoutedExperts.weight_loader`.
   Save checkpoint shape/dtype/loader metadata before cold PWAL.
2. **Conversion and runtime allocation — implemented.**
   `_prepare_moe_runtime` allocates final weight/scale shells from source
   metadata before conversion. `_convert_moe_runtime` performs requantization
   and backend layout conversion. `_install_moe_kernel` fills cold runtime
   storage and constructs the kernel. Reload uses cloned source tensors and
   an isolated layer/config view so padding does not mutate the live config.
3. **Staging, FINISH and kernel reuse — implemented.** Load into source-layout
   staging, track actual expert/row/column writes, validate source completeness
   and every output layout before copying. Reuse kernel, Parameter objects,
   storage and quant-config-owned alpha/reciprocal tensors. Repeated refresh
   with no staged data is a no-op.
4. **Later work, not enabled here.** Other MoE backends, EPLB and FNUZ. EPLB
   explicitly stays on fallback. Native VLLM_CUTLASS variants are already
   disabled by the method's existing backend selector.

## Layouts

For gated experts, E is local expert count, N intermediate size, K hidden size.

| Tensor | Checkpoint | FlashInfer CUTLASS runtime |
| --- | --- | --- |
| w13_weight | E × 2N × K, gate/up | up/gate, per-tensor N alignment padding |
| w2_weight | E × K × N | per-tensor N alignment padding |
| w13_weight_scale | E × 2 | E, max scale with gate/up requantization |
| w2_weight_scale | E | E |
| w13_weight_scale_inv | E × 2ceil(N/Bn) × ceil(K/Bk) | swapped block rows, existing minimum clamp |
| w2_weight_scale_inv | E × ceil(K/Bn) × ceil(N/Bk) | existing minimum clamp |
| static input scales | E per projection | scalar maximum without EPLB |

Per-tensor g1/g2 alpha = weight scale × activation scale. a1/a2 global scale
= reciprocal activation scale. Kernel-held references are updated with copy_.
Identical w1/w3 activation scales can share one source slot; a repeated shard
or conflicting alias is rejected. Conflicting values are not silently
resolved by checkpoint iteration order.

## H200 evidence

All Python tests ran inside H200 with the fixed interpreter:
`/inspire/hdd/global_user/wangtongyu-25057/miniconda3/envs/vllm/bin/python`.
Use `CUDA_VISIBLE_DEVICES=1`, `HF_HUB_OFFLINE=1`, and PYTHONPATH pointing to
the dedicated remote worktree. No local GPU tests were run.

- `tests/quantization/test_fp8.py -k moe_ -q --tb=short`: **20 passed**, H200
  `status=ok rc=0 seconds=18.79` on the final runtime-shell implementation.
- Actual lifecycle cases: per-tensor N=256, per-tensor N=257 (padding),
  block-wise N=256 with 128×128 blocks; four experts, top-k=2. Two reloads,
  runtime unchanged before FINISH, no PWAL/kernel reconstruction, stable
  Parameters and kernel/config-derived pointers, exact cold-B runtime tensor
  and actual CUTLASS kernel-output comparisons.
- Other selected tests cover real expert weight/scale loaders, fused 3D
  expert copies, overlap rejection, missing expert/scale rejection before
  conversion, invalid destination rejection before any write, real W31 and
  requant conversions, and alpha/reciprocal refresh.
- Dense FP8 regression: `-k per_tensor_refresh_without_pwal`, **20 passed**.
- Ruff 0.14.0: the three modified Python files formatted and checked in H200.

## Reusing the compiled backend

Set FLASHINFER_WORKSPACE_BASE to the remote worktree's `jit-cuda130` directory.
For rebuilding, MAX_JOBS=4 and CPATH containing only these fixed-environment
subdirectories were used:
`lib/python3.12/site-packages/nvidia/cublas/include` and
`lib/python3.12/site-packages/nvidia/cuda_nvrtc/include`.
Do not add the broad cu13/include directory: it conflicted with nvcc 13.0.88.
Initial successful build/test job `320454c75c06`: rc=0, two original lifecycle
cases passed in 3204.44 seconds. Subsequent runs reuse its cache.

## Limitations for review

These are real-layer/kernel tests, not a pretrained-model quality evaluation,
multi-node/EP acceptance or CUDA graph replay test. Conversion allocates
staging and temporary tensors. Prevalidation prevents ordinary layout errors
from causing partial writes but is not model-wide transactional rollback if
a GPU copy fails. Cold load may still install activation and derived scale
Parameters; the no-replacement invariant applies to runtime reload writes.
