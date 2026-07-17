# RFC: Enhanced Copy-Back for Unmanaged Tensors During Weight Reload

## Problem

vLLM's layerwise weight reload mechanism preserves CUDA graph pointer validity
by copying new weight values into old tensor storage (`_copy_and_restore_kernel_tensors`).
However, this copy-back only covers tensors registered as `nn.Parameter` or `nn.Buffer`
via `get_layer_params_buffers()`.

Many kernels and quantization methods create **unmanaged CUDA tensors** during
`process_weights_after_loading()` (PWAL) — tensors stored as plain Python
attributes on the layer or on nested objects. After reload, PWAL recreates these
tensors at new device addresses, but CUDA graphs still hold pointers to the old
addresses. This causes:

- **Silent stale reads** — graph replay reads old/freed memory
- **Illegal memory access** — when old allocations are reclaimed
- **Graph-replay livelock** — when workspace memory is recycled

## Scope

This fix addresses **Category 1 (Storage Identity)** from
[#48312](https://github.com/vllm-project/vllm/issues/48312).

### Confirmed bugs covered

| Component | Unmanaged tensors |
|-----------|-------------------|
| Generic MLA (#48251) | `W_UV`, `W_UK_T` (plain attrs on attention layer) |
| Marlin linear (#48438) | `workspace`, `g_idx_sort_indices` (kernel object + layer attr) |
| CUTLASS FP8 MoE (#41670) | `ab_strides1/2`, `c_strides1/2` (nested on experts object) |
| Machete act-order (#48539) | `act_perm` (layer attr) |
| FlashInfer CUTLASS MoE | `gemm1_alpha`, `gemm1_beta`, `gemm1_clamp_limit` |

### High-risk candidates covered

AITER MLA FP4/FP8 derived weights, FlashInfer B12x scales, NVFP4 CUTLASS scales,
CUTLASS W4A8 strides, RDNA3 WNA16 scratch, WNA8O8 scale copies, TRT-LLM constants.

## Solution: Recursive Tensor Collection + Copy-Back

### Design

```
initialize_layerwise_reload:
  1. Save kernel_tensors (params + buffers)         ← existing
  2. collect_extra_tensors(layer) → snapshot         ← NEW
  3. Restore layer to meta device
  4. Wrap weight loaders

After PWAL (in _layerwise_process and _finalize_attention_layer):
  1. _copy_and_restore_kernel_tensors               ← existing
  2. copy_back_extra_tensors(layer, snapshot)        ← NEW
```

### Key mechanisms

**`collect_extra_tensors(layer)`** recursively walks the layer's attribute tree:
- Skips `_parameters`, `_buffers`, `_modules` (already managed)
- Skips `_`-prefixed attrs at layer level (PyTorch internals)
- Descends into nested Python objects (quant methods, kernel objects, experts)
- Records `(dotted_path, tensor)` for every CUDA tensor with unmanaged storage
- Does NOT add tensor `id` to visited set — allows alias detection

**`copy_back_extra_tensors(layer, slots)`** after PWAL:
- Resolves each path to find the newly-created tensor
- `old_tensor.data.copy_(new_tensor)` — writes new value into old address
- `set_by_path(layer, path, old_tensor)` — points attribute back to old tensor
- Deduplicates copy by `storage.data_ptr()` but restores every alias path
- Validates shape/dtype match, warns and skips on mismatch

### What it walks

| Container type | Traversal |
|---------------|-----------|
| `torch.Tensor` | Leaf — record if CUDA + unmanaged |
| `nn.Module` | Stop — has its own reload cycle |
| `dict` | Descend values |
| `list`/`tuple` | Descend elements |
| `functools.partial` | Descend args + keywords |
| `types.FunctionType` | Descend closure cells |
| Generic Python object | Descend `__dict__` attrs (skip `__dunder__`) |

Max depth: 8. Real-world deepest path: ~5.

## Files changed

| File | Change |
|------|--------|
| `reload/tensor_collector.py` | **New** — `collect_extra_tensors`, `copy_back_extra_tensors`, `resolve_path`, `set_by_path`, `_walk` |
| `reload/types.py` | Add `extra_tensor_slots` field to `LayerReloadingInfo` |
| `reload/layerwise.py` | Import + 3 call sites: collection in `initialize_layerwise_reload`, copy-back in `_layerwise_process` and `_finalize_attention_layer` |

## Trade-offs

### Why not an explicit registration API (#48478)?

The explicit registry approach requires every kernel/quantization author to
manually register their graph-visible tensors. This is the correct long-term
production design (fail-closed), but:
- It requires changes to every affected kernel (13+ sites)
- New kernels that forget to register will silently break
- Our approach serves as a **safety net / fallback** that catches everything

The two approaches are complementary: explicit registration for production
contracts, recursive collection as a defense-in-depth backup.

### Why not FX graph passes?

FX passes operate on the computation graph (Inductor IR), not on `nn.Module`
attribute trees. They cannot see plain Python attributes or nested kernel
objects. Wrong abstraction level for this problem.

## Testing

1. **Unit tests**: Mock layers with unmanaged tensors, verify collect/copy-back
   preserves `data_ptr` across simulated PWAL
2. **Integration**: Qwen3-0.6B reload with CUDA graph capture, verify
   KL-divergence = 0 between cold-load and warm-reload
3. **MLA mock test**: Verify W_UV/W_UK_T address preservation
4. **Storage Identity regression**: All 13 confirmed/candidate cases from #48312
