# DeepSeek V4.1 FlashMLA Megakernel Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run DeepSeek V4.1 attention on SM100 through FlashMLA's fused Q-RoPE + sparse-attention + inverse-RoPE + FP8-cast kernel, feeding the `wo_a` einsum directly, and add the V4.1 fp8 (528 B) sliding-window and V4.1 fp4 (288 B) compressed KV cache formats behind `--kv-cache-dtype nvfp4_ds_mla`.

**Architecture:** A new attention subclass `DeepseekV4FlashMLAFusedAttention` (selected by `AttentionConfig.dsv4_fused_attention`) permutes `wq_b` rows and `wo_a` columns once at load time so the MXFP8 GEMMs produce and consume the kernel's chunk-interleaved layouts, writes the post-`wo_a` tensor `z` from the eager attention region, and keeps the existing split-KV FlashMLA path as a per-step fallback for small decode batches. KV formats are described by one layout table; insert and gather kernels dispatch on bytes per token.

**Tech Stack:** vLLM (this worktree, Python 3.13 venv at `/home/yongye/sra/.venv`), FlashMLA fork at `/home/yongye/FlashMLA` (stable-ABI port in progress there, consumed via `FLASH_MLA_SRC_DIR`), DeepGEMM (vendored `vllm/third_party/deep_gemm`), Triton, CMake/Ninja preset `release` (`cmake-build-release/`).

**Spec:** `20260910-v41-megakernel-vllm-integration-plan.md` (repo root of this worktree). Source description of the kernel: `20260910-upstream-pr221-v41-megakernel.md`.

## Global Constraints

- Python: always `/home/yongye/sra/.venv/bin/python` (never system python, never bare pip). Run from the worktree root `/home/yongye/sra/.claude/worktrees/v41-megakernel`; import vLLM from the worktree with `sys.path.insert(0, ".")` inside scripts or `python -m pytest` from the worktree root (do not export `PYTHONPATH` in the shell: the worktree guard refuses it).
- Kernel build: `cmake --preset release` then `ninja -C cmake-build-release _flashmla_C` and `cp cmake-build-release/_flashmla_C.abi3.so vllm/` (memory note: ccache can return stale objects on this filesystem; verify with `strings -a <obj> | grep <schema>`).
- FlashMLA source: `FLASH_MLA_SRC_DIR=/home/yongye/FlashMLA` (cache variable in the untracked `CMakeUserPresets.json`). Do not edit `/home/yongye/FlashMLA`; another session owns that port. The op schemas it registers (read 2026-09-10 06:18) are `fused_norm_rope_attn_rope_cast_fwd(...) -> Tensor[]` and `fused_norm_rope_attn_rope_cast_decode(...) -> Tensor[]` with the argument order in Task 3.
- Line length 88, Google-style docstrings, minimal comments (AGENTS.md). `pre-commit run --files <changed files>` before every commit.
- Fixed model geometry (DSv4.1, `ckpt20260903`): `head_dim=512`, `qk_rope_head_dim=64`, `num_attention_heads=64`, `o_groups=8` (8 heads per group), `q_lora_rank=1280`, `o_lora_rank=1024`, `sliding_window=128`, `index_topk=512`, MXFP8 weights (`weight_block_size=[32,32]`, ue8m0). TP4 gives 16 local heads, 2 local groups, `padded_heads=64`.
- Kernel constraints: `h_q in {64,128}`, `q.stride(1)==512`, `topk % 8 == 0`, int32 `token_positions`, fp32 `cos_sin_cache [*,64]`, `num_per_channels=32`, `enable_q_norm=False`. Decode has no split-KV.
- GPU: GB200 (SM100). All GPU tests skip when `is_flashmla_fused_sparse_supported()` is false.
- Commit each task on branch `v41-megakernel` with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## Part 1: build, wrappers, spike

### Task 1: Layout permutation helpers (pure torch)

**Files:**

- Create: `vllm/models/deepseek_v4_1/common/ops/fused_layout.py`
- Test: `tests/kernels/test_dsv41_fused_layout.py`

**Interfaces:**

- Produces: `q_fused_permutation(num_heads, head_dim=512) -> LongTensor[num_heads*head_dim]` with `fused = standard[perm]`; `o_fused_permutation(heads_per_group=8, head_dim=512) -> LongTensor[G*D]`; `o_fused_chunk_permutation(heads_per_group=8, head_dim=512) -> LongTensor[G*D//32]`; `inverse_permutation(perm)`; `permute_q_to_fused(q[N,H,D]) -> [N,H,D]`; `permute_q_from_fused(q)`; `permute_wq_b_(weight, weight_scale, num_local_heads)`; `permute_wo_a_(weight, weight_scale, heads_per_group=8)`.

- [x] **Step 1: Write the failing tests**

```python
# tests/kernels/test_dsv41_fused_layout.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index permutations between the standard [h, d] layout and the layouts the
FlashMLA fused sparse-attention kernel reads (Q) and writes (O)."""

import torch

from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    inverse_permutation,
    o_fused_chunk_permutation,
    o_fused_permutation,
    permute_q_from_fused,
    permute_q_to_fused,
    permute_wo_a_,
    permute_wq_b_,
    q_fused_permutation,
)


def test_q_fused_permutation_matches_kernel_formula():
    num_heads, head_dim = 4, 64
    perm = q_fused_permutation(num_heads, head_dim)
    for h in range(num_heads):
        for d in range(head_dim):
            fused = (d // 16) * (num_heads * 16) + h * 16 + d % 16
            assert perm[fused].item() == h * head_dim + d


def test_o_fused_permutation_matches_kernel_formula():
    perm = o_fused_permutation(8, 512)
    for h in range(8):
        for c in range(16):
            for j in (0, 17, 31):
                fused = (c * 8 + h) * 32 + j
                assert perm[fused].item() == h * 512 + c * 32 + j
    chunk_perm = o_fused_chunk_permutation(8, 512)
    assert torch.equal(chunk_perm, perm[::32] // 32)


def test_inverse_permutation_round_trip():
    perm = q_fused_permutation(16, 512)
    inv = inverse_permutation(perm)
    assert torch.equal(perm[inv], torch.arange(perm.numel()))
    q = torch.randn(3, 16, 512)
    assert torch.equal(permute_q_from_fused(permute_q_to_fused(q)), q)


def test_permuted_wq_b_produces_fused_q():
    torch.manual_seed(0)
    n, heads, k = 5, 16, 1280
    x = torch.randn(n, k)
    w = torch.randn(heads * 512, k)
    scale = torch.arange(heads * 512, dtype=torch.int32).view(-1, 1)
    scale = scale.expand(-1, k // 32).to(torch.uint8)
    w_perm, scale_perm = w.clone(), scale.clone()
    permute_wq_b_(w_perm, scale_perm, heads)
    q_std = (x @ w.T).view(n, heads, 512)
    torch.testing.assert_close((x @ w_perm.T).view(n, heads, 512),
                               permute_q_to_fused(q_std))
    perm = q_fused_permutation(heads, 512)
    assert torch.equal(scale_perm, scale[perm])


def test_permuted_wo_a_consumes_fused_o():
    torch.manual_seed(0)
    n, rank, group_in = 5, 1024, 8 * 512
    o_std = torch.randn(n, group_in)
    w = torch.randn(rank, group_in)
    scale = torch.arange(group_in // 32, dtype=torch.int32).view(1, -1)
    scale = scale.expand(rank, -1).to(torch.uint8)
    w_perm, scale_perm = w.clone(), scale.clone()
    permute_wo_a_(w_perm, scale_perm, heads_per_group=8)
    o_fused = o_std[:, o_fused_permutation(8, 512)]
    torch.testing.assert_close(o_fused @ w_perm.T, o_std @ w.T)
    assert torch.equal(scale_perm, scale[:, o_fused_chunk_permutation(8, 512)])


def test_permute_helpers_accept_fp8_storage():
    w = torch.randn(16 * 512, 64).to(torch.float8_e4m3fn)
    s = torch.zeros(16 * 512, 2, dtype=torch.uint8)
    permute_wq_b_(w, s, 16)
    w2 = torch.randn(1024, 8 * 512).to(torch.float8_e4m3fn)
    s2 = torch.zeros(1024, 128, dtype=torch.uint8)
    permute_wo_a_(w2, s2)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_fused_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: ... fused_layout`

- [x] **Step 3: Write the module**

```python
# vllm/models/deepseek_v4_1/common/ops/fused_layout.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout permutations for FlashMLA's fused sparse-attention kernel.

The fused kernel reads Q with 16-element head-dim chunks interleaved across
heads and writes O with 32-element chunks interleaved across the 8 heads of a
``wo_a`` group. Every permutation ``perm`` here satisfies
``fused = standard[perm]``; ``wq_b`` rows and ``wo_a`` columns are permuted
once at load time so the GEMMs produce and consume the fused layouts directly.
"""

import torch

HEAD_DIM = 512
Q_CHUNK = 16
O_CHUNK = 32
WV_GROUP_SIZE = 8


def q_fused_permutation(num_heads: int, head_dim: int = HEAD_DIM) -> torch.Tensor:
    """``fused[(d // 16) * (H * 16) + h * 16 + d % 16] = standard[h * D + d]``."""
    h = torch.arange(num_heads).view(num_heads, 1)
    d = torch.arange(head_dim).view(1, head_dim)
    fused_index = (d // Q_CHUNK) * (num_heads * Q_CHUNK) + h * Q_CHUNK + d % Q_CHUNK
    perm = torch.empty(num_heads * head_dim, dtype=torch.long)
    perm[fused_index.reshape(-1)] = torch.arange(num_heads * head_dim)
    return perm


def o_fused_permutation(
    heads_per_group: int = WV_GROUP_SIZE, head_dim: int = HEAD_DIM
) -> torch.Tensor:
    """``fused[(c * G + h) * 32 + j] = standard[h * D + c * 32 + j]`` per group."""
    h = torch.arange(heads_per_group).view(-1, 1, 1)
    c = torch.arange(head_dim // O_CHUNK).view(1, -1, 1)
    j = torch.arange(O_CHUNK).view(1, 1, -1)
    fused_index = (c * heads_per_group + h) * O_CHUNK + j
    perm = torch.empty(heads_per_group * head_dim, dtype=torch.long)
    perm[fused_index.reshape(-1)] = torch.arange(heads_per_group * head_dim)
    return perm


def o_fused_chunk_permutation(
    heads_per_group: int = WV_GROUP_SIZE, head_dim: int = HEAD_DIM
) -> torch.Tensor:
    """Per-32-element-chunk form of :func:`o_fused_permutation` (for scales)."""
    return o_fused_permutation(heads_per_group, head_dim)[::O_CHUNK] // O_CHUNK


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device)
    return inv


def permute_q_to_fused(q: torch.Tensor) -> torch.Tensor:
    """``[N, H, D]`` standard layout -> the same nominal shape, fused layout."""
    n, h, d = q.shape
    perm = q_fused_permutation(h, d).to(q.device)
    return q.reshape(n, h * d)[:, perm].view(n, h, d)


def permute_q_from_fused(q: torch.Tensor) -> torch.Tensor:
    n, h, d = q.shape
    inv = inverse_permutation(q_fused_permutation(h, d)).to(q.device)
    return q.reshape(n, h * d)[:, inv].view(n, h, d)


def _bytes_view(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t


def permute_wq_b_(
    weight: torch.Tensor, weight_scale: torch.Tensor, num_local_heads: int
) -> None:
    """Permute the rows of an MXFP8 ``wq_b`` shard and its per-row scale in place."""
    head_dim = weight.shape[0] // num_local_heads
    perm = q_fused_permutation(num_local_heads, head_dim).to(weight.device)
    for t in (weight, weight_scale):
        b = _bytes_view(t)
        b.copy_(b[perm])


def permute_wo_a_(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    heads_per_group: int = WV_GROUP_SIZE,
) -> None:
    """Permute the input columns of an MXFP8 ``wo_a`` shard and its per-32 scale."""
    head_dim = weight.shape[1] // heads_per_group
    perm = o_fused_permutation(heads_per_group, head_dim).to(weight.device)
    w = _bytes_view(weight)
    w.copy_(w[:, perm])
    chunk_perm = o_fused_chunk_permutation(heads_per_group, head_dim)
    s = _bytes_view(weight_scale)
    s.copy_(s[:, chunk_perm.to(weight.device)])
```

- [x] **Step 4: Run the tests**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_fused_layout.py -v`
Expected: 6 passed

- [x] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4_1/common/ops/fused_layout.py tests/kernels/test_dsv41_fused_layout.py
git commit -m "[DSv4.1] Add fused-kernel layout permutation helpers"
```

### Task 2: Build `_flashmla_C` from the PR-221 FlashMLA tree

> **Revision (user direction, 2026-09-10):** pin `flashmla.cmake` to upstream
> `deepseek-ai/FlashMLA` at `07a1089857b63e74e3133630c02b083b75e8d4b2` ("Add
> kernels for DeepSeek v4.1 (#221)") instead of `FLASH_MLA_SRC_DIR`. Upstream is
> a pybind11 module, so `_flashmla_C` is built without `USE_SABI` /
> `TORCH_TARGET_VERSION`, linked against `torch_python`, and the vendored
> Python files get `import vllm._flashmla_C as flash_mla_cuda`. Upstream has no
> `csrc/extension` (the `_flashmla_extension_C` target is built only when that
> directory exists) and its Python interface has no `out=`, which
> `vllm/v1/attention/ops/flashmla.py` now emulates with a copy. The steps below
> describe the original `FLASH_MLA_SRC_DIR` variant; the committed
> `flashmla.cmake` is the reference.

**Files:**

- Modify: `cmake/external_projects/flashmla.cmake:76-140` (source list, includes)
- Modify (untracked, local only): `CMakeUserPresets.json` (add `FLASH_MLA_SRC_DIR`, `CMAKE_CUDA_ARCHITECTURES`)

**Interfaces:**

- Produces: `vllm/_flashmla_C.abi3.so` exposing `torch.ops._flashmla_C.{sparse_decode_fwd, sparse_prefill_fwd, dense_decode_fwd, dense_prefill_fwd, fused_norm_rope_attn_rope_cast_fwd, fused_norm_rope_attn_rope_cast_decode, permute_q_b_proj, permute_wv_proj}`.

- [x] **Step 1: Add the preset cache variables**

Edit `CMakeUserPresets.json` `cacheVariables` (untracked file, worktree copy) and add:

```json
"FLASH_MLA_SRC_DIR": "/home/yongye/FlashMLA",
"CMAKE_CUDA_ARCHITECTURES": "100a"
```

- [x] **Step 2: Replace the FlashMLA source and include lists**

In `cmake/external_projects/flashmla.cmake` replace the `set(FlashMLA_SOURCES ...)` block with the PR-221 tree (mirrors the FlashMLA `setup.py` list without the dense backward, which vLLM does not register). Every path is `${flashmla_SOURCE_DIR}/csrc/...`:

```cmake
    set(FlashMLA_SOURCES
        ${flashmla_SOURCE_DIR}/csrc/api/api.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/sparse_prefill.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/sparse_decode.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/dense_decode.cpp
        ${flashmla_SOURCE_DIR}/csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp

        # Misc kernels for decoding
        ${flashmla_SOURCE_DIR}/csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/smxx/decode/combine/combine.cu

        # sm90 dense decode
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/dense/instantiations/fp16.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/dense/instantiations/bf16.cu

        # sm90 sparse decode
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v4_persistent_h128.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_persistent_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/decode/sparse/instantiations/v32_persistent_h128.cu

        # sm90 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu

        # sm100 dense prefill
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu

        # sm100 sparse prefill
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_k512.cu

        # sm100 sparse decode (head64, and native head128). The fork's V3.2
        # nvfp4_ds_mla instantiation is not in this tree yet (FlashMLA PR #18
        # re-add pending).
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v32_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v4_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v4_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_splitkv.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu

        # sm100 fused norm + rope + sparse attn + rope + fp8 cast (DeepSeek V4 / V4.1).
        # The API dispatches on enable_q_norm at runtime, so both variants link.
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_prefill_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_prefill_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_prefill_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_prefill_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h64_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h128_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h64_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h64_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h128_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41_h128_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h64_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h64_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h128_decode_norm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v41fp4_h128_decode_nonorm.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu
        ${flashmla_SOURCE_DIR}/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu
    )
```

and replace `set(FlashMLA_INCLUDES ...)` with

```cmake
    set(FlashMLA_INCLUDES
        ${flashmla_SOURCE_DIR}/csrc
        ${flashmla_SOURCE_DIR}/csrc/kerutils/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/include
        ${flashmla_SOURCE_DIR}/csrc/cutlass/tools/util/include
    )
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
        # CUDA 13 moved cuda/std headers under include/cccl; nvcc finds them,
        # the host compiler building csrc/api/*.cpp does not.
        list(APPEND FlashMLA_INCLUDES ${CUDA_TOOLKIT_ROOT_DIR}/include/cccl)
    endif()
```

- [x] **Step 3: Configure and build**

```bash
cmake --preset release 2>&1 | tail -5
/home/yongye/sra/.venv/bin/ninja -C cmake-build-release _flashmla_C 2>&1 | tail -3
cp cmake-build-release/_flashmla_C.abi3.so vllm/
grep -n "out: Optional" vllm/third_party/flashmla/flash_mla_interface.py | head -3
```

Expected: build succeeds; the last grep must print the `out:` parameters of `flash_mla_with_kvcache` and `flash_mla_sparse_fwd`. If it prints nothing, the FlashMLA checkout's Python interface has not yet restored the fork's `out=` parameters: restore the previous vendored copy with `cp /home/yongye/sra/vllm/third_party/flashmla/flash_mla_interface.py vllm/third_party/flashmla/` for local testing and tell the user (their FlashMLA port must add `out=` back before vLLM's decode path works from a clean configure).

If nvcc rejects a fused-kernel source for an `sm_100a`-only feature under the `10.0f` gencode, change the `SUPPORT_ARCHS` entry `"10.0f"` to `"10.0a"` for `CMAKE_CUDA_COMPILER_VERSION >= 12.9` in the same file and rebuild.

- [x] **Step 4: Verify the ops and the existing paths**

```bash
/home/yongye/sra/.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
import torch, vllm._flashmla_C
ops = torch.ops._flashmla_C
for name in ("sparse_decode_fwd", "sparse_prefill_fwd", "fused_norm_rope_attn_rope_cast_fwd",
             "fused_norm_rope_attn_rope_cast_decode"):
    print(name, hasattr(ops, name))
PY
/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_sparse.py -v -k "smoke"
```

Expected: four `True`; the smoke tests pass (they exercise V3.2 fp8 decode/prefill through the regenerated interface).

- [x] **Step 5: Commit**

```bash
git add cmake/external_projects/flashmla.cmake
git commit -m "[Build] Point the FlashMLA extension at the PR-221 source tree"
```

### Task 3: vLLM wrappers for the fused ops

**Files:**

- Modify: `vllm/v1/attention/ops/flashmla.py` (append after `is_flashmla_sparse_supported`)
- Test: `tests/kernels/attention/test_flashmla_fused_sparse.py` (new)

**Interfaces:**

- Produces:
    - `is_flashmla_fused_sparse_supported() -> tuple[bool, str | None]`
    - `flash_mla_fused_sparse_prefill(q, kv, indices, sm_scale, token_positions, cos_sin_cache, n_wv_group, attn_sink=None, topk_length=None) -> (out_fp8 [s_q, n_wv_group, 4096] e4m3, out_sf [s_q, n_wv_group, 32] int32, max_logits [s_q, h_q] f32, lse [s_q, h_q] f32)`
    - `flash_mla_fused_sparse_decode(q, k_cache, indices, sm_scale, token_positions, cos_sin_cache, n_wv_group, attn_sink=None, topk_length=None, extra_k_cache=None, extra_indices=None, extra_topk_length=None) -> (out_fp8, out_sf, lse)`

- [x] **Step 1: Write the failing smoke test**

```python
# tests/kernels/attention/test_flashmla_fused_sparse.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashMLA fused norm + RoPE + sparse attention + RoPE + FP8 cast kernel."""

import pytest
import torch

import vllm.v1.attention.ops.flashmla as fm


def _skip_unless_supported():
    ok, reason = fm.is_flashmla_fused_sparse_supported()
    if not ok:
        pytest.skip(reason)


def test_fused_decode_smoke_shapes():
    _skip_unless_supported()
    device = torch.device("cuda")
    s_q, h_q, topk, page = 3, 64, 128, 32
    q = torch.zeros(s_q, h_q, 512, dtype=torch.bfloat16, device=device)
    k_cache = torch.zeros(4, page, 1, 584, dtype=torch.uint8, device=device)
    indices = torch.full((s_q, topk), -1, dtype=torch.int32, device=device)
    indices[:, 0] = 0
    positions = torch.zeros(s_q, dtype=torch.int32, device=device)
    cos_sin = torch.zeros(16, 64, dtype=torch.float32, device=device)
    cos_sin[:, :32] = 1.0
    out_fp8, out_sf, lse = fm.flash_mla_fused_sparse_decode(
        q, k_cache, indices, 512**-0.5, positions, cos_sin, n_wv_group=h_q // 8
    )
    assert out_fp8.shape == (s_q, h_q // 8, 4096)
    assert out_fp8.dtype == torch.float8_e4m3fn
    assert out_sf.shape == (s_q, h_q // 8, 32) and out_sf.dtype == torch.int32
    assert out_sf.stride(0) == 1
    assert lse.shape == (s_q, h_q)
```

- [x] **Step 2: Run it to verify it fails**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_fused_sparse.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'is_flashmla_fused_sparse_supported'`

- [x] **Step 3: Add the wrappers**

Append to `vllm/v1/attention/ops/flashmla.py`:

```python
def is_flashmla_fused_sparse_supported() -> tuple[bool, str | None]:
    """FlashMLA's fused Q-RoPE + sparse attention + O-RoPE + FP8 cast kernel."""
    is_available, maybe_reason = is_flashmla_sparse_supported()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(100):
        return False, "FlashMLA fused sparse attention requires sm_10x GPUs."
    if not hasattr(torch.ops._flashmla_C, "fused_norm_rope_attn_rope_cast_decode"):
        return (
            False,
            "vllm._flashmla_C was built without the fused sparse attention ops.",
        )
    return True, None


_FUSED_ROPE_DIM = 64
_FUSED_QUANT_GROUP = 32


def flash_mla_fused_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    token_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_wv_group: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused sparse prefill over a non-paged bf16 ``kv``.

    Args:
        q: ``[s_q, h_q, 512]`` bf16 in the fused (chunk-interleaved) layout,
            before RoPE.
        kv: ``[s_kv, 1, 512]`` bf16, RoPE already applied.
        indices: ``[s_q, 1, topk]`` int32; entries ``< 0`` or ``>= s_kv`` are
            skipped.
        token_positions: ``[s_q]`` int32 RoPE positions of the queries.
        cos_sin_cache: ``[max_pos, 64]`` fp32, cos in ``[:, :32]``.
        n_wv_group: ``h_q // 8``.

    Returns:
        ``(out_fp8 [s_q, n_wv_group, 4096], out_sf [s_q, n_wv_group, 32] int32,
        max_logits [s_q, h_q], lse [s_q, h_q])``. ``out_fp8`` holds the
        inverse-RoPE'd output in the fused chunk order; ``out_sf`` is
        DeepGEMM's TMA-aligned packed-ue8m0 layout (``stride(0) == 1``).
    """
    out_fp8, out_sf, max_logits, lse = (
        torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_fwd(
            q,
            kv,
            indices,
            sm_scale,
            512,
            attn_sink,
            topk_length,
            False,
            0.0,
            token_positions,
            False,
            _FUSED_ROPE_DIM,
            cos_sin_cache,
            n_wv_group,
            _FUSED_QUANT_GROUP,
            True,
            True,
            True,
        )
    )
    return out_fp8, out_sf, max_logits, lse


def flash_mla_fused_sparse_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    token_positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_wv_group: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused sparse decode over paged quantized caches (batch flattened).

    ``k_cache`` / ``extra_k_cache`` are ``[num_blocks, page, 1, bytes]`` with
    the format detected from ``bytes`` (584 V4, 528 V4.1 fp8, 288 V4.1 fp4;
    fp4 only for ``extra_k_cache`` next to a V4.1 fp8 ``k_cache``).
    ``indices`` / ``extra_indices`` are ``[s_q, topk]`` int32 slot ids
    (``block * page + offset``, ``-1`` invalid). Returns
    ``(out_fp8, out_sf, lse)`` as in :func:`flash_mla_fused_sparse_prefill`.
    """
    out_fp8, out_sf, lse = (
        torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
            q,
            k_cache,
            indices,
            sm_scale,
            512,
            attn_sink,
            topk_length,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
            False,
            0.0,
            token_positions,
            False,
            _FUSED_ROPE_DIM,
            cos_sin_cache,
            n_wv_group,
            _FUSED_QUANT_GROUP,
            True,
            True,
            True,
        )
    )
    return out_fp8, out_sf, lse
```

- [x] **Step 4: Run the smoke test**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_fused_sparse.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add vllm/v1/attention/ops/flashmla.py tests/kernels/attention/test_flashmla_fused_sparse.py
git commit -m "[Attention] Add wrappers for the FlashMLA fused sparse attention ops"
```

### Task 4: Fused kernel equivalence tests against the existing pipeline

**Files:**

- Modify: `tests/kernels/attention/test_flashmla_fused_sparse.py`

**Interfaces:**

- Consumes: Task 1 helpers, Task 3 wrappers, `vllm.models.deepseek_v4_1.common.ops.quantize_and_insert_k_cache`, `vllm.models.deepseek_v4_1.common.ops.fused_inv_rope_fp8_quant`, `vllm.utils.deep_gemm.fp8_einsum`, `vllm.model_executor.layers.quantization.utils.fp8_utils.deepgemm_post_process_fp8_weight_block`.
- Produces (test helpers reused by later tasks): `make_cos_sin_cache(max_pos, device)`, `rope_gptj(x, positions, cos_sin)`, `build_v4_cache(k [T,512] bf16, block_size) -> [num_blocks, block_size, 1, 584] uint8`, `dequant_fused_output(out_fp8, out_sf) -> fp32 [s_q, G, 4096]`, `make_wo_a(n_groups, device) -> (w_perm3d, sf_perm, w_std3d, sf_std)`, `_random_indices(s_q, topk, num_slots, device, min_len=1)`.

- [x] **Step 1: Add the helpers and the decode equivalence test**

```python
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
)
from vllm.models.deepseek_v4_1.common.ops import (
    fused_inv_rope_fp8_quant,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    inverse_permutation,
    o_fused_permutation,
    permute_q_to_fused,
    permute_wo_a_,
)
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.math_utils import round_up

HEAD_DIM, ROPE_DIM, NOPE_DIM = 512, 64, 448
V4_BYTES, V4_ROW_ALIGN = 584, 576


def make_cos_sin_cache(max_pos: int, device) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
                  / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device),
                        inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def rope_gptj(x: torch.Tensor, positions: torch.Tensor,
              cos_sin: torch.Tensor) -> torch.Tensor:
    """GPT-J (interleaved pairs) RoPE on the last 64 dims of [N, ..., 512]."""
    cs = cos_sin[positions].float()
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [ROPE_DIM // 2]
    cos, sin = cs[:, : ROPE_DIM // 2].view(shape), cs[:, ROPE_DIM // 2:].view(shape)
    r = x[..., NOPE_DIM:].float().unflatten(-1, (ROPE_DIM // 2, 2))
    x0, x1 = r[..., 0], r[..., 1]
    rot = torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).flatten(-2)
    return torch.cat([x[..., :NOPE_DIM].float(), rot], -1).to(x.dtype)


def build_v4_cache(k: torch.Tensor, block_size: int) -> torch.Tensor:
    num_tokens = k.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    block_bytes = round_up(block_size * V4_BYTES, V4_ROW_ALIGN)
    cache = torch.zeros(num_blocks, block_bytes, dtype=torch.uint8, device=k.device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=k.device)
    quantize_and_insert_k_cache(k, cache, slots, block_size=block_size)
    return cache[:, : block_size * V4_BYTES].unflatten(1, (block_size, 1, V4_BYTES))


def dequant_fused_output(out_fp8: torch.Tensor, out_sf: torch.Tensor) -> torch.Tensor:
    """[s_q, G, 4096] fp32 from e4m3 values and packed ue8m0 per-32 scales."""
    sf_bytes = out_sf.contiguous().view(torch.uint8).view(*out_sf.shape[:2], 128)
    scale = torch.exp2(sf_bytes.float() - 127.0)
    vals = out_fp8.float().unflatten(-1, (128, 32))
    return (vals * scale.unsqueeze(-1)).flatten(-2)


def _quant_rows_per32(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = w.float().unflatten(-1, (-1, 32))
    exp = torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448.0))
    q = (grouped / torch.exp2(exp).unsqueeze(-1)).clamp(-448, 448)
    return q.flatten(-2).to(torch.float8_e4m3fn), (exp + 127).to(torch.uint8)


def make_wo_a(n_groups: int, device):
    """Random MXFP8 wo_a: (fused-permuted, standard) DeepGEMM (weight, sf) pairs."""
    w = torch.randn(n_groups * 1024, 8 * HEAD_DIM, device=device) * 0.02
    q, s = _quant_rows_per32(w)
    q_perm, s_perm = q.clone(), s.clone()
    permute_wo_a_(q_perm, s_perm, heads_per_group=8)
    std = deepgemm_post_process_fp8_weight_block(
        wq=q, ws=s, quant_block_shape=(1, 32), use_e8m0=False, is_bmm=True,
        bmm_batch_size=n_groups)
    perm = deepgemm_post_process_fp8_weight_block(
        wq=q_perm, ws=s_perm, quant_block_shape=(1, 32), use_e8m0=False,
        is_bmm=True, bmm_batch_size=n_groups)
    return perm[0], perm[1], std[0], std[1]


def _random_indices(s_q, topk, num_slots, device, min_len=1):
    lens = torch.randint(min_len, topk + 1, (s_q,), device=device, dtype=torch.int32)
    idx = torch.randint(0, num_slots, (s_q, topk), device=device, dtype=torch.int32)
    idx[torch.arange(topk, device=device).view(1, -1) >= lens.view(-1, 1)] = -1
    return idx, lens


@pytest.mark.parametrize("s_q", [1, 7, 64, 300])
@pytest.mark.parametrize("with_extra", [False, True])
def test_fused_decode_matches_split_kv_pipeline(s_q: int, with_extra: bool):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    h_q, n_groups, block_size = 64, 8, 32
    topk_swa, topk_extra = 128, 512
    scale = HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(8192, device)
    positions = torch.randint(0, 8192, (s_q,), device=device)
    q_std = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    sink = torch.randn(h_q, device=device) - 2.0
    sink[h_q // 2:] = -float("inf")

    k_swa = torch.randn(2048, HEAD_DIM, device=device, dtype=torch.bfloat16)
    swa_cache = build_v4_cache(k_swa, block_size)
    swa_idx, swa_len = _random_indices(s_q, topk_swa, k_swa.shape[0], device)
    extra_cache = extra_idx = extra_len = None
    if with_extra:
        k_extra = torch.randn(4096, HEAD_DIM, device=device, dtype=torch.bfloat16)
        extra_cache = build_v4_cache(k_extra, 128)
        extra_idx, extra_len = _random_indices(s_q, topk_extra, k_extra.shape[0],
                                               device, min_len=0)

    out_ref, lse_ref = fm.flash_mla_with_kvcache(
        q=rope_gptj(q_std, positions, cos_sin).unsqueeze(1),
        k_cache=swa_cache, block_table=None, head_dim_v=HEAD_DIM,
        tile_scheduler_metadata=fm.FlashMLASchedMeta(), cache_seqlens=None,
        is_fp8_kvcache=True, indices=swa_idx.view(s_q, 1, topk_swa),
        topk_length=swa_len, softmax_scale=scale, attn_sink=sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=None if extra_idx is None
        else extra_idx.view(s_q, 1, topk_extra),
        extra_topk_length=extra_len)
    o_ref_fp8, o_ref_sf = fused_inv_rope_fp8_quant(
        out_ref.squeeze(1), positions, cos_sin, n_groups=n_groups,
        heads_per_group=8, quant_group_size=32, tma_aligned_scales=True)

    out_fp8, out_sf, lse = fm.flash_mla_fused_sparse_decode(
        permute_q_to_fused(q_std), swa_cache, swa_idx, scale,
        positions.to(torch.int32), cos_sin, n_groups, attn_sink=sink,
        topk_length=swa_len, extra_k_cache=extra_cache, extra_indices=extra_idx,
        extra_topk_length=extra_len)

    torch.testing.assert_close(lse, lse_ref.view(s_q, h_q), rtol=1e-3, atol=1e-3)
    inv = inverse_permutation(o_fused_permutation(8, HEAD_DIM)).to(device)
    deq = dequant_fused_output(out_fp8, out_sf)[..., inv]
    deq_ref = dequant_fused_output(o_ref_fp8, o_ref_sf)
    rel = (deq - deq_ref).abs().mean() / deq_ref.abs().mean()
    assert rel < 2e-2, rel

    w_perm, sf_perm, w_std, sf_std = make_wo_a(n_groups, device)
    z = torch.empty(s_q, n_groups, 1024, device=device, dtype=torch.bfloat16)
    z_ref = torch.empty_like(z)
    fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (w_perm, sf_perm), z,
               recipe=(1, 1, 32))
    fp8_einsum("bhr,hdr->bhd", (o_ref_fp8, o_ref_sf), (w_std, sf_std), z_ref,
               recipe=(1, 1, 32))
    torch.testing.assert_close(z, z_ref, rtol=2e-2,
                               atol=2e-2 * z_ref.abs().max().item())


def test_fused_output_sliced_groups_feed_einsum():
    """TP>1 keeps only the first local groups; DeepGEMM must accept the slice."""
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    s_q, n_groups, keep = 33, 8, 2
    cos_sin = make_cos_sin_cache(64, device)
    positions = torch.zeros(s_q, dtype=torch.int32, device=device)
    q = torch.randn(s_q, 64, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cache = build_v4_cache(torch.randn(256, HEAD_DIM, device=device,
                                       dtype=torch.bfloat16), 32)
    idx, lens = _random_indices(s_q, 128, 256, device)
    out_fp8, out_sf, _ = fm.flash_mla_fused_sparse_decode(
        q, cache, idx, HEAD_DIM**-0.5, positions, cos_sin, n_groups,
        topk_length=lens)
    w_perm, sf_perm, _, _ = make_wo_a(n_groups, device)
    z_full = torch.empty(s_q, n_groups, 1024, device=device, dtype=torch.bfloat16)
    fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (w_perm, sf_perm), z_full,
               recipe=(1, 1, 32))
    z_slice = torch.empty(s_q, keep, 1024, device=device, dtype=torch.bfloat16)
    fp8_einsum("bhr,hdr->bhd", (out_fp8[:, :keep], out_sf[:, :keep]),
               (w_perm[:keep], sf_perm[:keep]), z_slice, recipe=(1, 1, 32))
    torch.testing.assert_close(z_slice, z_full[:, :keep], rtol=0, atol=0)
```

- [x] **Step 2: Run and fix until green**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_fused_sparse.py -v -x`
Expected: PASS. If `lse` mismatches only on rows whose every index is invalid, those rows are `+inf` in the fused kernel; all rows here have `min_len=1` so that should not happen. If the sliced-groups test fails inside DeepGEMM's scale-layout check, spec assumption 1.4 is wrong: make the fused class copy `out_fp8[:, :G].contiguous()` and rebuild the SF slice with `get_mn_major_tma_aligned_packed_ue8m0_tensor`, and record that in the spec.

- [x] **Step 3: Add the prefill equivalence test**

```python
@pytest.mark.parametrize("s_q", [1, 184, 2123])
def test_fused_prefill_matches_sparse_fwd_pipeline(s_q: int):
    _skip_unless_supported()
    torch.manual_seed(0)
    device = torch.device("cuda")
    h_q, n_groups, topk, s_kv = 64, 8, 640, 4096
    scale = HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(8192, device)
    positions = torch.randint(0, 8192, (s_q,), device=device)
    q_std = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(s_kv, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    idx, lens = _random_indices(s_q, topk, s_kv, device)
    sink = torch.randn(h_q, device=device) - 2.0

    out_ref, max_logits_ref, lse_ref = fm.flash_mla_sparse_fwd(
        rope_gptj(q_std, positions, cos_sin), kv, idx.view(s_q, 1, topk), scale,
        attn_sink=sink, topk_length=lens)
    o_ref_fp8, o_ref_sf = fused_inv_rope_fp8_quant(
        out_ref, positions, cos_sin, n_groups=n_groups, heads_per_group=8,
        quant_group_size=32, tma_aligned_scales=True)
    out_fp8, out_sf, max_logits, lse = fm.flash_mla_fused_sparse_prefill(
        permute_q_to_fused(q_std), kv, idx.view(s_q, 1, topk), scale,
        positions.to(torch.int32), cos_sin, n_groups, attn_sink=sink,
        topk_length=lens)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(max_logits, max_logits_ref, rtol=1e-3, atol=1e-3)
    w_perm, sf_perm, w_std, sf_std = make_wo_a(n_groups, device)
    z = torch.empty(s_q, n_groups, 1024, device=device, dtype=torch.bfloat16)
    z_ref = torch.empty_like(z)
    fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (w_perm, sf_perm), z,
               recipe=(1, 1, 32))
    fp8_einsum("bhr,hdr->bhd", (o_ref_fp8, o_ref_sf), (w_std, sf_std), z_ref,
               recipe=(1, 1, 32))
    torch.testing.assert_close(z, z_ref, rtol=2e-2,
                               atol=2e-2 * z_ref.abs().max().item())
```

- [x] **Step 4: Run the whole file**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_fused_sparse.py -v`
Expected: all pass (`flash_mla_sparse_fwd` returns `(out, max_logits, lse)`; the vendored signature is `flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None)`).

- [x] **Step 5: Commit**

```bash
git add tests/kernels/attention/test_flashmla_fused_sparse.py
git commit -m "[Test] Check the FlashMLA fused sparse kernel against the split-KV pipeline"
```

### Task 5: Spike benchmark (fused vs split-KV) and the decode threshold

**Files:**

- Create: `benchmarks/kernels/benchmark_dsv41_fused_attention.py`

**Interfaces:**

- Consumes: Task 3 wrappers. Benchmarks must not import tests, so the small cache/RoPE helpers are repeated here.
- Produces: a table `s_q, fused_us, splitkv_attn_us, splitkv_total_us` for decode and `s_q, fused_us, sparse_fwd_total_us` for prefill, eagerly and under CUDA-graph capture. The smallest `s_q` where `fused_us <= splitkv_total_us` becomes the default of `AttentionConfig.dsv4_fused_decode_min_tokens` (Task 7).

- [x] **Step 1: Write the benchmark**

```python
# benchmarks/kernels/benchmark_dsv41_fused_attention.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashMLA fused sparse attention vs the split-KV decode pipeline (DSv4.1).

Decode: topk_swa=128 (V4 584 B cache) + topk_extra=512, h_q=64. The unfused
pipeline is Q RoPE (torch) + flash_mla_with_kvcache + fused_inv_rope_fp8_quant.
Run: .venv/bin/python benchmarks/kernels/benchmark_dsv41_fused_attention.py
"""

import argparse

import torch

import vllm.v1.attention.ops.flashmla as fm
from vllm.models.deepseek_v4_1.common.ops import (
    fused_inv_rope_fp8_quant,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import permute_q_to_fused
from vllm.utils.math_utils import round_up

HEAD_DIM, ROPE_DIM, NOPE_DIM = 512, 64, 448


def make_cos_sin_cache(max_pos, device):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, device=device).float() / ROPE_DIM)
    )
    freqs = torch.outer(torch.arange(max_pos, device=device).float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], -1)


def rope_gptj(x, positions, cos_sin):
    cs = cos_sin[positions].float()
    cos, sin = cs[:, None, :32], cs[:, None, 32:]
    r = x[..., NOPE_DIM:].float().unflatten(-1, (32, 2))
    x0, x1 = r[..., 0], r[..., 1]
    rot = torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1).flatten(-2)
    return torch.cat([x[..., :NOPE_DIM].float(), rot], -1).to(x.dtype)


def build_v4_cache(k, block_size):
    n = k.shape[0]
    num_blocks = (n + block_size - 1) // block_size + 1
    cache = torch.zeros(
        num_blocks, round_up(block_size * 584, 576), dtype=torch.uint8, device=k.device
    )
    quantize_and_insert_k_cache(
        k, cache, torch.arange(n, device=k.device), block_size=block_size
    )
    return cache[:, : block_size * 584].unflatten(1, (block_size, 1, 584))


def random_indices(s_q, topk, num_slots, device):
    idx = torch.randint(0, num_slots, (s_q, topk), device=device, dtype=torch.int32)
    lens = torch.full((s_q,), topk, device=device, dtype=torch.int32)
    return idx, lens


def time_fn(fn, iters=50, warmup=10, graph=False):
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        fn = g.replay
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def bench_decode(s_q, graph, device):
    h_q, groups, scale = 64, 8, HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(65536, device)
    positions = torch.randint(0, 65536, (s_q,), device=device)
    q = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_fused = permute_q_to_fused(q)
    n_swa = s_q * 128 + 128
    swa = build_v4_cache(
        torch.randn(n_swa, HEAD_DIM, device=device, dtype=torch.bfloat16), 32
    )
    extra = build_v4_cache(
        torch.randn(65536, HEAD_DIM, device=device, dtype=torch.bfloat16), 128
    )
    swa_idx, swa_len = random_indices(s_q, 128, n_swa, device)
    ex_idx, ex_len = random_indices(s_q, 512, 65536, device)
    sink = torch.zeros(h_q, device=device)
    sched = fm.FlashMLASchedMeta()
    pos32 = positions.to(torch.int32)

    def fused():
        fm.flash_mla_fused_sparse_decode(
            q_fused, swa, swa_idx, scale, pos32, cos_sin, groups, attn_sink=sink,
            topk_length=swa_len, extra_k_cache=extra, extra_indices=ex_idx,
            extra_topk_length=ex_len,
        )

    def splitkv_attn(q_in):
        return fm.flash_mla_with_kvcache(
            q=q_in.unsqueeze(1), k_cache=swa, block_table=None, head_dim_v=HEAD_DIM,
            tile_scheduler_metadata=sched, cache_seqlens=None, is_fp8_kvcache=True,
            indices=swa_idx.view(s_q, 1, -1), topk_length=swa_len,
            softmax_scale=scale, attn_sink=sink, extra_k_cache=extra,
            extra_indices_in_kvcache=ex_idx.view(s_q, 1, -1),
            extra_topk_length=ex_len,
        )[0]

    def splitkv_total():
        o = splitkv_attn(rope_gptj(q, positions, cos_sin))
        fused_inv_rope_fp8_quant(
            o.squeeze(1), positions, cos_sin, n_groups=groups, heads_per_group=8,
            quant_group_size=32, tma_aligned_scales=True,
        )

    splitkv_attn(q)  # plans the tile scheduler once
    return (
        time_fn(fused, graph=graph),
        time_fn(lambda: splitkv_attn(q), graph=graph),
        time_fn(splitkv_total, graph=graph),
    )


def bench_prefill(s_q, device):
    h_q, groups, topk, s_kv, scale = 64, 8, 640, 8192, HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(65536, device)
    positions = torch.randint(0, 65536, (s_q,), device=device)
    q = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)
    q_fused = permute_q_to_fused(q)
    kv = torch.randn(s_kv, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    idx, lens = random_indices(s_q, topk, s_kv, device)
    pos32 = positions.to(torch.int32)

    def fused():
        fm.flash_mla_fused_sparse_prefill(
            q_fused, kv, idx.view(s_q, 1, -1), scale, pos32, cos_sin, groups,
            topk_length=lens,
        )

    def unfused():
        o = fm.flash_mla_sparse_fwd(
            rope_gptj(q, positions, cos_sin), kv, idx.view(s_q, 1, -1), scale,
            topk_length=lens,
        )[0]
        fused_inv_rope_fp8_quant(
            o, positions, cos_sin, n_groups=groups, heads_per_group=8,
            quant_group_size=32, tma_aligned_scales=True,
        )

    return time_fn(fused), time_fn(unfused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decode-s-q", nargs="+", type=int, default=[1, 6, 16, 64, 256, 1024]
    )
    parser.add_argument("--prefill-s-q", nargs="+", type=int, default=[184, 2123])
    parser.add_argument("--cudagraph", action="store_true")
    args = parser.parse_args()
    ok, reason = fm.is_flashmla_fused_sparse_supported()
    if not ok:
        raise SystemExit(reason)
    device = torch.device("cuda")
    torch.manual_seed(0)
    print("decode: s_q  fused_us  splitkv_attn_us  splitkv_total_us")
    for s_q in args.decode_s_q:
        f, a, t = bench_decode(s_q, args.cudagraph, device)
        print(f"{s_q:>11d} {f:9.1f} {a:16.1f} {t:17.1f}")
    print("prefill: s_q  fused_us  sparse_fwd_total_us")
    for s_q in args.prefill_s_q:
        f, u = bench_prefill(s_q, device)
        print(f"{s_q:>12d} {f:9.1f} {u:19.1f}")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run eagerly and under CUDA graphs**

```bash
/home/yongye/sra/.venv/bin/python benchmarks/kernels/benchmark_dsv41_fused_attention.py
/home/yongye/sra/.venv/bin/python benchmarks/kernels/benchmark_dsv41_fused_attention.py --cudagraph
```

Expected: two tables. Record them in the spec (`20260910-v41-megakernel-vllm-integration-plan.md`, new section "Phase 0 results"). Decision rule: `dsv4_fused_decode_min_tokens` default = smallest `s_q` in the table with `fused_us <= splitkv_total_us`; if the fused kernel wins at every `s_q` the default is `0` and Task 12's fallback branch stays but is off by default. If graph capture of the fused decode fails, note it and make Task 12 raise when `cudagraph_mode` is not `NONE` with the fused class.

- [x] **Step 3: Commit**

```bash
git add benchmarks/kernels/benchmark_dsv41_fused_attention.py 20260910-v41-megakernel-vllm-integration-plan.md
git commit -m "[Bench] Compare the FlashMLA fused sparse kernel with the split-KV pipeline"
```

---

## Part 2: layer integration on the existing `fp8_ds_mla` (V4) caches

### Task 6: int32 positions in the SWA metadata

**Files:**

- Modify: `vllm/v1/attention/backends/mla/sparse_swa.py` (dataclass `DeepseekSparseSWAMetadata` around line 161; builder `__init__` around line 465; `build()` return around line 688)

**Interfaces:**

- Produces: `DeepseekSparseSWAMetadata.positions_int32: torch.Tensor | None` (`[num_tokens]` int32, graph-stable buffer slice), filled once per step for all layers.

- [x] **Step 1: Add the field**

In the `DeepseekSparseSWAMetadata` dataclass, after `token_to_req_indices`:

```python
    # int32 copy of the batch positions for the FlashMLA fused kernels.
    positions_int32: torch.Tensor | None = None  # [num_tokens]
```

- [x] **Step 2: Add the buffer and fill it**

In `DeepseekSparseSWAMetadataBuilder.__init__`, next to `self.decode_swa_lens = torch.zeros(...)`:

```python
        self.positions_int32 = torch.zeros(
            max_tokens, dtype=torch.int32, device=self.device
        )
```

In `build()`, before `return DeepseekSparseSWAMetadata(`:

```python
        positions = common_attn_metadata.positions
        positions_int32 = None
        if positions is not None:
            num_position_tokens = positions.shape[0]
            self.positions_int32[:num_position_tokens].copy_(positions)
            positions_int32 = self.positions_int32[:num_position_tokens]
```

and pass `positions_int32=positions_int32,` in the constructor call (after `token_to_req_indices=token_to_req_indices,`).

- [x] **Step 3: Verify**

Run: `pre-commit run --files vllm/v1/attention/backends/mla/sparse_swa.py` and `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_sparse.py -v -k "chunk_planning or adaptive_width"`
Expected: clean; existing tests pass. End-to-end coverage arrives with Task 13.

- [x] **Step 4: Commit**

```bash
git add vllm/v1/attention/backends/mla/sparse_swa.py
git commit -m "[Attention] Expose int32 positions in the DeepSeek sparse SWA metadata"
```

### Task 7: Attention config flags

**Files:**

- Modify: `vllm/config/attention.py` (after `sparse_mla_force_mqa`, ~line 84; validation in `__post_init__`)
- Test: `tests/config/test_dsv4_fused_attention_config.py` (new)

**Interfaces:**

- Produces: `AttentionConfig.dsv4_fused_attention: bool | None = None`, `AttentionConfig.dsv4_fused_decode_min_tokens: int = <Phase 0 result, 0 if unknown>`.

- [x] **Step 1: Write the failing test**

```python
# tests/config/test_dsv4_fused_attention_config.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.attention import AttentionConfig


def test_dsv4_fused_attention_defaults():
    cfg = AttentionConfig()
    assert cfg.dsv4_fused_attention is None
    assert cfg.dsv4_fused_decode_min_tokens >= 0


def test_dsv4_fused_decode_min_tokens_rejects_negative():
    with pytest.raises(ValueError, match="dsv4_fused_decode_min_tokens"):
        AttentionConfig(dsv4_fused_decode_min_tokens=-1)
```

- [x] **Step 2: Run it to verify it fails**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/config/test_dsv4_fused_attention_config.py -v`
Expected: FAIL (`unexpected keyword argument` / attribute missing)

- [x] **Step 3: Add the fields and validation**

After `sparse_mla_force_mqa` in `AttentionConfig`:

```python
    dsv4_fused_attention: bool | None = None
    """DeepSeek V4.1 on FlashMLA/SM100: run Q RoPE, sparse attention, inverse
    RoPE and the FP8 cast of the output in FlashMLA's fused kernel and feed the
    ``wo_a`` einsum directly. None enables it when the kernel is available;
    True raises at startup if it is not."""

    dsv4_fused_decode_min_tokens: int = 0
    """With ``dsv4_fused_attention``, decode steps with fewer query tokens than
    this use the split-KV FlashMLA decode kernel instead (the fused kernel runs
    one query token per CTA and has no split-KV). 0 always uses the fused
    kernel."""
```

Set the default to the Task 5 result once known. In `__post_init__` (next to the existing `use_fp4_indexer_cache` handling):

```python
        if self.dsv4_fused_decode_min_tokens < 0:
            raise ValueError(
                "dsv4_fused_decode_min_tokens must be >= 0, got "
                f"{self.dsv4_fused_decode_min_tokens}"
            )
```

- [x] **Step 4: Run the test**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/config/test_dsv4_fused_attention_config.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add vllm/config/attention.py tests/config/test_dsv4_fused_attention_config.py
git commit -m "[Config] Add DeepSeek V4.1 fused attention flags"
```

### Task 8: Triton Q layout kernel (pad to 64 heads; fused or standard+RoPE)

**Files:**

- Create: `vllm/models/deepseek_v4_1/common/ops/q_layout.py`
- Test: `tests/kernels/test_dsv41_q_layout.py`

**Interfaces:**

- Produces: `dsv41_q_layout(q_fused_local: Tensor[N, H_local, 512] bf16, padded_heads: int, mode: Literal["fused", "standard_rope"], positions: Tensor | None = None, cos_sin_cache: Tensor | None = None) -> Tensor[N, padded_heads, 512]`. `"fused"` scatters the local fused layout into the 64-head fused layout with zero padding heads (no RoPE); `"standard_rope"` un-permutes to `[N, padded_heads, 512]`, applies GPT-J RoPE to dims 448..511 at `positions` (int64 or int32), zero padding heads. When `padded_heads == H_local` and mode is `"fused"` the input is returned unchanged.

- [x] **Step 1: Write the failing tests**

```python
# tests/kernels/test_dsv41_q_layout.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from tests.kernels.attention.test_flashmla_fused_sparse import (
    make_cos_sin_cache,
    rope_gptj,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    permute_q_from_fused,
    permute_q_to_fused,
)
from vllm.models.deepseek_v4_1.common.ops.q_layout import dsv41_q_layout


@pytest.mark.parametrize("num_tokens", [1, 5, 300])
@pytest.mark.parametrize("local_heads", [16, 32, 64])
def test_fused_mode_pads_fused_layout(num_tokens, local_heads):
    device = torch.device("cuda")
    q_std = torch.randn(num_tokens, local_heads, 512, device=device,
                        dtype=torch.bfloat16)
    out = dsv41_q_layout(permute_q_to_fused(q_std), 64, "fused")
    expected = permute_q_to_fused(F.pad(q_std, (0, 0, 0, 64 - local_heads)))
    assert out.shape == (num_tokens, 64, 512)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("num_tokens", [1, 5, 300])
def test_standard_rope_mode_matches_reference(num_tokens):
    device = torch.device("cuda")
    local_heads = 16
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    q_std = torch.randn(num_tokens, local_heads, 512, device=device,
                        dtype=torch.bfloat16)
    out = dsv41_q_layout(permute_q_to_fused(q_std), 64, "standard_rope",
                         positions=positions, cos_sin_cache=cos_sin)
    expected = rope_gptj(F.pad(q_std, (0, 0, 0, 64 - local_heads)), positions,
                         cos_sin)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2,
                               atol=1e-2)
    assert torch.equal(permute_q_from_fused(permute_q_to_fused(q_std)), q_std)


def test_standard_rope_mode_matches_cuda_q_path():
    """Same RoPE convention as the existing fused Q-norm/RoPE/KV-insert op."""
    device = torch.device("cuda")
    num_tokens, local_heads, block_size = 7, 16, 32
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    q_std = torch.randn(num_tokens, local_heads, 512, device=device,
                        dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, 512, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(2, block_size * 584, dtype=torch.uint8, device=device)
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    q_ref = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q_std, kv, k_cache, slot_mapping, positions, cos_sin, 64, 1e-20,
        block_size, False)
    out = dsv41_q_layout(permute_q_to_fused(q_std), 64, "standard_rope",
                         positions=positions, cos_sin_cache=cos_sin)
    torch.testing.assert_close(out.float(), q_ref.float(), rtol=1e-2, atol=1e-2)
```

- [x] **Step 2: Run to verify failure**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_q_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: ... q_layout`

- [x] **Step 3: Write the kernel**

```python
# vllm/models/deepseek_v4_1/common/ops/q_layout.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pad the local-head Q of the fused FlashMLA layout to the kernel's head count.

``wq_b`` is permuted so the Q GEMM writes each token as 32 chunks of
``[h_local, 16]``. The fused kernel wants 32 chunks of ``[padded_heads, 16]``
with zero padding heads (``"fused"``); the split-KV fallback wants the standard
``[padded_heads, 512]`` layout with GPT-J RoPE applied (``"standard_rope"``).
"""

from typing import Literal

import torch

from vllm.triton_utils import tl, triton

_HEAD_DIM = 512
_Q_CHUNK = 16
_NUM_CHUNKS = _HEAD_DIM // _Q_CHUNK
_ROPE_START_CHUNK = 448 // _Q_CHUNK


@triton.jit
def _q_layout_kernel(
    q_in,
    q_out,
    positions,
    cos_sin,
    COS_STRIDE: tl.constexpr,
    N_LOCAL: tl.constexpr,
    N_PADDED: tl.constexpr,
    STANDARD_ROPE: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    offs = tl.arange(0, N_PADDED * 16)
    valid = offs < N_LOCAL * 16
    src = q_in + t * (N_LOCAL * 512) + c * (N_LOCAL * 16)
    x = tl.load(src + offs, mask=valid, other=0.0)
    if not STANDARD_ROPE:
        tl.store(q_out + t * (N_PADDED * 512) + c * (N_PADDED * 16) + offs, x)
    else:
        h = offs // 16
        j = offs % 16
        d = c * 16 + j
        if c >= 28:
            pos = tl.load(positions + t)
            pair = (d - 448) // 2
            cos = tl.load(cos_sin + pos * COS_STRIDE + pair)
            sin = tl.load(cos_sin + pos * COS_STRIDE + 32 + pair)
            partner = tl.load(src + (offs ^ 1), mask=valid, other=0.0)
            xf = x.to(tl.float32)
            pf = partner.to(tl.float32)
            rotated = tl.where(j % 2 == 0, xf * cos - pf * sin, xf * cos + pf * sin)
            x = rotated.to(q_out.dtype.element_ty)
        tl.store(q_out + t * (N_PADDED * 512) + h * 512 + d, x)


def dsv41_q_layout(
    q: torch.Tensor,
    padded_heads: int,
    mode: Literal["fused", "standard_rope"],
    positions: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad fused-layout local Q to ``padded_heads`` (see module docstring).

    Args:
        q: ``[num_tokens, local_heads, 512]`` bf16 in the fused layout, each
            token contiguous.
        padded_heads: 64 or 128; must be >= ``local_heads``.
        mode: ``"fused"`` keeps the fused layout; ``"standard_rope"`` returns
            the standard layout with RoPE on the last 64 dims.
        positions: ``[num_tokens]`` int positions (``"standard_rope"`` only).
        cos_sin_cache: ``[max_pos, 64]`` fp32 (``"standard_rope"`` only).
    """
    num_tokens, local_heads, head_dim = q.shape
    assert head_dim == _HEAD_DIM and q.stride(2) == 1 and q.stride(1) == head_dim
    assert padded_heads >= local_heads and padded_heads % 16 == 0
    standard_rope = mode == "standard_rope"
    if not standard_rope and padded_heads == local_heads:
        return q
    if standard_rope:
        assert positions is not None and cos_sin_cache is not None
        assert cos_sin_cache.dtype == torch.float32 and cos_sin_cache.shape[1] == 64
        assert cos_sin_cache.stride(1) == 1
    out = torch.empty(
        (num_tokens, padded_heads, head_dim), dtype=q.dtype, device=q.device
    )
    if num_tokens == 0:
        return out
    _q_layout_kernel[(num_tokens, _NUM_CHUNKS)](
        q,
        out,
        positions if standard_rope else out,
        cos_sin_cache if standard_rope else out,
        COS_STRIDE=cos_sin_cache.stride(0) if standard_rope else 0,
        N_LOCAL=local_heads,
        N_PADDED=padded_heads,
        STANDARD_ROPE=standard_rope,
        num_warps=4,
    )
    return out
```

- [x] **Step 4: Run the tests**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_q_layout.py -v`
Expected: PASS. If `test_standard_rope_mode_matches_cuda_q_path` fails on the CUDA op call signature, check `csrc/libtorch_stable/ops.h:265` (argument order `q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, eps, cache_block_size, apply_q_norm`).

- [x] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4_1/common/ops/q_layout.py tests/kernels/test_dsv41_q_layout.py
git commit -m "[DSv4.1] Add the fused-layout Q padding kernel"
```

### Task 9: `fused_inv_rope_fp8_quant(permuted_output=True)` for the fallback path

**Files:**

- Modify: `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`
- Test: `tests/kernels/test_dsv41_fused_layout.py` (append a GPU test)

**Interfaces:**

- Produces: `fused_inv_rope_fp8_quant(..., permuted_output: bool = False)`; with `True` (requires `quant_group_size == 32` and `tma_aligned_scales`) the values of group `g` are stored with chunk `(h, c)` at chunk position `c * heads_per_group + h` and the scale byte at the same chunk position, i.e. exactly the megakernel's `out_fp8` / `out_sf` layout. Output shapes unchanged.

- [x] **Step 1: Write the failing test** (append to `tests/kernels/test_dsv41_fused_layout.py`)

```python
@pytest.mark.parametrize("num_tokens", [1, 5, 129])
def test_inv_rope_quant_permuted_output_matches_standard(num_tokens):
    from tests.kernels.attention.test_flashmla_fused_sparse import (
        make_cos_sin_cache,
    )
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    device = torch.device("cuda")
    n_groups = 2
    o = torch.randn(num_tokens, n_groups * 8, 512, device=device,
                    dtype=torch.bfloat16)
    positions = torch.randint(0, 4096, (num_tokens,), device=device)
    cos_sin = make_cos_sin_cache(4096, device)
    kwargs = dict(n_groups=n_groups, heads_per_group=8, quant_group_size=32,
                  tma_aligned_scales=True)
    std_fp8, std_sf = fused_inv_rope_fp8_quant(o, positions, cos_sin, **kwargs)
    perm_fp8, perm_sf = fused_inv_rope_fp8_quant(o, positions, cos_sin,
                                                 permuted_output=True, **kwargs)
    perm = o_fused_permutation(8, 512).to(device)
    assert torch.equal(perm_fp8.view(torch.uint8),
                       std_fp8.view(torch.uint8)[..., perm])
    chunk_perm = o_fused_chunk_permutation(8, 512).to(device)
    std_bytes = std_sf.contiguous().view(torch.uint8).view(num_tokens, n_groups, 128)
    perm_bytes = perm_sf.contiguous().view(torch.uint8).view(num_tokens, n_groups, 128)
    assert torch.equal(perm_bytes, std_bytes[..., chunk_perm])
    assert perm_sf.stride() == std_sf.stride()
```

Add `import pytest` at the top of that test file.

- [x] **Step 2: Run to verify failure**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_fused_layout.py -v -k permuted_output`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'permuted_output'`

- [x] **Step 3: Implement**

In `_fused_inv_rope_fp8_quant_per_head` add the constexpr `PERMUTED_OUTPUT: tl.constexpr` (after `TMA_ALIGNED_SCALES`) and change the two store sites:

```python
    if PERMUTED_OUTPUT:
        # Chunk (h, c) of this group goes to chunk position c * G + h.
        chunk_ids = tl.arange(0, CHUNKS_PER_HEAD) * heads_per_group + head_in_group
        out_offsets = (
            tl.reshape(chunk_ids, (CHUNKS_PER_HEAD, 1)) * QUANT_GROUP_SIZE
            + tl.reshape(tl.arange(0, QUANT_GROUP_SIZE), (1, QUANT_GROUP_SIZE))
        )
        out_offsets = tl.reshape(out_offsets, (HEAD_DIM,))
    else:
        out_offsets = qb_start * QUANT_GROUP_SIZE + offsets
```

Use `out_offsets` instead of `qb_start * QUANT_GROUP_SIZE + offsets` for both the non-quantized store and the fp8 store (`tl.store(out_base_group + out_offsets, ...)` where `out_base_group = out_ptr + g * out_stride_group + pid_token * out_stride_token`). For the scales in permuted mode the kernel receives `scale_ptr` as a `uint8` view and writes one byte per chunk:

```python
    if PERMUTED_OUTPUT:
        ue8m0_bytes = (scales.to(tl.int32, bitcast=True) >> 23) & 0xFF
        byte_addr = (
            scale_ptr
            + g * scale_stride_group
            + pid_token * 4
            + (chunk_ids // 4) * scale_stride_k
            + chunk_ids % 4
        )
        tl.store(byte_addr, ue8m0_bytes.to(tl.uint8))
```

(and the padding-row branch stores `tl.zeros((CHUNKS_PER_HEAD,), tl.uint8)` at the same addresses). In Python, `fused_inv_rope_fp8_quant` gains `permuted_output: bool = False`; when set, assert `quant_group_size == 32 and tma_aligned_scales and quantize`, and pass through the custom op as a new trailing `bool` argument (`_fused_inv_rope_fp8_quant_kernel_impl(..., permuted_output: bool)` and the fake impl). In the impl, when `permuted_output`, launch with `scale_buf.view(torch.uint8)` as `scale_ptr`, `scale_stride_group=scale_buf.stride(0) * 4`, `scale_stride_k=scale_buf.stride(2) * 4`, and `PERMUTED_OUTPUT=True`. `pid_token * 4` in the kernel is the byte stride of the token dimension (`scale_buf.stride(1) == 1` int32).

- [x] **Step 4: Run the tests**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_fused_layout.py tests/kernels/attention/test_flashmla_fused_sparse.py -v`
Expected: PASS (the existing standard-mode callers are untouched).

- [x] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py tests/kernels/test_dsv41_fused_layout.py
git commit -m "[DSv4] Let fused_inv_rope_fp8_quant emit the fused-kernel chunk layout"
```

### Task 10: Weight permutation at load

**Files:**

- Modify: `vllm/models/deepseek_v4_1/attention.py` (base class: `uses_fused_kernel_layouts` ClassVar and a no-op `finalize_loaded_weights`)
- Modify: `vllm/models/deepseek_v4_1/nvidia/model.py` (`process_weights_after_loading` ~line 1106 and the inner model class next to `finalize_mega_moe_weights` ~line 851; `load_weights` ~line 1100)
- Modify: `vllm/models/deepseek_v4_1/nvidia/dspark.py:513-523`
- Modify: `vllm/models/deepseek_v4_1/nvidia/vl_model.py:354-360`
- Test: `tests/kernels/test_dsv41_fused_layout.py` (append)

**Interfaces:**

- Produces: `DeepseekV4Attention.finalize_loaded_weights(loaded_params: set[str]) -> None` (no-op in the base; the fused subclass from Task 12 permutes `wq_b`/`wo_a` when `f"{self.prefix}.wq_b.weight"` / `f"{self.prefix}.wo_a.weight"` are in `loaded_params`); inner model `finalize_attention_weights(loaded_params)`; the outer models' `process_weights_after_loading(loaded_params)`.

- [x] **Step 1: Write the failing test** (append to `tests/kernels/test_dsv41_fused_layout.py`)

```python
def test_fused_attention_finalize_permutes_only_loaded_layers():
    from types import SimpleNamespace

    from vllm.models.deepseek_v4_1.nvidia.flashmla_fused import (
        DeepseekV4FlashMLAFusedAttention,
    )

    def fake_attn(prefix):
        w_q = torch.randn(16 * 512, 64).to(torch.float8_e4m3fn)
        s_q = torch.randint(0, 255, (16 * 512, 2), dtype=torch.uint8)
        w_o = torch.randn(2 * 1024, 8 * 512).to(torch.float8_e4m3fn)
        s_o = torch.randint(0, 255, (2 * 1024, 128), dtype=torch.uint8)
        return SimpleNamespace(
            prefix=prefix, n_local_heads=16, n_local_groups=2,
            wq_b=SimpleNamespace(weight=w_q, weight_scale=s_q),
            wo_a=SimpleNamespace(weight=w_o, weight_scale=s_o))

    attn = fake_attn("model.layers.3.attn")
    before = (attn.wq_b.weight.clone(), attn.wo_a.weight.clone())
    DeepseekV4FlashMLAFusedAttention.finalize_loaded_weights(
        attn, {"model.layers.4.attn.wq_b.weight"})
    assert torch.equal(attn.wq_b.weight.view(torch.uint8),
                       before[0].view(torch.uint8))
    DeepseekV4FlashMLAFusedAttention.finalize_loaded_weights(
        attn, {"model.layers.3.attn.wq_b.weight", "model.layers.3.attn.wo_a.weight"})
    perm = q_fused_permutation(16, 512)
    assert torch.equal(attn.wq_b.weight.view(torch.uint8),
                       before[0].view(torch.uint8)[perm])
    assert torch.equal(attn.wo_a.weight.view(torch.uint8),
                       before[1].view(torch.uint8)[:, o_fused_permutation(8, 512)])
```

This test also needs Task 12's module to import; until then it fails with `ModuleNotFoundError`, which is the expected red state.

- [x] **Step 2: Base-class hook** (`attention.py`, in `DeepseekV4Attention` after `PREFILL_CHUNK_SIZE`)

```python
    # True when wq_b rows / wo_a columns are permuted for the FlashMLA fused
    # kernel (finalize_loaded_weights); the platform subclass sets it.
    uses_fused_kernel_layouts: ClassVar[bool] = False
```

and after `_uses_fp8_ds_mla_layout`:

```python
    def finalize_loaded_weights(self, loaded_params: set[str]) -> None:
        """Post-load hook per attention layer; runs before quant-method packing."""
        return None
```

- [x] **Step 3: Model wiring** (`nvidia/model.py`)

In the inner model class next to `finalize_mega_moe_weights`:

```python
    def finalize_attention_weights(self, loaded_params: set[str]) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            attn = getattr(layer, "attn", None)
            if attn is not None:
                attn.finalize_loaded_weights(loaded_params)
```

In the outer class:

```python
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.process_weights_after_loading(loaded_params)
        return loaded_params

    def process_weights_after_loading(self, loaded_params: set[str]) -> None:
        self.model.finalize_attention_weights(loaded_params)
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()
```

`vl_model.py:354-360`: `process_weights_after_loading(self, loaded_params)` forwards the set to `self.language_model.process_weights_after_loading(loaded_params)`; its `load_weights` passes the set it returns. `dspark.py:513-523`: `self.process_weights_after_loading(loaded_params)` and

```python
    def process_weights_after_loading(self, loaded_params: set[str]) -> None:
        for layer in self.model.layers:
            layer.attn.finalize_loaded_weights(loaded_params)
        self._finalize_moe()
```

Grep for every other caller of these two `process_weights_after_loading` methods (`grep -rn "process_weights_after_loading()" vllm/models/deepseek_v4_1/`) and pass the set.

- [x] **Step 4: Verify**

Run: `pre-commit run --files vllm/models/deepseek_v4_1/attention.py vllm/models/deepseek_v4_1/nvidia/model.py vllm/models/deepseek_v4_1/nvidia/dspark.py vllm/models/deepseek_v4_1/nvidia/vl_model.py`
Expected: clean. The new test goes green after Task 12.

- [x] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4_1/attention.py vllm/models/deepseek_v4_1/nvidia/model.py vllm/models/deepseek_v4_1/nvidia/dspark.py vllm/models/deepseek_v4_1/nvidia/vl_model.py tests/kernels/test_dsv41_fused_layout.py
git commit -m "[DSv4.1] Add a per-layer post-load weight hook for fused-kernel layouts"
```

### Task 11: Base-class forward hooks (behavior-preserving refactor)

**Files:**

- Modify: `vllm/models/deepseek_v4_1/attention.py:541-576` (`forward`), `672-676` (`project_query_and_cache_kv`), new methods next to `_fused_qnorm_rope_kv_insert`

**Interfaces:**

- Produces: `_alloc_attn_out(num_tokens, hidden_states) -> Tensor` (base: `[N, padded_heads, 512]` in `hidden_states.dtype`), `_finish_o_proj(attn_out, positions) -> Tensor` (base: `self._o_proj(attn_out[:, :n_local_heads], positions)`), `_prepare_q_and_insert_kv(q, kv, positions, attn_metadata) -> Tensor` (base: `self._fused_qnorm_rope_kv_insert(...)`). `forward_mqa(q, kv, positions, out)` keeps its signature; `out` is whatever `_alloc_attn_out` returned.

- [x] **Step 1: Refactor `forward`**

```python
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The eager attention region writes into a caller-owned buffer
        # (breakable_cudagraph requires in-place outputs); the platform
        # subclass decides its shape and how it is projected afterwards.
        attn_out = self._alloc_attn_out(hidden_states.shape[0], hidden_states)

        qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(
            hidden_states
        )
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)

        self._prepare_and_attn_fn(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_weights,
            positions,
            attn_out,
        )
        return self._finish_o_proj(attn_out, positions)

    def _alloc_attn_out(
        self, num_tokens: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Attention output buffer: bf16 ``[N, padded_heads, head_dim]``."""
        return torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def _finish_o_proj(
        self, attn_out: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self._o_proj(attn_out[:, : self.n_local_heads, :], positions)

    def _prepare_q_and_insert_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        """Q RoPE/padding and SWA cache insert; returns the Q attention consumes."""
        return self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
```

and in `_prepare_and_attn` replace the body of `project_query_and_cache_kv` with `return self._prepare_q_and_insert_kv(q, kv, positions, attn_metadata)` (keeping the `self._wq_b_proj(...).view(-1, self.n_local_heads, self.head_dim)` line before it). Rename the `o_padded` parameters of `_prepare_and_attn_eager` / `_prepare_and_attn` / `_sparse_indexer_and_attn` to `attn_out` (the eager-break decorator does not care about names).

- [x] **Step 2: Verify**

Run: `pre-commit run --files vllm/models/deepseek_v4_1/attention.py` and `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_sparse.py -v -k "chunk_planning or flashinfer"`.
Expected: clean, tests pass (FlashInfer and ROCm subclasses inherit the defaults unchanged).

- [x] **Step 3: Commit**

```bash
git add vllm/models/deepseek_v4_1/attention.py
git commit -m "[DSv4.1] Factor the attention output buffer and o_proj tail into hooks"
```

### Task 12: `DeepseekV4FlashMLAFusedAttention`

**Files:**

- Create: `vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py`
- Modify: `vllm/models/deepseek_v4_1/nvidia/model.py:114-149` (`_select_dsv4_attn_cls`)
- Test: `tests/kernels/test_dsv41_fused_layout.py::test_fused_attention_finalize_permutes_only_loaded_layers` (from Task 10) plus Task 13's end-to-end run.

**Interfaces:**

- Consumes: Task 3 wrappers, Task 6 `positions_int32`, Task 7 flags, Task 8 `dsv41_q_layout`, Task 9 `permuted_output`, Task 1 `permute_wq_b_` / `permute_wo_a_`, `rope_quant_insert` from `common/ops/fused_compress_quant_cache.py` (KV-only V4 insert with `compress_ratio=1`), `fp8_einsum`.
- Produces: the class below; `_select_dsv4_attn_cls` returns it when `_dsv4_fused_attention_enabled(vllm_config)`.

- [x] **Step 1: Write the class**

```python
# vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 attention on FlashMLA's fused sparse kernel (SM100).

Q RoPE, sparse attention, inverse RoPE and the FP8 cast run in one kernel whose
output feeds the ``wo_a`` einsum; ``wq_b`` rows and ``wo_a`` columns are
permuted at load (``finalize_loaded_weights``) so the GEMMs speak the kernel's
chunk-interleaved layouts. Small decode batches fall back to the split-KV
kernel (``dsv4_fused_decode_min_tokens``).
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4_1.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.models.deepseek_v4_1.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    permute_wo_a_,
    permute_wq_b_,
)
from vllm.models.deepseek_v4_1.common.ops.q_layout import dsv41_q_layout
from vllm.models.deepseek_v4_1.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV4FlashMLAMetadata
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.ops.flashmla import (
    flash_mla_fused_sparse_decode,
    flash_mla_fused_sparse_prefill,
    flash_mla_with_kvcache,
    is_flashmla_fused_sparse_supported,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def dsv4_fused_attention_enabled(vllm_config: VllmConfig) -> bool:
    """Resolve ``attention_config.dsv4_fused_attention`` for this model."""
    flag = vllm_config.attention_config.dsv4_fused_attention
    if flag is False:
        return False
    ok, reason = is_flashmla_fused_sparse_supported()
    hf_config = vllm_config.model_config.hf_config
    heads_per_group = hf_config.num_attention_heads // hf_config.o_groups
    if ok and heads_per_group != WV_GROUP_SIZE:
        ok, reason = False, f"needs {WV_GROUP_SIZE} heads per o_group"
    if ok and vllm_config.cache_config.cache_dtype not in ("auto", "fp8", "fp8_ds_mla"):
        ok, reason = False, "needs the fp8_ds_mla KV cache layout"
    if not ok and flag is True:
        raise ValueError(f"dsv4_fused_attention=True but unsupported: {reason}")
    return ok


class DeepseekV4FlashMLAFusedAttention(DeepseekV4FlashMLAAttention):
    uses_fused_kernel_layouts: ClassVar[bool] = True

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        if self.n_local_heads // self.n_local_groups != WV_GROUP_SIZE:
            raise ValueError("fused attention needs 8 heads per wo_a group")
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                f"fused attention does not support kv-cache dtype {self.kv_cache_dtype}"
            )
        self.n_wv_group = self.padded_heads // WV_GROUP_SIZE
        self.fused_decode_min_tokens = (
            vllm_config.attention_config.dsv4_fused_decode_min_tokens
        )

    # ---- weights -------------------------------------------------------

    def finalize_loaded_weights(self, loaded_params: set[str]) -> None:
        if f"{self.prefix}.wq_b.weight" in loaded_params:
            permute_wq_b_(
                self.wq_b.weight.data, self.wq_b.weight_scale.data, self.n_local_heads
            )
        if f"{self.prefix}.wo_a.weight" in loaded_params:
            permute_wo_a_(
                self.wo_a.weight.data,
                self.wo_a.weight_scale.data,
                self.n_local_heads // self.n_local_groups,
            )

    # ---- forward plumbing ----------------------------------------------

    def _alloc_attn_out(
        self, num_tokens: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Post-``wo_a`` activation ``z``; ``wo_b`` consumes it in the graph."""
        return torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )

    def _finish_o_proj(
        self, attn_out: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return self.wo_b(attn_out.flatten(1))

    def _prepare_q_and_insert_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor:
        # q is [N, n_local_heads, 512] in the fused layout (permuted wq_b).
        if isinstance(attn_metadata, dict):
            swa_metadata = cast(
                "DeepseekSparseSWAMetadata",
                attn_metadata[self.swa_cache_layer.prefix],
            )
            rope_quant_insert(
                kv,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.swa_cache_layer.kv_cache,
                swa_metadata.slot_mapping,
                compress_ratio=1,
            )
        return q

    def _wo_a_einsum(
        self, out_fp8: torch.Tensor, out_sf: torch.Tensor, z: torch.Tensor
    ) -> None:
        groups = self.n_local_groups
        fp8_einsum(
            "bhr,hdr->bhd",
            (out_fp8[:, :groups], out_sf[:, :groups]),
            (self.wo_a.weight, self.wo_a.weight_scale),
            z,
            recipe=self._einsum_recipe,
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            self._reserve_dummy_run_workspace(q)
            dsv41_q_layout(q, self.padded_heads, "fused")
            output.zero_()
            return
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata",
            attn_metadata[self.swa_cache_layer.prefix],
        )
        assert swa_metadata.positions_int32 is not None
        num_decode_tokens = swa_metadata.num_decode_tokens
        if swa_metadata.num_prefills > 0:
            self._forward_prefill_fused(
                q[num_decode_tokens:],
                swa_metadata.positions_int32[num_decode_tokens:],
                flashmla_metadata,
                swa_metadata,
                output[num_decode_tokens:],
            )
        if swa_metadata.num_decodes > 0:
            if num_decode_tokens >= self.fused_decode_min_tokens:
                self._forward_decode_fused(
                    q[:num_decode_tokens],
                    swa_metadata.positions_int32[:num_decode_tokens],
                    flashmla_metadata,
                    swa_metadata,
                    output[:num_decode_tokens],
                )
            else:
                self._forward_decode_split_kv(
                    q[:num_decode_tokens],
                    positions[:num_decode_tokens],
                    flashmla_metadata,
                    swa_metadata,
                    output[:num_decode_tokens],
                )

    def _reserve_dummy_run_workspace(self, q: torch.Tensor) -> None:
        swa_only = self.compress_ratio == 0
        n = 0 if swa_only else -(-self.max_model_len // self.compress_ratio)
        m = n + self.window_size + self.max_num_batched_tokens
        top_k = 0 if swa_only else self.topk_indices_buffer.shape[-1]  # type: ignore[union-attr]
        combined_topk = round_up(top_k + self.window_size + self.max_image_tokens, 128)
        current_workspace_manager().get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, m, q.shape[-1]), torch.bfloat16),
            ((self.max_num_batched_tokens, combined_topk), torch.int32),
            ((self.max_num_batched_tokens,), torch.int32),
        )

    def _decode_indices(
        self,
        flashmla_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """(extra cache, extra indices [n, topk], extra lens) or all None."""
        if self.compress_ratio == 0:
            return None, None, None
        assert flashmla_metadata is not None
        assert swa_metadata.is_valid_token is not None
        assert self.topk_indices_buffer is not None
        num_decode_tokens = swa_metadata.num_decode_tokens
        indices, lens = compute_global_topk_indices_and_lens(
            self.topk_indices_buffer[:num_decode_tokens],
            swa_metadata.token_to_req_indices,
            flashmla_metadata.block_table[: swa_metadata.num_decodes],
            flashmla_metadata.block_size // self.compress_ratio,
            swa_metadata.is_valid_token[:num_decode_tokens],
        )
        return self._compressed_kv_cache().unsqueeze(-2), indices, lens

    def _forward_decode_fused(self, q, positions_int32, flashmla_metadata,
                              swa_metadata, z) -> None:
        extra_cache, extra_idx, extra_len = self._decode_indices(
            flashmla_metadata, swa_metadata
        )
        assert swa_metadata.decode_swa_indices is not None
        out_fp8, out_sf, _ = flash_mla_fused_sparse_decode(
            dsv41_q_layout(q, self.padded_heads, "fused"),
            self.swa_cache_layer.kv_cache.unsqueeze(-2),
            swa_metadata.decode_swa_indices.view(q.shape[0], -1),
            self.scale,
            positions_int32,
            self.rotary_emb.cos_sin_cache,
            self.n_wv_group,
            attn_sink=self.attn_sink,
            topk_length=swa_metadata.decode_swa_lens,
            extra_k_cache=extra_cache,
            extra_indices=extra_idx,
            extra_topk_length=extra_len,
        )
        self._wo_a_einsum(out_fp8, out_sf, z)

    def _forward_decode_split_kv(self, q, positions, flashmla_metadata,
                                 swa_metadata, z) -> None:
        extra_cache, extra_idx, extra_len = self._decode_indices(
            flashmla_metadata, swa_metadata
        )
        num_tokens = q.shape[0]
        q_std = dsv41_q_layout(
            q, self.padded_heads, "standard_rope",
            positions=positions, cos_sin_cache=self.rotary_emb.cos_sin_cache,
        )
        tile_metadata = {
            0: swa_metadata.tile_sched_swaonly,
            1: swa_metadata.tile_sched_c1a,
            2: swa_metadata.tile_sched_c2a,
        }[self.compress_ratio]
        assert tile_metadata is not None
        out, _ = flash_mla_with_kvcache(
            q=q_std.unsqueeze(1),
            k_cache=self.swa_cache_layer.kv_cache.unsqueeze(-2),
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_metadata.decode_swa_indices,
            topk_length=swa_metadata.decode_swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=None if extra_idx is None
            else extra_idx.view(num_tokens, 1, -1),
            extra_topk_length=extra_len,
        )
        o_fp8, o_sf = fused_inv_rope_fp8_quant(
            out.squeeze(1)[:, : self.n_local_heads],
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=WV_GROUP_SIZE,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            quant_group_size=32,
            tma_aligned_scales=True,
            permuted_output=True,
        )
        self._wo_a_einsum(o_fp8, o_sf, z)

    def _forward_prefill_fused(self, q, positions_int32, flashmla_metadata,
                               swa_metadata, z) -> None:
        swa_only = self.compress_ratio == 0
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert seq_lens is not None and gather_lens is not None
        assert query_start_loc_cpu is not None and query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:][
            : swa_metadata.num_prefill_tokens
        ]
        top_k = 0 if swa_only else topk_indices.shape[-1]
        q_pad = dsv41_q_layout(q, self.padded_heads, "fused")
        chunk_plan = swa_metadata.get_prefill_chunk_plan(
            compress_ratio=self.compress_ratio,
            prefill_chunk_size=self.PREFILL_CHUNK_SIZE,
            has_compressed=not swa_only,
        )
        workspace_manager = current_workspace_manager()
        combined_topk = round_up(top_k + self.window_size + self.max_image_tokens, 128)
        for chunk_start, chunk_end, chunk_n, chunk_m in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv_ws, idx_ws, lens_ws = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_m, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            if not swa_only:
                assert flashmla_metadata is not None
                dequantize_and_gather_k_cache(
                    kv_ws[:chunk_size],
                    self._compressed_kv_cache(),
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=flashmla_metadata.block_table[num_decodes:][
                        chunk_start:chunk_end
                    ],
                    block_size=flashmla_metadata.block_size // self.compress_ratio,
                    offset=0,
                )
            dequantize_and_gather_k_cache(
                kv_ws[:chunk_size],
                self.swa_cache_layer.kv_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_metadata.block_table[num_decodes:][
                    chunk_start:chunk_end
                ],
                block_size=swa_metadata.block_size,
                offset=chunk_n,
            )
            qs = query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            qe = query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[qs:qe],
                query_start_loc[num_decodes + chunk_start : num_decodes + chunk_end + 1],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_m,
                chunk_n,
                out=(idx_ws[: qe - qs], lens_ws[: qe - qs]),
                left_visible=(
                    swa_metadata.prefill_left_visible[
                        num_decode_tokens + qs : num_decode_tokens + qe
                    ]
                    if swa_metadata.prefill_left_visible is not None
                    else None
                ),
                right_visible=(
                    swa_metadata.prefill_right_visible[
                        num_decode_tokens + qs : num_decode_tokens + qe
                    ]
                    if swa_metadata.prefill_right_visible is not None
                    else None
                ),
                max_image_tokens=self.max_image_tokens,
            )
            out_fp8, out_sf, _, _ = flash_mla_fused_sparse_prefill(
                q_pad[qs:qe],
                kv_ws.view(-1, 1, q.shape[-1]),
                combined_indices.unsqueeze(1),
                self.scale,
                positions_int32[qs:qe],
                self.rotary_emb.cos_sin_cache,
                self.n_wv_group,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
            )
            self._wo_a_einsum(out_fp8, out_sf, z[qs:qe])
```

Note `self._einsum_recipe` comes from `DeepseekV4FlashMLAAttention.__init__` (`(1, 1, 32)` on SM100). `combined_indices.unsqueeze(1)` has `stride(0) == combined_topk`, a multiple of 128, so the kernel's 32 B index-row alignment holds.

- [x] **Step 2: Wire selection** (`nvidia/model.py::_select_dsv4_attn_cls`)

Import `DeepseekV4FlashMLAFusedAttention, dsv4_fused_attention_enabled` from `vllm.models.deepseek_v4_1.nvidia.flashmla_fused`. Replace each `return DeepseekV4FlashMLAAttention` with `return _flashmla_attn_cls(vllm_config)` where

```python
def _flashmla_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    if dsv4_fused_attention_enabled(vllm_config):
        logger.info_once("Using the FlashMLA fused sparse attention kernel.")
        return DeepseekV4FlashMLAFusedAttention
    return DeepseekV4FlashMLAAttention
```

- [x] **Step 3: Run the unit tests**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_fused_layout.py tests/kernels/test_dsv41_q_layout.py -v && pre-commit run --files vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py vllm/models/deepseek_v4_1/nvidia/model.py`
Expected: PASS (including Task 10's finalize test), lint clean, `pre-commit run mypy-3.12 --files vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py --hook-stage manual` clean.

- [x] **Step 4: Commit**

```bash
git add vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py vllm/models/deepseek_v4_1/nvidia/model.py
git commit -m "[DSv4.1] Add the FlashMLA fused sparse attention layer"
```

### Task 13: End-to-end parity and latency

**Files:**

- Create: `recipe/dsv41/sra_vigil_tp4_gsm8k_fused.yaml` (copy of `recipe/dsv41/sra_vigil_tp4_gsm8k.yaml` with `--attention-config '{"dsv4_fused_attention": true}'` in the serve command and `PYTHONPATH` pointing at this worktree, as `sra_vigil_tp4_8k1k_32k1k_bs1_dspark_synth.yaml` does)
- Create: `recipe/dsv41/sra_vigil_tp4_8k1k_32k1k_bs1_fused.yaml` (same treatment of `sra_vigil_tp4_8k1k_32k1k_bs1.yaml`)

- [x] **Step 1: Local single-GPU smoke** (worktree root; needs 1 GPU free)

```bash
/home/yongye/sra/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from vllm import LLM, SamplingParams
llm = LLM('/mnt/lustre/sra-weights/ckpt20260903', tensor_parallel_size=4,
          tokenizer_mode='deepseek_v4', max_model_len=8192,
          attention_config={'dsv4_fused_attention': True},
          compilation_config={'cudagraph_mode': 'FULL_AND_PIECEWISE'})
out = llm.generate(['The capital of France is'], SamplingParams(max_tokens=16, temperature=0))
print(out[0].outputs[0].text)
"
```

Expected: coherent text (matches the same prompt with `dsv4_fused_attention: False`). Uses all 4 local GB200s; skip to the recipes if they are busy.

- [x] **Step 2: gsm8k parity and bs1 TPOT**

```bash
vigil -c recipe/dsv41/sra_vigil_tp4_gsm8k_fused.yaml
vigil -c recipe/dsv41/sra_vigil_tp4_gsm8k.yaml
vigil -c recipe/dsv41/sra_vigil_tp4_8k1k_32k1k_bs1_fused.yaml
```

Expected: gsm8k within noise of the non-fused run; bs1 TPOT within noise of the 2026-09-10 baseline (no-spec ~6.6 ms) given the Task 5 threshold. Record all three in the spec under "Phase 2 results".

- [x] **Step 3: Commit**

```bash
git add recipe/dsv41/sra_vigil_tp4_gsm8k_fused.yaml recipe/dsv41/sra_vigil_tp4_8k1k_32k1k_bs1_fused.yaml 20260910-v41-megakernel-vllm-integration-plan.md
git commit -m "[Recipe] Add fused-attention gsm8k and bs1 latency recipes"
```

---

## Part 3: `nvfp4_ds_mla` (V4.1 fp8 sliding-window cache + V4.1 fp4 compressed cache)

Requires `dsv4_fused_attention` (the base class's Q+KV CUDA insert op only knows the 584 B layout). Reference quantizers come from FlashMLA `tests/quant.py` (`KVCacheLayout.V41_FP8Sparse`, `KVCacheLayout.V41_FP4`).

### Task 14: KV layout table and spec plumbing

**Files:**

- Modify: `vllm/models/deepseek_v4_1/attention.py:112-147` (`_resolve_dsv4_kv_cache_dtype`), `441-453` (SWA cache construction), `926-950` (`get_kv_cache_spec`)
- Modify: `vllm/v1/attention/backends/mla/sparse_swa.py:72-124` (`DeepseekV4SWACache`)
- Modify: `vllm/models/deepseek_v4_1/sparse_mla.py:79-84,130-133` (`supported_kv_cache_dtypes`, add `supports_combination`)
- Modify: `vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py` (`dsv4_fused_attention_enabled`, `__init__` dtype check)
- Test: `tests/kernels/test_dsv41_kv_layout.py` (new, CPU)

**Interfaces:**

- Produces: `DSv4KVLayout(cache_dtype, torch_dtype, swa_bytes, swa_alignment, compressed_bytes, compressed_alignment)` frozen dataclass and `DSV4_KV_LAYOUTS = {"fp8_ds_mla": DSv4KVLayout("fp8_ds_mla", torch.uint8, 584, 576, 584, 576), "nvfp4_ds_mla": DSv4KVLayout("nvfp4_ds_mla", torch.uint8, 528, 512, 288, 256)}`; `_resolve_dsv4_kv_cache_dtype(...) -> DSv4KVLayout` (plain-row FlashInfer layouts get `swa_bytes=None`, `compressed_bytes=None`, alignment 512); `DeepseekV4Attention.kv_layout`; `DeepseekV4SWACache(..., state_content_bytes: int | None, alignment: int)`.

- [x] **Step 1: Write the failing test**

```python
# tests/kernels/test_dsv41_kv_layout.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.models.deepseek_v4_1.attention import (
    DSV4_KV_LAYOUTS,
    _resolve_dsv4_kv_cache_dtype,
)


def test_layout_table_page_geometry():
    fp8, fp4 = DSV4_KV_LAYOUTS["fp8_ds_mla"], DSV4_KV_LAYOUTS["nvfp4_ds_mla"]
    assert (fp8.swa_bytes, fp8.compressed_bytes) == (584, 584)
    assert (fp4.swa_bytes, fp4.compressed_bytes) == (528, 288)
    # SWA pages (32 tokens) and compressed pages (128 / 64 states) must be
    # multiples of the FlashMLA TMA stride for their format.
    assert (32 * fp4.swa_bytes) % fp4.swa_alignment == 0
    assert (64 * fp4.compressed_bytes) % fp4.compressed_alignment == 0


@pytest.mark.parametrize("dtype", ["auto", "fp8", "fp8_ds_mla"])
def test_resolve_fp8_ds_mla(dtype):
    layout = _resolve_dsv4_kv_cache_dtype(True, dtype, None)
    assert layout is DSV4_KV_LAYOUTS["fp8_ds_mla"]
    assert layout.torch_dtype == torch.uint8


def test_resolve_nvfp4_ds_mla_requires_ds_mla_layout():
    assert (_resolve_dsv4_kv_cache_dtype(True, "nvfp4_ds_mla", None)
            is DSV4_KV_LAYOUTS["nvfp4_ds_mla"])
    with pytest.raises(ValueError, match="nvfp4_ds_mla"):
        _resolve_dsv4_kv_cache_dtype(False, "nvfp4_ds_mla", None)


def test_resolve_plain_rows():
    bf16 = _resolve_dsv4_kv_cache_dtype(False, "auto", None)
    assert bf16.torch_dtype == torch.bfloat16 and bf16.swa_bytes is None
    fp8 = _resolve_dsv4_kv_cache_dtype(False, "fp8", None)
    assert fp8.torch_dtype == torch.float8_e4m3fn and fp8.alignment_for_swa == 512
```

(Use the attribute name `swa_alignment`; drop `alignment_for_swa` in the last line, it should read `fp8.swa_alignment == 512`.)

- [x] **Step 2: Run it to verify it fails**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_layout.py -v`
Expected: FAIL with `ImportError: cannot import name 'DSV4_KV_LAYOUTS'`

- [x] **Step 3: Implement the table and resolver** (`attention.py`)

```python
@dataclass(frozen=True)
class DSv4KVLayout:
    """Per-token KV format of the two DeepSeek V4.1 caches (None: plain rows)."""

    cache_dtype: str
    torch_dtype: torch.dtype
    swa_bytes: int | None
    swa_alignment: int
    compressed_bytes: int | None
    compressed_alignment: int


DSV4_KV_LAYOUTS: dict[str, DSv4KVLayout] = {
    # V4: 448 e4m3 NoPE + 64 bf16 RoPE + 8 B scale row (7 ue8m0 per 64).
    "fp8_ds_mla": DSv4KVLayout("fp8_ds_mla", torch.uint8, 584, 576, 584, 576),
    # SWA: V4.1 fp8, 512 e4m3 (RoPE quantized) + 16 ue8m0 per 32.
    # Compressed: V4.1 fp4, 512 e2m1 + 32 e4m3 scales per 16.
    "nvfp4_ds_mla": DSv4KVLayout("nvfp4_ds_mla", torch.uint8, 528, 512, 288, 256),
}


def _resolve_dsv4_kv_cache_dtype(
    use_fp8_ds_mla_layout: bool,
    kv_cache_dtype: str,
    cache_config: CacheConfig | None,
) -> DSv4KVLayout:
    """Map ``(backend layout family, --kv-cache-dtype)`` to a KV layout."""
    if kv_cache_dtype == "nvfp4_ds_mla":
        if not use_fp8_ds_mla_layout:
            raise ValueError(
                "nvfp4_ds_mla is a FlashMLA-only DeepSeek V4.1 KV cache format."
            )
        return DSV4_KV_LAYOUTS["nvfp4_ds_mla"]
    if use_fp8_ds_mla_layout:
        if kv_cache_dtype == "auto":
            kv_cache_dtype = "fp8"
        if not kv_cache_dtype.startswith("fp8"):
            raise ValueError(
                "DeepseekV4 fp8_ds_mla layout only supports fp8 "
                f"kv-cache, got {kv_cache_dtype}. Please set "
                "`--kv-cache-dtype fp8` or select a backend that supports "
                "bfloat16 KV cache."
            )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return DSV4_KV_LAYOUTS["fp8_ds_mla"]
    if kv_cache_dtype.startswith("fp8"):
        return DSv4KVLayout(kv_cache_dtype, torch.float8_e4m3fn, None, 512, None, 512)
    return DSv4KVLayout(kv_cache_dtype, torch.bfloat16, None, 512, None, 512)
```

In `__init__`: `self.kv_layout = _resolve_dsv4_kv_cache_dtype(...)`, `self.kv_cache_dtype = self.kv_layout.cache_dtype`, `self.kv_cache_torch_dtype = self.kv_layout.torch_dtype`, and `DeepseekV4SWACache(..., state_content_bytes=self.kv_layout.swa_bytes, alignment=self.kv_layout.swa_alignment)`. In `get_kv_cache_spec` use `dtype=self.kv_cache_torch_dtype`, `alignment=self.kv_layout.compressed_alignment`, `state_content_bytes=self.kv_layout.compressed_bytes`; `kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype)` stays. `DeepseekV4SWACache.__init__` takes `state_content_bytes: int | None = None, alignment: int | None = None`; when both are None keep the current `cache_dtype == "fp8_ds_mla"` logic, otherwise use them in `get_kv_cache_spec`. Add `"nvfp4_ds_mla"` to `DeepseekV4SparseMLABackend.supported_kv_cache_dtypes` and

```python
    @classmethod
    def supports_combination(cls, head_size, dtype, kv_cache_dtype, block_size,
                             use_mla, has_sink, use_sparse, use_mm_prefix,
                             device_capability) -> str | None:
        if kv_cache_dtype == "nvfp4_ds_mla" and device_capability.major != 10:
            return "nvfp4_ds_mla needs SM100 (FlashMLA V4.1 fp4 decode)"
        return None
```

In `flashmla_fused.py`, `dsv4_fused_attention_enabled` accepts `"nvfp4_ds_mla"` too, and `__init__` accepts `self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")`. In `attention.py.__init__`, when `self.kv_cache_dtype == "nvfp4_ds_mla" and not self.uses_fused_kernel_layouts` raise `ValueError("nvfp4_ds_mla requires dsv4_fused_attention")`.

- [x] **Step 4: Run the tests**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_layout.py -v && pre-commit run --files vllm/models/deepseek_v4_1/attention.py vllm/v1/attention/backends/mla/sparse_swa.py vllm/models/deepseek_v4_1/sparse_mla.py vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py`
Expected: PASS, clean.

- [x] **Step 5: Commit**

```bash
git add tests/kernels/test_dsv41_kv_layout.py vllm/models/deepseek_v4_1/attention.py vllm/v1/attention/backends/mla/sparse_swa.py vllm/models/deepseek_v4_1/sparse_mla.py vllm/models/deepseek_v4_1/nvidia/flashmla_fused.py
git commit -m "[DSv4.1] Describe the FlashMLA KV cache formats in one layout table"
```

### Task 15: V4.1 fp8 (528 B) insert and reference quantizer

**Files:**

- Modify: `vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py` (`rope_quant_insert` dispatch + new kernel)
- Create: `tests/kernels/dsv41_kv_reference.py` (torch ports of FlashMLA `tests/quant.py` for V4.1 fp8 and fp4)
- Test: `tests/kernels/test_dsv41_kv_formats.py` (new)

**Interfaces:**

- Produces: `rope_quant_insert(...)` accepts `kv_cache.shape[-1] == 528` (row `page + slot*512`, scale row `page + block*512 + slot*16`); reference `quantize_v41_fp8(k_roped [T,512] bf16) -> (values uint8 [T,512], scales uint8 [T,16])` with `scale = 2**ceil(log2(clamp_min(amax/448, 1e-4)))` per 32 and `values = (k / scale).to(e4m3)`; `dequantize_v41_fp8(values, scales) -> bf16 [T,512]`.

- [x] **Step 1: Write the reference and the failing test**

```python
# tests/kernels/dsv41_kv_reference.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch ports of FlashMLA tests/quant.py for the DeepSeek V4.1 KV formats."""

import torch

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_v41_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tiles = k.float().view(-1, 16, 32)
    scale_inv = (tiles.abs().amax(-1) / 448.0).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(scale_inv)))
    values = (tiles / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    e8m0 = (torch.log2(scale) + 127).round().to(torch.uint8)
    return values.view(-1, 512).view(torch.uint8), e8m0


def dequantize_v41_fp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(scales.float() - 127.0)
    v = values.view(torch.float8_e4m3fn).float().view(-1, 16, 32)
    return (v * scale.unsqueeze(-1)).view(-1, 512).to(torch.bfloat16)


def quantize_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even e2m1 codes (0..15) with saturation at 6."""
    mag = x.abs().clamp(max=6.0)
    values = _E2M1_VALUES.to(x.device)
    idx = torch.bucketize(mag, values, right=False)  # values[idx-1] < mag <= values[idx]
    idx = idx.clamp(max=7)
    lo, hi = values[(idx - 1).clamp(min=0)], values[idx]
    pick_hi = (mag - lo) > (hi - mag)
    tie = (mag - lo) == (hi - mag)
    code = torch.where(pick_hi | (tie & (idx % 2 == 0)), idx, (idx - 1).clamp(min=0))
    code = torch.where(mag == 0, torch.zeros_like(code), code)
    return (code + 8 * (x < 0).to(code.dtype)).to(torch.uint8)


def quantize_v41_fp4(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tiles = k.float().view(-1, 32, 16)
    amax = tiles.abs().amax(-1)
    scale = (amax / 6.0).clamp(2.0**-9, 448.0).to(torch.float8_e4m3fn)
    codes = quantize_e2m1_codes(tiles / scale.float().unsqueeze(-1)).view(-1, 256, 2)
    packed = codes[..., 0] | (codes[..., 1] << 4)
    return packed.to(torch.uint8), scale.view(torch.uint8)


def dequantize_v41_fp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], -1).view(-1, 512).long()
    mag = _E2M1_VALUES.to(packed.device)[codes & 7]
    vals = torch.where(codes >= 8, -mag, mag).view(-1, 32, 16)
    scale = scales.view(torch.float8_e4m3fn).float()
    return (vals * scale.unsqueeze(-1)).view(-1, 512).to(torch.bfloat16)
```

```python
# tests/kernels/test_dsv41_kv_formats.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.attention.test_flashmla_fused_sparse import (
    make_cos_sin_cache,
    rope_gptj,
)
from tests.kernels.dsv41_kv_reference import (
    dequantize_v41_fp4,
    dequantize_v41_fp8,
    quantize_v41_fp4,
    quantize_v41_fp8,
)
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.utils.math_utils import round_up


def _paged(num_tokens, block_size, bytes_per_token, align, device):
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    page = round_up(block_size * bytes_per_token, align)
    cache = torch.zeros(num_blocks, page, dtype=torch.uint8, device=device)
    return cache[:, : block_size * bytes_per_token].view(num_blocks, block_size,
                                                          bytes_per_token)


def _split_v41_fp8(cache, block_size):
    flat = cache.reshape(cache.shape[0], -1)
    values = flat[:, : block_size * 512].reshape(-1, 512)
    scales = flat[:, block_size * 512:].reshape(-1, 16)
    return values, scales


@pytest.mark.parametrize("num_tokens", [1, 17, 300])
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_v41_fp8_insert_matches_reference(num_tokens, compress_ratio):
    device = torch.device("cuda")
    block_size = 32
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.arange(num_tokens, device=device) + 5
    latent = torch.randn(num_tokens, 512, device=device, dtype=torch.bfloat16)
    cache = _paged(num_tokens, block_size, 528, 512, device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    rope_quant_insert(latent, positions, cos_sin, cache, slots, compress_ratio)
    values, scales = _split_v41_fp8(cache, block_size)
    written = (positions + 1) % compress_ratio == 0
    k_pos = positions // compress_ratio * compress_ratio
    ref_vals, ref_scales = quantize_v41_fp8(rope_gptj(latent, k_pos, cos_sin))
    assert torch.equal(values[:num_tokens][written], ref_vals[written])
    assert torch.equal(scales[:num_tokens][written], ref_scales[written])
    deq = dequantize_v41_fp8(values[:num_tokens][written], scales[:num_tokens][written])
    ref = rope_gptj(latent, k_pos, cos_sin)[written].float()
    assert ((deq.float() - ref).abs() <= ref.abs() * 0.07 + 1e-3).all()
```

- [x] **Step 2: Run to verify failure**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_formats.py -v`
Expected: FAIL on the `assert kv_cache.shape[-1] == 584` inside `rope_quant_insert`.

- [x] **Step 3: Add the 528 B branch**

In `rope_quant_insert`, replace `if kv_cache.dtype == torch.uint8:` body's `assert kv_cache.shape[-1] == 584` with a dispatch on `kv_cache.shape[-1]` (`584` -> existing `_rope_quant_insert_kernel`, `528` -> `_rope_quant_insert_v41_kernel`, else `ValueError`). New kernel:

```python
@triton.jit
def _rope_quant_insert_v41_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    row = tl.interleave(even * c - odd * s, odd * c + even * s)
    # RoPE is applied in fp32 then rounded to bf16 like the bf16 caches.
    row = row.to(tl.bfloat16).to(tl.float32)
    tiles = tl.reshape(row, (16, 32))
    scale_inv = tl.maximum(tl.max(tl.abs(tiles), 1) * (1.0 / 448.0), 1e-4)
    exponent = tl.ceil(tl.log2(scale_inv))
    scaled = tiles * tl.reshape(tl.exp2(-exponent), (16, 1))
    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    tl.store(page + (slot % CACHE_BLOCK) * 512 + d,
             tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,)))
    scales = page + CACHE_BLOCK * 512 + (slot % CACHE_BLOCK) * 16
    tl.store(scales + tl.arange(0, 16), (exponent + 127.0).to(tl.uint8))
```

Launch it with the same arguments as `_rope_quant_insert_kernel` minus `SANITIZE_CACHE_NANS`. Update the `rope_quant_insert` docstring to name the 528 B format. If the exact-match assertion fails only on tiles whose amax lands on an fp8 rounding boundary, compare the reference with `row` rounded through bf16 the same way (the reference already receives bf16 `rope_gptj` output, so both sides round identically).

- [x] **Step 4: Run**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_formats.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py tests/kernels/dsv41_kv_reference.py tests/kernels/test_dsv41_kv_formats.py
git commit -m "[DSv4.1] Insert RoPE'd KV rows into the V4.1 fp8 (528 B) paged layout"
```

### Task 16: V4.1 fp4 (288 B) compressed insert

**Files:**

- Modify: `vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py`
- Test: `tests/kernels/test_dsv41_kv_formats.py` (append)

**Interfaces:**

- Produces: `rope_quant_insert(...)` accepts `kv_cache.shape[-1] == 288` (data `page + slot*256`, scales `page + block*256 + slot*32`); reuses `_fp32x2_to_fp4x2` from `vllm.models.deepseek_v4_1.common.ops` (MXFP4 indexer path; even element in the low nibble, round-to-nearest-even via `cvt.rn.satfinite.e2m1x2.f32`).

- [x] **Step 1: Write the failing test** (append)

```python
def _split_v41_fp4(cache, block_size):
    flat = cache.reshape(cache.shape[0], -1)
    packed = flat[:, : block_size * 256].reshape(-1, 256)
    scales = flat[:, block_size * 256:].reshape(-1, 32)
    return packed, scales


@pytest.mark.parametrize("num_tokens", [1, 17, 300])
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_v41_fp4_insert_matches_reference(num_tokens, compress_ratio):
    device = torch.device("cuda")
    block_size = 64
    cos_sin = make_cos_sin_cache(4096, device)
    positions = torch.arange(num_tokens, device=device) + 5
    latent = torch.randn(num_tokens, 512, device=device, dtype=torch.bfloat16)
    cache = _paged(num_tokens, block_size, 288, 256, device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    rope_quant_insert(latent, positions, cos_sin, cache, slots, compress_ratio)
    packed, scales = _split_v41_fp4(cache, block_size)
    written = (positions + 1) % compress_ratio == 0
    k_pos = positions // compress_ratio * compress_ratio
    roped = rope_gptj(latent, k_pos, cos_sin)
    ref_packed, ref_scales = quantize_v41_fp4(roped)
    assert torch.equal(scales[:num_tokens][written], ref_scales[written])
    assert torch.equal(packed[:num_tokens][written], ref_packed[written])
    deq = dequantize_v41_fp4(packed[:num_tokens][written], scales[:num_tokens][written])
    ref = roped[written].float()
    tile_amax = ref.view(-1, 32, 16).abs().amax(-1, keepdim=True).expand(-1, -1, 16)
    assert ((deq.float() - ref).abs() <= tile_amax.reshape(-1, 512) / 6 + 1e-3).all()
```

- [x] **Step 2: Run to verify failure**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_formats.py -v -k fp4`
Expected: FAIL with the `ValueError` from the dispatch added in Task 15.

- [x] **Step 3: Add the 288 B branch**

```python
@triton.jit
def _rope_fp4_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(cache_slots + t)
    if slot < 0:
        return
    position = tl.load(positions + t)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    d = tl.arange(0, 512)
    normed = tl.load(latent + t.to(tl.int64) * 512 + d).to(tl.float32)
    even, odd = tl.split(tl.reshape(normed, (256, 2)))
    pair = tl.arange(0, 256) - 224
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO) * COS_STRIDE
    c = tl.load(cs + tl.maximum(pair, 0), pair >= 0, other=1.0).to(tl.float32)
    s = tl.load(cs + 32 + tl.maximum(pair, 0), pair >= 0, other=0.0).to(tl.float32)
    row = tl.interleave(even * c - odd * s, odd * c + even * s)
    row = row.to(tl.bfloat16).to(tl.float32)
    tiles = tl.reshape(row, (32, 16))
    amax = tl.max(tl.abs(tiles), 1)
    scale = tl.clamp(amax * (1.0 / 6.0), 0.001953125, 448.0).to(tl.float8e4nv)
    scale_f32 = scale.to(tl.float32)
    scaled = tl.reshape(tiles / tl.reshape(scale_f32, (32, 1)), (512,))
    lo, hi = tl.split(tl.reshape(scaled, (256, 2)))
    packed = _fp32x2_to_fp4x2(lo, hi)
    page = cache + (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    tl.store(page + (slot % CACHE_BLOCK) * 256 + tl.arange(0, 256), packed)
    scales_ptr = page + CACHE_BLOCK * 256 + (slot % CACHE_BLOCK) * 32
    tl.store(scales_ptr + tl.arange(0, 32), scale.to(tl.uint8, bitcast=True))
```

Import `_fp32x2_to_fp4x2` from `vllm.models.deepseek_v4_1.common.ops` (it is re-exported there for `indexer_k_store`; check its argument order — the first argument must land in the low nibble; if the indexer helper packs the opposite way, swap `lo, hi`). Dispatch `288 -> _rope_fp4_insert_kernel`.

- [x] **Step 4: Run**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_formats.py -v`
Expected: PASS. If codes differ only on exact ties, the PTX conversion rounds to nearest-even like the reference; a systematic off-by-one means the nibble order is swapped.

- [x] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py tests/kernels/test_dsv41_kv_formats.py
git commit -m "[DSv4.1] Insert compressed KV rows into the V4.1 fp4 (288 B) paged layout"
```

### Task 17: Prefill gather / dequant for 528 B and 288 B pages

**Files:**

- Modify: `vllm/models/deepseek_v4_1/common/ops/cache_utils.py` (`dequantize_and_gather_k_cache` ~line 394 and new Triton kernels next to `_dequantize_and_gather_k_kernel`)
- Test: `tests/kernels/test_dsv41_kv_formats.py` (append)

**Interfaces:**

- Produces: `dequantize_and_gather_k_cache(out, k_cache, seq_lens, gather_lens, block_table, block_size, offset)` dispatches on `k_cache.shape[-1]`: 584 (existing, including the CuteDSL path), 528 (`_dequantize_and_gather_v41_kernel`), 288 (`_dequantize_and_gather_fp4_kernel`). Same grid `(num_reqs, NUM_WORKERS=128)`, same `seq_lens` / `gather_lens` / `offset` semantics as the existing Triton kernel (each worker walks tokens `worker, worker + 128, ...` of its request; with `gather_lens` the last `gather_lens[r]` tokens of the sequence are gathered, otherwise the first `seq_lens[r]`).

- [x] **Step 1: Write the failing test** (append)

```python
from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache


@pytest.mark.parametrize("bytes_per_token", [528, 288])
def test_gather_dequant_new_formats(bytes_per_token):
    device = torch.device("cuda")
    block_size, num_reqs, max_len = 32, 3, 200
    cos_sin = make_cos_sin_cache(4096, device)
    seq_lens = torch.tensor([200, 33, 1], device=device, dtype=torch.int32)
    total = int(seq_lens.sum())
    latent = torch.randn(total, 512, device=device, dtype=torch.bfloat16)
    positions = torch.arange(total, device=device)
    align = 512 if bytes_per_token == 528 else 256
    cache = _paged(total, block_size, bytes_per_token, align, device)
    slots = torch.arange(total, dtype=torch.int64, device=device)
    rope_quant_insert(latent, positions, cos_sin, cache, slots, 1)
    blocks_per_req = (max_len + block_size - 1) // block_size
    block_table = torch.zeros(num_reqs, blocks_per_req, dtype=torch.int32, device=device)
    starts = torch.tensor([0, 200, 233], device=device)
    for r in range(num_reqs):
        n = int(seq_lens[r])
        first = int(starts[r]) // block_size
        block_table[r, : (n + block_size - 1) // block_size] = torch.arange(
            first, first + (n + block_size - 1) // block_size, device=device)
    out = torch.zeros(num_reqs, max_len, 512, device=device, dtype=torch.bfloat16)
    dequantize_and_gather_k_cache(out, cache, seq_lens, None, block_table,
                                  block_size, 0)
    roped = rope_gptj(latent, positions, cos_sin)
    for r in range(num_reqs):
        n = int(seq_lens[r])
        ref = roped[int(starts[r]) : int(starts[r]) + n].float()
        got = out[r, :n].float()
        tol = ref.abs() * 0.07 + 1e-3 if bytes_per_token == 528 else (
            ref.view(-1, 32, 16).abs().amax(-1, keepdim=True).expand(-1, -1, 16)
            .reshape(-1, 512) / 6 + 1e-3)
        assert ((got - ref).abs() <= tol).all()
```

`starts` must be block aligned for this table construction: change `seq_lens` to `[192, 64, 1]` and `starts` to `[0, 192, 256]` if the assertion trips on block boundaries.

- [x] **Step 2: Run to verify failure**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_formats.py -v -k gather`
Expected: FAIL (existing kernel decodes the bytes as 584 B rows).

- [x] **Step 3: Implement the two kernels**

Copy `_dequantize_and_gather_k_kernel`'s program structure (request id from `program_id(0)`, worker from `program_id(1)`, token loop, `physical_block_idx = block_table[req, tok // block_size]`, output at `out + req*out_stride0 + (offset + tok)*out_stride1`). Per token, the V4.1 fp8 body:

```python
        page = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride
        d = tl.arange(0, 512)
        raw = tl.load(page + (tok % cache_block_size) * 512 + d)
        vals = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        e8m0 = tl.load(page + cache_block_size * 512 + (tok % cache_block_size) * 16
                       + tl.arange(0, 16))
        scale = tl.exp2(e8m0.to(tl.float32) - 127.0)
        row = tl.reshape(tl.reshape(vals, (16, 32)) * tl.reshape(scale, (16, 1)), (512,))
        tl.store(out_row_ptr + d, row.to(tl.bfloat16))
```

and the fp4 body:

```python
        b = tl.arange(0, 256)
        packed = tl.load(page + (tok % cache_block_size) * 256 + b)
        lo = packed & 0xF
        hi = packed >> 4
        codes = tl.interleave(lo, hi)  # [512], even element from the low nibble
        mag_code = codes & 7
        exp = mag_code >> 1
        man = (mag_code & 1).to(tl.float32)
        mag = tl.where(exp == 0, man * 0.5, (1.0 + man * 0.5) * tl.exp2(exp.to(tl.float32) - 1.0))
        vals = tl.where(codes >= 8, -mag, mag)
        sf = tl.load(page + cache_block_size * 256 + (tok % cache_block_size) * 32
                     + tl.arange(0, 32))
        scale = sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        row = tl.reshape(tl.reshape(vals, (32, 16)) * tl.reshape(scale, (32, 1)), (512,))
        tl.store(out_row_ptr + tl.arange(0, 512), row.to(tl.bfloat16))
```

`dequantize_and_gather_k_cache` picks the kernel by `k_cache.shape[-1]` and only uses the CuteDSL path for 584.

- [x] **Step 4: Run**

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/test_dsv41_kv_formats.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add vllm/models/deepseek_v4_1/common/ops/cache_utils.py tests/kernels/test_dsv41_kv_formats.py
git commit -m "[DSv4.1] Gather and dequantize V4.1 fp8 and fp4 pages for prefill"
```

### Task 18: Fused decode on V4.1 fp8 + fp4 caches, DSpark context insert

**Files:**

- Modify: `tests/kernels/attention/test_flashmla_fused_sparse.py` (append)
- Modify: `vllm/models/deepseek_v4_1/nvidia/dspark.py:232-300` (`_insert_context_kv`)

**Interfaces:**

- Consumes: Tasks 15-16 insert kernels, Task 3 wrappers.
- Produces: `_insert_context_kv` uses `rope_quant_insert(kv, positions, cos_sin, swa_cache, slot_mapping, 1)` for every `uint8` SWA cache (584 or 528) instead of the CUDA op with a dummy Q.

- [x] **Step 1: Write the failing test** (append to the fused-sparse test file)

```python
def test_fused_decode_v41_fp8_swa_with_fp4_extra_matches_split_kv():
    _skip_unless_supported()
    from tests.kernels.dsv41_kv_reference import (
        dequantize_v41_fp4,
        dequantize_v41_fp8,
    )
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
        rope_quant_insert,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    s_q, h_q, n_groups = 40, 64, 8
    scale = HEAD_DIM**-0.5
    cos_sin = make_cos_sin_cache(8192, device)
    positions = torch.randint(0, 8192, (s_q,), device=device)
    q_std = torch.randn(s_q, h_q, HEAD_DIM, device=device, dtype=torch.bfloat16)

    def paged(n_tokens, block_size, bytes_per_token, align):
        num_blocks = (n_tokens + block_size - 1) // block_size + 1
        page = round_up(block_size * bytes_per_token, align)
        cache = torch.zeros(num_blocks, page, dtype=torch.uint8, device=device)
        return cache[:, : block_size * bytes_per_token].unflatten(
            1, (block_size, 1, bytes_per_token))

    k_swa = torch.randn(2048, HEAD_DIM, device=device, dtype=torch.bfloat16)
    swa = paged(2048, 32, 528, 512)
    rope_quant_insert(k_swa, torch.zeros(2048, dtype=torch.int64, device=device),
                      cos_sin, swa.squeeze(2), torch.arange(2048, device=device), 1)
    k_ex = torch.randn(4096, HEAD_DIM, device=device, dtype=torch.bfloat16)
    extra = paged(4096, 64, 288, 256)
    rope_quant_insert(k_ex, torch.zeros(4096, dtype=torch.int64, device=device),
                      cos_sin, extra.squeeze(2), torch.arange(4096, device=device), 1)
    swa_idx, swa_len = _random_indices(s_q, 128, 2048, device)
    ex_idx, ex_len = _random_indices(s_q, 512, 4096, device, min_len=0)

    out_ref, lse_ref = fm.flash_mla_with_kvcache(
        q=rope_gptj(q_std, positions, cos_sin).unsqueeze(1), k_cache=swa,
        block_table=None, head_dim_v=HEAD_DIM,
        tile_scheduler_metadata=fm.FlashMLASchedMeta(), cache_seqlens=None,
        is_fp8_kvcache=True, indices=swa_idx.view(s_q, 1, -1), topk_length=swa_len,
        softmax_scale=scale, extra_k_cache=extra,
        extra_indices_in_kvcache=ex_idx.view(s_q, 1, -1), extra_topk_length=ex_len)
    out_fp8, out_sf, lse = fm.flash_mla_fused_sparse_decode(
        permute_q_to_fused(q_std), swa, swa_idx, scale, positions.to(torch.int32),
        cos_sin, n_groups, topk_length=swa_len, extra_k_cache=extra,
        extra_indices=ex_idx, extra_topk_length=ex_len)
    torch.testing.assert_close(lse, lse_ref.view(s_q, h_q), rtol=1e-3, atol=1e-3)

    # bf16 torch reference over the dequantized rows (validates the formats).
    swa_rows = dequantize_v41_fp8(*(lambda f: (f[:, : 32 * 512].reshape(-1, 512),
                                               f[:, 32 * 512:].reshape(-1, 16)))(
        swa.reshape(swa.shape[0], -1)))
    ex_rows = dequantize_v41_fp4(*(lambda f: (f[:, : 64 * 256].reshape(-1, 256),
                                              f[:, 64 * 256:].reshape(-1, 32)))(
        extra.reshape(extra.shape[0], -1)))
    q_r = rope_gptj(q_std, positions, cos_sin).float()
    for t in range(s_q):
        rows = torch.cat([swa_rows[swa_idx[t, : swa_len[t]].long()],
                          ex_rows[ex_idx[t, : ex_len[t]].long()]]).float()
        logits = q_r[t] @ rows.T * scale
        ref_lse = torch.logsumexp(logits, -1)
        torch.testing.assert_close(lse[t], ref_lse, rtol=2e-2, atol=2e-2)
```

- [x] **Step 2: Run** (should already pass once Tasks 15-16 are in; it fails before them on the 528 B assert)

Run: `/home/yongye/sra/.venv/bin/python -m pytest tests/kernels/attention/test_flashmla_fused_sparse.py -v -k v41`
Expected: PASS

- [x] **Step 3: DSpark context insert**

Replace the `uint8` branch of `_insert_context_kv` with

```python
    if cache_dtype == torch.uint8:
        rope_quant_insert(kv, positions, cos_sin_cache, swa_cache, slot_mapping, 1)
        return
```

(import `rope_quant_insert` from `vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache`; `dummy_q` is then only needed for the plain-row branches, so build it after the early return). Run `pre-commit run --files vllm/models/deepseek_v4_1/nvidia/dspark.py`.

- [x] **Step 4: Commit**

```bash
git add tests/kernels/attention/test_flashmla_fused_sparse.py vllm/models/deepseek_v4_1/nvidia/dspark.py
git commit -m "[DSv4.1] Cover fused decode on V4.1 fp8/fp4 caches; layout-agnostic DSpark context insert"
```

### Task 19: End-to-end evaluation with `nvfp4_ds_mla`

**Files:**

- Create: `recipe/dsv41/sra_vigil_tp4_gsm8k_nvfp4.yaml`, `recipe/dsv41/sra_vigil_tp4_gpqad_nvfp4.yaml`, `recipe/dsv41/sra_vigil_tp4_8k1k_32k1k_bs1_nvfp4.yaml` (copies of the fused recipes from Task 13 with `--kv-cache-dtype nvfp4_ds_mla`)

- [x] **Step 1: Local smoke** as in Task 13 Step 1 with `kv_cache_dtype="nvfp4_ds_mla"` and `attention_config={"dsv4_fused_attention": True}`; the log must show the SWA spec at 528 B and the compressed spec at 288 B per token (add a `logger.info_once` of the layout in `DeepseekV4Attention.__init__`).

- [ ] **Step 2: Evals and latency**

```bash
vigil -c recipe/dsv41/sra_vigil_tp4_gsm8k_nvfp4.yaml
vigil -c recipe/dsv41/sra_vigil_tp4_gpqad_nvfp4.yaml
vigil -c recipe/dsv41/sra_vigil_tp4_8k1k_32k1k_bs1_nvfp4.yaml
```

Expected: gsm8k/gpqa within noise of `fp8_ds_mla`; record the 32k TPOT and the KV capacity (`# GPU blocks` line in the engine log) against the fp8 run. The long-context accuracy gate is the user's call (spec open question 3); report the numbers without picking the gate.

- [ ] **Step 3: Commit**

```bash
git add recipe/dsv41/sra_vigil_tp4_gsm8k_nvfp4.yaml recipe/dsv41/sra_vigil_tp4_gpqad_nvfp4.yaml recipe/dsv41/sra_vigil_tp4_8k1k_32k1k_bs1_nvfp4.yaml 20260910-v41-megakernel-vllm-integration-plan.md
git commit -m "[Recipe] Evaluate DeepSeek V4.1 with the nvfp4_ds_mla KV cache"
```

---

## Self-review notes

- Spec coverage: 3.1 (Tasks 7, 12), 3.2 (Task 11), 3.3 (Task 8), 3.4 (Tasks 12, 15, 16, 18), 3.5 (Task 12), 3.6 (Task 17), 3.7 (Task 9), 3.8 (Task 14), 3.9 (Task 6), tests T1-T9 (Tasks 1, 4, 8, 9, 15-18), E1/P1/P2 (Task 13), E2 (Task 19), Phase 0 (Task 5), Phase 1 (Tasks 2-3). The fork-side stable-ABI port is owned by the other FlashMLA session and is a dependency of Task 2, not a task here.
- Names used across tasks: `dsv41_q_layout(q, padded_heads, mode, positions=, cos_sin_cache=)`, `permute_wq_b_`, `permute_wo_a_`, `finalize_loaded_weights(loaded_params)`, `positions_int32`, `flash_mla_fused_sparse_{prefill,decode}`, `DSv4KVLayout`, `rope_quant_insert(latent, positions, cos_sin_cache, kv_cache, slot_mapping, compress_ratio)`.
