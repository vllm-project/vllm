# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for canonicalize_singleton_dim_strides.

Background
----------
When num_kv_heads_per_rank == 1 (e.g. Qwen3.5-397B with TP=8 → 1 KV head
per rank), PyTorch's is_contiguous() returns True for *any* stride on the
size-1 dimension.  The KV cache allocator can therefore produce a tensor
where that singleton dim has stride = 1 element (2 bytes for bf16) instead
of the canonical product-of-remaining-dims value.

CUDA TMA (used by FlashInfer XQA SM90 and Flash-Attention 3/4 on H100+)
requires all non-outermost strides to be multiples of 16 bytes.  A 2-byte
stride triggers cudaErrorIllegalInstruction.

canonicalize_singleton_dim_strides() patches degenerate strides on all
size-1 dimensions via torch.as_strided — zero-copy.

The degenerate stride manifests at different positions in different backends:
- FlashInfer: stride(-3) after kv_cache.permute() → shape [..., 1, B, D]
- FlashAttention: stride(-2) after kv_cache.unbind(0) → shape [N, B, 1, D]
"""

import torch

from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inject_degenerate_stride(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Return a view of t with a degenerate (stride=1) on a size-1 dim."""
    assert t.shape[dim] == 1, f"dim {dim} must have size 1"
    strides = list(t.stride())
    strides[dim] = 1  # inject the bug
    return t.as_strided(t.shape, strides)


# ---------------------------------------------------------------------------
# Tests: canonicalize_singleton_dim_strides
# ---------------------------------------------------------------------------


class TestCanonicalizeSingletonDimStrides:
    def test_flashinfer_layout_dim_neg3(self):
        """FlashInfer path: degenerate stride at dim -3 (num_kv_heads)."""
        # Shape after permute: [num_blocks, 2, num_kv_heads, block_size, head_size]
        num_blocks, block_size, head_size = 64, 16, 128
        t = torch.zeros(num_blocks, 2, 1, block_size, head_size, dtype=torch.bfloat16)
        t_deg = _inject_degenerate_stride(t, dim=-3)

        assert t_deg.stride(-3) == 1  # confirm degenerate
        assert t_deg.is_contiguous()  # PyTorch doesn't notice

        fixed = canonicalize_singleton_dim_strides(t_deg)

        assert fixed.stride(-3) == block_size * head_size  # canonical = 2048
        assert fixed.stride(-2) == head_size  # inner dims unchanged
        assert fixed.stride(-1) == 1

    def test_flash_attn_layout_dim_neg2(self):
        """FlashAttention path: degenerate stride at dim -2 (num_kv_heads)."""
        # Shape after unbind(0): [num_blocks, block_size, num_kv_heads, head_size]
        num_blocks, block_size, head_size = 64, 16, 128
        t = torch.zeros(num_blocks, block_size, 1, head_size, dtype=torch.bfloat16)
        t_deg = _inject_degenerate_stride(t, dim=-2)

        assert t_deg.stride(-2) == 1
        assert t_deg.is_contiguous()

        fixed = canonicalize_singleton_dim_strides(t_deg)

        assert fixed.stride(-2) == head_size  # canonical = 128
        assert fixed.stride(-1) == 1

    def test_canonical_strides_returned_as_is(self):
        """No degenerate strides → same object returned (no copy, no new view)."""
        t = torch.zeros(64, 2, 1, 16, 128, dtype=torch.bfloat16)
        result = canonicalize_singleton_dim_strides(t)
        assert result is t

    def test_multi_kv_heads_unchanged(self):
        """num_kv_heads > 1 → strides are already canonical → unchanged."""
        t = torch.zeros(16, 2, 4, 16, 128, dtype=torch.bfloat16)
        original_strides = t.stride()
        result = canonicalize_singleton_dim_strides(t)
        assert result.stride() == original_strides

    def test_data_pointer_preserved(self):
        """Fix is zero-copy: same underlying storage."""
        t = torch.zeros(8, 2, 1, 16, 128, dtype=torch.bfloat16)
        t_deg = _inject_degenerate_stride(t, dim=-3)
        fixed = canonicalize_singleton_dim_strides(t_deg)
        assert fixed.data_ptr() == t_deg.data_ptr()
        assert fixed.storage_offset() == t_deg.storage_offset()

    def test_multiple_singleton_dims(self):
        """All size-1 dims with degenerate strides are fixed."""
        # Shape: [1, 1, 8, 32] — two size-1 dims
        t = torch.zeros(1, 1, 8, 32, dtype=torch.float16)
        # Both size-1 dims get degenerate strides
        t_deg = t.as_strided(t.shape, (1, 1, 32, 1))  # both leading dims = 1

        fixed = canonicalize_singleton_dim_strides(t_deg)

        assert fixed.stride(0) == 1 * 8 * 32  # canonical: 256
        assert fixed.stride(1) == 1 * 8 * 32  # canonical: 256 (same since size-1)
        assert fixed.stride(2) == 32
        assert fixed.stride(3) == 1

    def test_various_shapes_flashinfer(self):
        """Correctness across different block_size / head_size for FlashInfer layout."""
        for block_size, head_size in [(16, 64), (16, 128), (32, 128), (16, 256)]:
            t = torch.zeros(8, 2, 1, block_size, head_size, dtype=torch.bfloat16)
            t_deg = _inject_degenerate_stride(t, dim=-3)
            fixed = canonicalize_singleton_dim_strides(t_deg)
            assert fixed.stride(-3) == block_size * head_size, (
                f"Failed for block_size={block_size}, head_size={head_size}: "
                f"got stride(-3)={fixed.stride(-3)}"
            )

    def test_various_shapes_flash_attn(self):
        """Correctness across different shapes for FlashAttention layout."""
        for block_size, head_size in [(16, 64), (16, 128), (32, 128)]:
            t = torch.zeros(8, block_size, 1, head_size, dtype=torch.bfloat16)
            t_deg = _inject_degenerate_stride(t, dim=-2)
            fixed = canonicalize_singleton_dim_strides(t_deg)
            assert fixed.stride(-2) == head_size, (
                f"Failed for block_size={block_size}, head_size={head_size}: "
                f"got stride(-2)={fixed.stride(-2)}"
            )

    def test_tma_alignment_satisfied_after_fix_bf16(self):
        """After fix, all strides meet 16-byte TMA alignment for bf16."""
        t = torch.zeros(64, 2, 1, 16, 128, dtype=torch.bfloat16)
        t_deg = _inject_degenerate_stride(t, dim=-3)
        fixed = canonicalize_singleton_dim_strides(t_deg)

        element_size = fixed.element_size()  # 2 bytes for bf16
        for i, s in enumerate(fixed.stride()):
            assert (s * element_size) % 16 == 0 or i == len(fixed.stride()) - 1, (
                f"dim {i} stride {s} * {element_size} bytes not 16-byte aligned"
            )

    def test_non_contiguous_outer_dims_preserved(self):
        """Outer (non-size-1) non-contiguous strides are left unchanged."""
        # Simulate cross-layer unified allocation: num_blocks stride is non-canonical
        # but the inner dims should be fixed.
        base = torch.zeros(200, 2, 1, 16, 128, dtype=torch.bfloat16)
        # Slice every 2nd block → non-canonical outer stride
        t_sliced = base[::2]  # shape [100, 2, 1, 16, 128], stride[0] = 2*canonical
        t_deg = _inject_degenerate_stride(t_sliced, dim=-3)

        fixed = canonicalize_singleton_dim_strides(t_deg)

        # Outer stride should be unchanged (not a size-1 dim)
        assert fixed.stride(0) == t_sliced.stride(0)
        # Inner degenerate stride should be fixed
        assert fixed.stride(-3) == 16 * 128
