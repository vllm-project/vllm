# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm.v1.attention.ops.triton_prefill_attention as prefill_ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_prefill_attention import (
    context_attention_fwd,
)

DEVICE_TYPE = current_platform.device_type


def ref_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
) -> torch.Tensor:
    """Reference implementation using PyTorch SDPA."""
    # q, k, v: [total_tokens, num_heads, head_dim]
    # SDPA expects [batch, num_heads, seq_len, head_dim]

    total_tokens = q.shape[0]

    # Add batch dimension and transpose
    q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_tokens, head_dim]
    k = k.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_tokens, head_dim]
    v = v.unsqueeze(0).transpose(1, 2)  # [1, num_heads, total_tokens, head_dim]

    # Create attention mask if needed
    attn_mask = None
    use_causal = is_causal

    # If we have sliding window or need custom masking, create explicit mask
    sliding_window_q = sliding_window_q if sliding_window_q is not None else 0
    sliding_window_k = sliding_window_k if sliding_window_k is not None else 0
    if (sliding_window_q > 0) or (sliding_window_k > 0):
        # Position indices
        pos_q = torch.arange(total_tokens, device=q.device).unsqueeze(1)
        pos_k = torch.arange(total_tokens, device=q.device).unsqueeze(0)

        # Start with valid mask (False = no masking)
        mask = torch.ones(
            (total_tokens, total_tokens), dtype=torch.bool, device=q.device
        )

        # Apply causal mask
        if is_causal:
            mask = mask & (pos_q >= pos_k)

        # Apply sliding window masks
        sliding_window_mask = torch.ones_like(mask)
        if sliding_window_q > 0:
            sliding_window_mask &= pos_q - pos_k <= sliding_window_q

        if sliding_window_k > 0:
            sliding_window_mask &= pos_k - pos_q <= sliding_window_k

        mask = mask & sliding_window_mask

        attn_mask = torch.where(mask, 0.0, float("-inf")).to(q.dtype)
        use_causal = False  # Don't use is_causal when providing explicit mask

    # Use SDPA
    output = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=use_causal, dropout_p=0.0
    )

    # Convert back to original shape: [total_tokens, num_heads, head_dim]
    output = output.transpose(1, 2).squeeze(0)

    return output


@pytest.mark.parametrize("B", [5])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [64, 72, 80, 96, 128])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_attention(
    B: int,
    max_seq_len: int,
    H_Q: int,
    H_KV: int,
    D: int,
    is_causal: bool,
    dtype: torch.dtype,
):
    """Test basic context attention without sliding window."""
    torch.manual_seed(42)

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(
        max_seq_len // 2, max_seq_len + 1, (B,), device=DEVICE_TYPE
    )
    total_tokens = seq_lens.sum().item()

    # Create batch start locations
    b_start_loc = torch.zeros(B, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    # Create input tensors
    q = torch.randn(total_tokens, H_Q, D, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    o = torch.zeros_like(q)

    # Call Triton kernel
    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        seq_lens,
        max_seq_len,
        is_causal=is_causal,
        sliding_window_q=None,
        sliding_window_k=None,
    )

    # Compute reference output for each sequence in batch
    o_ref = torch.zeros_like(q)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()

        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]

        # Expand KV heads if using GQA
        if H_Q != H_KV:
            kv_group_num = H_Q // H_KV
            k_seq = k_seq.repeat_interleave(kv_group_num, dim=1)
            v_seq = v_seq.repeat_interleave(kv_group_num, dim=1)

        o_ref[start:end] = ref_masked_attention(
            q_seq,
            k_seq,
            v_seq,
            is_causal=is_causal,
            sliding_window_q=None,
            sliding_window_k=None,
        )

    # Compare outputs
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("H_Q", [32])
@pytest.mark.parametrize("H_KV", [32, 8])
@pytest.mark.parametrize("D", [64, 72, 80, 96, 128])
@pytest.mark.parametrize("sliding_window", [(32, 32), (32, 0), (0, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_context_attention_sliding_window(
    B: int,
    max_seq_len: int,
    H_Q: int,
    H_KV: int,
    D: int,
    sliding_window: tuple[int, int],
    dtype: torch.dtype,
):
    sliding_window_q, sliding_window_k = sliding_window
    """Test context attention with sliding window."""
    torch.manual_seed(42)

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(
        max_seq_len // 2, max_seq_len + 1, (B,), device=DEVICE_TYPE
    )
    total_tokens = seq_lens.sum().item()

    # Create batch start locations
    b_start_loc = torch.zeros(B, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    # Create input tensors
    q = torch.randn(total_tokens, H_Q, D, dtype=dtype, device=DEVICE_TYPE)
    k = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    v = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=DEVICE_TYPE)
    o = torch.zeros_like(q)

    # Call Triton kernel
    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        seq_lens,
        max_seq_len,
        is_causal=False,
        sliding_window_q=sliding_window_q,
        sliding_window_k=sliding_window_k,
    )

    # Compute reference output for each sequence in batch
    o_ref = torch.zeros_like(q)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()

        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]

        # Expand KV heads if using GQA
        if H_Q != H_KV:
            kv_group_num = H_Q // H_KV
            k_seq = k_seq.repeat_interleave(kv_group_num, dim=1)
            v_seq = v_seq.repeat_interleave(kv_group_num, dim=1)

        o_ref[start:end] = ref_masked_attention(
            q_seq,
            k_seq,
            v_seq,
            is_causal=False,
            sliding_window_q=sliding_window_q if sliding_window_q > 0 else None,
            sliding_window_k=sliding_window_k if sliding_window_k > 0 else None,
        )

    # Compare outputs
    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


class _LaunchCapture:
    """Records the launch configuration in place of ``_fwd_kernel``."""

    grid: tuple
    kwargs: dict

    def __getitem__(self, grid):
        self.grid = grid
        return self._record

    def _record(self, *args, **kwargs) -> None:
        self.kwargs = kwargs


def _capture_tile_config(
    monkeypatch,
    *,
    is_rocm: bool,
    on_gfx1x: bool,
    dtype=torch.bfloat16,
    head_dim: int = 128,
    on_gfx1151: bool = False,
    head_stride_pad: int = 0,
) -> _LaunchCapture:
    """Capture the tile configuration with the platform predicates mocked.

    Both tile widths are numerically correct, so the tests above pass whichever
    one is selected. This needs no RDNA part and allocates no device memory, so
    it covers the RDNA branch on the CDNA and NVIDIA agents CI actually runs.
    """
    platform = prefill_ops.current_platform
    monkeypatch.setattr(platform, "is_rocm", lambda: is_rocm)
    # Every device this kernel targets is cuda-alike at capability 80 or better,
    # so the stock tile is 128 for 16-bit dtypes.
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(platform, "has_device_capability", lambda *a, **k: True)
    if is_rocm:
        import vllm.platforms.rocm as rocm_platform

        monkeypatch.setattr(rocm_platform, "on_gfx1x", lambda: on_gfx1x)
        monkeypatch.setattr(rocm_platform, "on_gfx1151", lambda: on_gfx1151)

    capture = _LaunchCapture()
    monkeypatch.setattr(prefill_ops, "_fwd_kernel", capture)

    def meta(tokens, heads, dim):
        # Padding the head axis gives a stride the alignment hint cannot claim.
        wide = torch.empty(
            (tokens, heads, dim + head_stride_pad), dtype=dtype, device="meta"
        )
        return wide[..., :dim]

    seq_lens = torch.empty(2, dtype=torch.int32, device="meta")
    context_attention_fwd(
        meta(256, 8, head_dim),
        meta(256, 2, head_dim),
        meta(256, 2, head_dim),
        meta(256, 8, head_dim),
        seq_lens,
        seq_lens,
        128,
    )
    return capture


@pytest.mark.parametrize(
    ("is_rocm", "on_gfx1x", "dtype", "expected_block_n"),
    [
        pytest.param(True, True, torch.bfloat16, 32, id="rdna"),
        # on_gfx1x() is what excludes gfx10xx, CDNA and gfx1250.
        pytest.param(True, False, torch.bfloat16, 128, id="rocm-not-rdna"),
        pytest.param(False, False, torch.bfloat16, 128, id="not-rocm"),
        # get_block_size already returns 32 here, so min() must not widen it.
        pytest.param(True, True, torch.float32, 32, id="rdna-float32"),
    ],
)
def test_kv_tile_width_is_gated_by_platform(
    monkeypatch, is_rocm: bool, on_gfx1x: bool, dtype, expected_block_n: int
) -> None:
    capture = _capture_tile_config(
        monkeypatch, is_rocm=is_rocm, on_gfx1x=on_gfx1x, dtype=dtype
    )
    assert capture.kwargs["BLOCK_N"] == expected_block_n


def test_rdna_narrows_the_kv_tile_and_nothing_else(monkeypatch) -> None:
    with monkeypatch.context() as m:
        tuned = _capture_tile_config(m, is_rocm=True, on_gfx1x=True)
    with monkeypatch.context() as m:
        stock = _capture_tile_config(m, is_rocm=True, on_gfx1x=False)

    assert tuned.kwargs.pop("BLOCK_N") != stock.kwargs.pop("BLOCK_N")
    assert tuned.kwargs == stock.kwargs
    assert tuned.grid == stock.grid


@pytest.mark.parametrize("D", [72, 128])
def test_context_attention_non_contiguous_heads(D: int):
    """A head stride the 16-byte alignment hint must not be applied to.

    The hint is keyed off the runtime strides, not head_dim, so a view whose
    head axis is not 8-element aligned has to fall back and stay correct.
    """
    torch.manual_seed(42)
    B, S, H = 2, 256, 8
    dtype = torch.bfloat16
    total_tokens = B * S

    seq_lens = torch.full((B,), S, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc = torch.zeros(B, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)

    # Slicing a padded head axis keeps D contiguous but breaks the 8-element
    # alignment of the head stride.
    q, k, v, o = (
        torch.randn(total_tokens, H, D + 4, dtype=dtype, device=DEVICE_TYPE)[..., :D]
        for _ in range(4)
    )
    assert q.stride(1) % 8 != 0

    context_attention_fwd(
        q, k, v, o, b_start_loc, seq_lens, S, is_causal=True, sliding_window_q=None
    )

    o_ref = torch.zeros_like(o)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()
        o_ref[start:end] = ref_masked_attention(
            q[start:end], k[start:end], v[start:end], is_causal=True
        )

    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [
        (64, (64, 0)),  # already a power of 2
        (72, (64, 16)),  # SigLIP, Qwen3-VL
        (80, (64, 16)),  # Qwen2.5-VL
        (96, (64, 32)),  # the other tail width the kernel can emit
        (112, (128, 0)),  # 64 + 64 covers the same 128 lanes, so don't split
        (128, (128, 0)),
    ],
)
def test_gfx1151_splits_the_head_dim(monkeypatch, head_dim, expected) -> None:
    """A non-power-of-2 head dim is covered by a main block plus a tail."""
    capture = _capture_tile_config(
        monkeypatch, is_rocm=True, on_gfx1x=True, on_gfx1151=True, head_dim=head_dim
    )
    got = (capture.kwargs["BLOCK_DMODEL"], capture.kwargs["BLOCK_DMODEL_TAIL"])
    assert got == expected


@pytest.mark.parametrize("head_dim", [72, 80])
def test_gfx1151_narrows_the_tile_for_the_split_dims(monkeypatch, head_dim) -> None:
    """The 64 + 16 head dims take a narrower tile and 4 warps."""
    capture = _capture_tile_config(
        monkeypatch, is_rocm=True, on_gfx1x=True, on_gfx1151=True, head_dim=head_dim
    )
    assert capture.kwargs["BLOCK_N"] == 16
    assert capture.kwargs["num_warps"] == 4


@pytest.mark.parametrize("head_dim", [64, 96, 128])
def test_gfx1151_leaves_the_other_head_dims_on_the_rdna_tile(
    monkeypatch, head_dim
) -> None:
    """Only the 64 + 16 dims are faster narrower; the rest keep the RDNA tile."""
    capture = _capture_tile_config(
        monkeypatch, is_rocm=True, on_gfx1x=True, on_gfx1151=True, head_dim=head_dim
    )
    assert capture.kwargs["BLOCK_N"] == 32


@pytest.mark.parametrize("head_dim", [64, 72, 80, 96, 128])
def test_alignment_hint_is_not_tied_to_the_tile(monkeypatch, head_dim) -> None:
    """The hint is an alignment fact, so it does not depend on the KV tile.

    It only tells the compiler something new when head_dim is 8 mod 16;
    elsewhere it is a tautology and measures exactly neutral.
    """
    capture = _capture_tile_config(
        monkeypatch, is_rocm=True, on_gfx1x=True, on_gfx1151=True, head_dim=head_dim
    )
    assert capture.kwargs["HEAD_STRIDE_ALIGNED_8"] is True


@pytest.mark.parametrize("head_dim", [72, 96])
def test_nothing_changes_off_gfx1151(monkeypatch, head_dim) -> None:
    """Every other part keeps the launch configuration it has today."""
    with pytest.MonkeyPatch.context() as m:
        tuned = _capture_tile_config(
            m, is_rocm=True, on_gfx1x=True, on_gfx1151=True, head_dim=head_dim
        )
    stock = _capture_tile_config(
        monkeypatch, is_rocm=True, on_gfx1x=True, on_gfx1151=False, head_dim=head_dim
    )
    assert stock.kwargs["BLOCK_DMODEL"] == triton.next_power_of_2(head_dim)
    assert stock.kwargs["BLOCK_DMODEL_TAIL"] == 0
    assert stock.kwargs["HEAD_STRIDE_ALIGNED_8"] is False
    assert stock.kwargs != tuned.kwargs


@pytest.mark.parametrize("head_dim", [72, 96])
def test_gfx1151_split_matches_the_reference(monkeypatch, head_dim) -> None:
    """Run the split-D kernel for real, so CI covers it off gfx1151 too."""
    import vllm.platforms.rocm as rocm_platform

    if not current_platform.is_rocm():
        pytest.skip("needs the ROCm platform module")
    monkeypatch.setattr(rocm_platform, "on_gfx1151", lambda: True)

    torch.manual_seed(42)
    B, S, H = 2, 256, 8
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc = torch.zeros(B, dtype=torch.int32, device=DEVICE_TYPE)
    b_start_loc[1:] = torch.cumsum(seq_lens[:-1], dim=0)
    q, k, v = (
        torch.randn(B * S, H, head_dim, dtype=torch.bfloat16, device=DEVICE_TYPE)
        for _ in range(3)
    )
    o = torch.zeros_like(q)
    context_attention_fwd(q, k, v, o, b_start_loc, seq_lens, S, is_causal=True)

    o_ref = torch.zeros_like(o)
    for i in range(B):
        start = b_start_loc[i].item()
        end = start + seq_lens[i].item()
        o_ref[start:end] = ref_masked_attention(
            q[start:end], k[start:end], v[start:end], is_causal=True
        )
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


def test_narrow_tile_needs_the_alignment_hint(monkeypatch) -> None:
    """A padded head stride disables the hint, and then the wide tile wins.

    head_dim 72 still splits as 64 + 16, but a view whose head stride is not
    8-aligned cannot assert the alignment, and without vectorized loads the
    narrow tile measures ~9% slower than the one #58225 picks.
    """
    capture = _capture_tile_config(
        monkeypatch,
        is_rocm=True,
        on_gfx1x=True,
        on_gfx1151=True,
        head_dim=72,
        head_stride_pad=4,
    )
    assert capture.kwargs["BLOCK_DMODEL_TAIL"] == 16, "still a 64 + 16 split"
    assert capture.kwargs["HEAD_STRIDE_ALIGNED_8"] is False
    assert capture.kwargs["BLOCK_N"] == 32
