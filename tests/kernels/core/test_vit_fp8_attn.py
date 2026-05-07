# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the full FP8 ViT attention path (quantize -> cuDNN -> un-pad)."""

import contextlib

import pytest
import torch

from vllm.triton_utils import HAS_TRITON
from vllm.utils.flashinfer import (
    is_flashinfer_cudnn_fp8_prefill_attn_supported,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _has_flashinfer_cudnn() -> bool:
    """Check if FlashInfer cuDNN backend is available."""
    try:
        from flashinfer.prefill import (
            cudnn_batch_prefill_with_kv_cache,  # noqa: F401
        )

        return True
    except ImportError:
        return False


HEAD_DIMS = [72, 80]
SEQ_LENS = [256]
NUM_HEADS = [16]


@pytest.fixture
def _fp8_attention():
    """Create FP8-enabled MMEncoderAttention via config."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig

    if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
        pytest.skip("FlashInfer cuDNN FP8 prefill attention not supported")

    mm_config = MultiModalConfig(mm_encoder_attn_dtype="fp8")
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(multimodal_config=mm_config)

    # MMEncoderAttention reads torch.get_default_dtype() during init
    # to determine the output dtype. In real model loading this is bf16.
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    with (
        set_current_vllm_config(vllm_config),
        patch(
            "vllm.model_executor.layers.attention.mm_encoder_attention"
            ".get_vit_attn_backend",
            return_value=AttentionBackendEnum.FLASHINFER,
        ),
    ):
        yield

    torch.set_default_dtype(old_dtype)


def _build_cu_seqlens_and_meta(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    fp8_padded_hidden_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build cu_seqlens, max_seqlen, sequence_lengths for a single sequence."""
    import numpy as np

    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    cu_seqlens_np = np.array([0, seq_len], dtype=np.int32)

    sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
        AttentionBackendEnum.FLASHINFER,
        cu_seqlens_np,
        torch.device("cuda"),
    )

    max_seqlen = torch.tensor(
        MMEncoderAttention.compute_max_seqlen(
            AttentionBackendEnum.FLASHINFER, cu_seqlens_np
        ),
        dtype=torch.int32,
    )

    cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
        AttentionBackendEnum.FLASHINFER,
        cu_seqlens_np,
        num_heads * head_dim,
        1,  # tp_size
        torch.device("cuda"),
        fp8_padded_hidden_size=fp8_padded_hidden_size,
    )

    return cu_seqlens, max_seqlen, sequence_lengths


@pytest.mark.skipif(
    not (HAS_TRITON and _has_flashinfer_cudnn()),
    reason="Triton and FlashInfer cuDNN required",
)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_fp8_attn_output_shape(
    head_dim: int,
    seq_len: int,
    num_heads: int,
    _fp8_attention,
) -> None:
    """Verify FP8 attention produces correct output shape after un-padding."""
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    from vllm.utils.math_utils import round_up

    attn = None
    with contextlib.suppress(ValueError, ImportError):
        attn = MMEncoderAttention(
            num_heads=num_heads,
            head_size=head_dim,
            prefix="visual.blocks.0.attn",
        ).to("cuda")

    if attn is None or not attn.fp8_enabled:
        pytest.skip("FP8 MMEncoderAttention not available")
    assert attn is not None  # mypy narrowing

    # FP8 always needs fp8_padded_hidden_size for correct cu_seqlens
    fp8_padded_hidden_size = num_heads * round_up(head_dim, 16)

    cu_seqlens, max_seqlen, sequence_lengths = _build_cu_seqlens_and_meta(
        seq_len, num_heads, head_dim, fp8_padded_hidden_size=fp8_padded_hidden_size
    )

    q = torch.randn(
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = attn._forward_flashinfer(q, k, v, cu_seqlens, max_seqlen, sequence_lengths)

    # Output should have original head_dim (un-padded)
    assert output.shape[-1] == head_dim
    assert output.dtype == torch.bfloat16


@pytest.mark.skipif(
    not (HAS_TRITON and _has_flashinfer_cudnn()),
    reason="Triton and FlashInfer cuDNN required",
)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
def test_fp8_vs_bf16_close(
    head_dim: int, seq_len: int, num_heads: int, _fp8_attention
) -> None:
    """FP8 attention output should be reasonably close to BF16 baseline."""
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    from vllm.utils.math_utils import round_up

    torch.manual_seed(42)
    q = torch.randn(
        1,
        seq_len,
        num_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # FP8 path
    attn_fp8 = None
    with contextlib.suppress(ValueError, ImportError):
        attn_fp8 = MMEncoderAttention(
            num_heads=num_heads,
            head_size=head_dim,
            prefix="visual.blocks.0.attn",
        ).to("cuda")

    if attn_fp8 is None or not attn_fp8.fp8_enabled:
        pytest.skip("FP8 MMEncoderAttention not available")
    assert attn_fp8 is not None  # mypy narrowing

    fp8_padded_hidden_size = num_heads * round_up(head_dim, 16)
    cu_seqlens, max_seqlen, seq_lengths = _build_cu_seqlens_and_meta(
        seq_len,
        num_heads,
        head_dim,
        fp8_padded_hidden_size=fp8_padded_hidden_size,
    )

    out_fp8 = attn_fp8._forward_flashinfer(
        q.clone(),
        k.clone(),
        v.clone(),
        cu_seqlens,
        max_seqlen,
        seq_lengths,
    )

    # BF16 baseline (create non-FP8 attention by using scale=attn_fp8.scale
    # and calling the wrapper directly without FP8 quantization)
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        _get_flashinfer_workspace_buffer,
    )
    from vllm.v1.attention.ops.vit_attn_wrappers import (
        vit_flashinfer_wrapper,
    )

    out_bf16 = vit_flashinfer_wrapper(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        scale=attn_fp8.scale,
        workspace_buffer=_get_flashinfer_workspace_buffer(),
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        sequence_lengths=seq_lengths,
    )

    out_fp8_f = out_fp8.float()
    out_bf16_f = out_bf16.float()

    abs_diff = (out_fp8_f - out_bf16_f).abs()
    abs_diff_flat = abs_diff.flatten()

    # Relative diff (avoid division by zero)
    denom = out_bf16_f.abs().clamp(min=1e-6)
    rel_diff_flat = (abs_diff / denom).flatten()

    cosine_sim = torch.nn.functional.cosine_similarity(
        out_fp8_f.flatten().unsqueeze(0),
        out_bf16_f.flatten().unsqueeze(0),
    ).item()

    pcts = [50, 90, 95, 99, 99.9]
    abs_pct = {p: torch.quantile(abs_diff_flat, p / 100).item() for p in pcts}
    rel_pct = {p: torch.quantile(rel_diff_flat, p / 100).item() for p in pcts}

    print(f"\nFP8 vs BF16 (head_dim={head_dim}, seq_len={seq_len}):")
    print(f"  cosine_sim={cosine_sim:.6f}")
    print(
        f"  abs_diff: max={abs_diff_flat.max().item():.6f}, "
        f"mean={abs_diff_flat.mean().item():.6f}, "
        + ", ".join(f"p{p}={abs_pct[p]:.6f}" for p in pcts)
    )
    print(
        f"  rel_diff: max={rel_diff_flat.max().item():.6f}, "
        f"mean={rel_diff_flat.mean().item():.6f}, "
        + ", ".join(f"p{p}={rel_pct[p]:.6f}" for p in pcts)
    )

    assert abs_diff_flat.max().item() < 0.3, (
        f"FP8 vs BF16 max abs diff too large: {abs_diff_flat.max().item()}"
    )
    assert abs_diff_flat.mean().item() < 0.03, (
        f"FP8 vs BF16 mean abs diff too large: {abs_diff_flat.mean().item()}"
    )
    assert cosine_sim > 0.99, f"Cosine similarity too low: {cosine_sim:.6f}"
