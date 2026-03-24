# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the full FP8 ViT attention path (quantize -> cuDNN -> un-pad)."""

import contextlib

import pytest
import torch

from vllm.triton_utils import HAS_TRITON
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


def _flashinfer_fp8_supported() -> bool:
    """Check if FlashInfer cuDNN FP8 is supported on this platform."""
    try:
        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        supported = current_platform.get_supported_vit_attn_backends()
        if AttentionBackendEnum.FLASHINFER not in supported:
            return False
    except (ImportError, AttributeError):
        return False

    # cuDNN FP8 requires >= 9.17.1
    try:
        import torch.backends.cudnn as cudnn

        if cudnn.is_available():
            ver = cudnn.version()
            if ver < 91701:
                return False
    except (ImportError, AttributeError):
        pass

    return True


@pytest.fixture
def _fp8_attention(monkeypatch, default_vllm_config):
    """Create FP8-enabled MMEncoderAttention via monkeypatch."""
    from unittest.mock import patch

    from vllm.envs import disable_envs_cache
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not _flashinfer_fp8_supported():
        yield
        return

    monkeypatch.setenv("VLLM_MM_ENCODER_FP8_ATTN", "1")
    monkeypatch.delenv("VLLM_MM_ENCODER_FP8_ATTN_SCALE_PATH", raising=False)
    disable_envs_cache()

    with patch(
        "vllm.model_executor.layers.attention.mm_encoder_attention"
        ".get_vit_attn_backend",
        return_value=AttentionBackendEnum.FLASHINFER,
    ):
        yield

    disable_envs_cache()


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

    padded_dim = round_up(head_dim, 16)
    fp8_hidden = num_heads * padded_dim if padded_dim != head_dim else None

    cu_seqlens, max_seqlen, sequence_lengths = _build_cu_seqlens_and_meta(
        seq_len, num_heads, head_dim, fp8_padded_hidden_size=fp8_hidden
    )

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

    output = attn._forward_flashinfer(q, k, v, cu_seqlens, max_seqlen, sequence_lengths)

    # Output should have original head_dim (un-padded)
    assert output.shape[-1] == head_dim
    assert output.dtype == torch.bfloat16


@pytest.mark.skipif(
    not (HAS_TRITON and _has_flashinfer_cudnn()),
    reason="Triton and FlashInfer cuDNN required",
)
def test_fp8_vs_bf16_close(_fp8_attention) -> None:
    """FP8 attention output should be reasonably close to BF16 baseline."""
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )

    head_dim, seq_len, num_heads = 80, 128, 16

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

    cu_seqlens, max_seqlen, seq_lengths = _build_cu_seqlens_and_meta(
        seq_len, num_heads, head_dim
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

    diff = (out_fp8.float() - out_bf16.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # FP8 quantization introduces error; check reasonable bounds
    assert max_diff < 0.5, f"FP8 vs BF16 max diff too large: {max_diff}"
    assert mean_diff < 0.05, f"FP8 vs BF16 mean diff too large: {mean_diff}"

    # Also check cosine similarity
    out_fp8_flat = out_fp8.flatten().float()
    out_bf16_flat = out_bf16.flatten().float()
    cosine_sim = torch.nn.functional.cosine_similarity(
        out_fp8_flat.unsqueeze(0), out_bf16_flat.unsqueeze(0)
    ).item()
    assert cosine_sim > 0.99, f"Cosine similarity too low: {cosine_sim:.6f}"
