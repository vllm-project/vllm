"""Unit tests for FlashInfer sparse MLA warmup C128A width coverage.

Verifies that the warmup path exercises every runtime-reachable C128A
width (extra_topk = 128, 256, 384 for max_model_len=33792),
not just the default short-sequence width (128).
"""

from unittest.mock import MagicMock, patch

import pytest


def test_c128a_max_width_calculation():
    """Verify that cdiv(33792/128) aligned to 128 equals 384."""
    from vllm.utils.math_utils import cdiv

    max_model_len = 33792
    compress_ratio = 128
    alignment = 128

    raw = cdiv(max_model_len, compress_ratio)
    aligned = cdiv(raw, alignment) * alignment
    assert aligned == 384, f"Expected 384, got {aligned}"


def test_compute_c128a_reachable_widths():
    """Verify that _compute_c128a_reachable_widths returns [128, 256, 384]."""
    from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
        _compute_c128a_reachable_widths,
    )

    result = _compute_c128a_reachable_widths(33792)
    widths = [w for w, _ in result]
    assert widths == [128, 256, 384], f"Expected [128, 256, 384], got {widths}"

    for width, seq_len in result:
        if width == 128:
            assert seq_len == 1
        elif width == 256:
            assert 16512 <= seq_len <= 32895
        elif width == 384:
            assert seq_len >= 32896


def test_warmup_calls_all_reachable_widths(mock_worker):
    """Verify warmup calls autotune for every reachable C128A width."""
    with patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._deepseek_v4_sparse_mla_decode_autotune"
    ) as mock_autotune, patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._has_deepseek_v4_sparse_mla_backend",
        return_value=True,
    ), patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._uses_v2_model_runner",
        return_value=False,
    ):
        mock_autotune.return_value = True

        from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
            deepseek_v4_sparse_mla_attention_warmup,
        )

        deepseek_v4_sparse_mla_attention_warmup(mock_worker)

        assert mock_autotune.call_count == 3

        first_kwargs = mock_autotune.call_args_list[0].kwargs
        assert first_kwargs.get("profile_seq_lens") is None

        for call in mock_autotune.call_args_list[1:]:
            assert call.kwargs.get("profile_seq_lens") is not None
            assert call.kwargs["profile_seq_lens"] >= 16512


@pytest.fixture
def mock_worker():
    worker = MagicMock()
    worker.model_runner.is_pooling_model = False
    worker.scheduler_config.max_num_batched_tokens = 8448
    worker.vllm_config.model_config.max_model_len = 33792
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = True
    worker.vllm_config.use_v2_model_runner = False
    worker.model_runner.max_num_reqs = 32
    return worker
