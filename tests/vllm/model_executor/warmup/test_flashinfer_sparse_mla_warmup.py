"""Unit tests for FlashInfer sparse MLA warmup C128A width coverage.

Verifies that the warmup path exercises every runtime-reachable C128A
width (extra_topk = 128, 256, 384 for max_model_len=33792),
not just the default short-sequence width (128).
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_worker():
    """Create a minimal mock worker with the attributes accessed during warmup."""
    worker = MagicMock()
    worker.model_runner.is_pooling_model = False
    worker.scheduler_config.max_num_batched_tokens = 8448
    worker.vllm_config.model_config.max_model_len = 33792
    worker.vllm_config.kernel_config.enable_flashinfer_autotune = True
    worker.vllm_config.use_v2_model_runner = False
    worker.model_runner.max_num_reqs = 32
    return worker


# ---------------------------------------------------------------------------
# Test: C128A width math
# ---------------------------------------------------------------------------


def test_c128a_max_width_calculation():
    """Verify that cdiv(33792/128) aligned to 128 equals 384."""
    from vllm.utils.math_utils import cdiv

    max_model_len = 33792
    compress_ratio = 128
    alignment = 128

    raw = cdiv(max_model_len, compress_ratio)
    aligned = cdiv(raw, alignment) * alignment
    assert aligned == 384, f"Expected 384, got {aligned}"


# ---------------------------------------------------------------------------
# Tests: _compute_c128a_reachable_widths
# ---------------------------------------------------------------------------


def test_compute_c128a_reachable_widths_33792():
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


def test_compute_c128a_reachable_widths_short_model():
    """When max_model_len is small, only 128 should be reachable."""
    from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
        _compute_c128a_reachable_widths,
    )

    result = _compute_c128a_reachable_widths(4096)
    widths = [w for w, _ in result]
    assert widths == [128], f"Expected [128], got {widths}"


def test_compute_c128a_reachable_widths_medium_model():
    """When max_model_len=20000, widths should be [128, 256]."""
    from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
        _compute_c128a_reachable_widths,
    )

    result = _compute_c128a_reachable_widths(20000)
    widths = [w for w, _ in result]
    assert widths == [128, 256], f"Expected [128, 256], got {widths}"


def test_compute_c128a_reachable_widths_boundary():
    """At the exact boundary seq_len=16512, width should transition to 256."""
    from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
        _compute_c128a_reachable_widths,
    )

    result = _compute_c128a_reachable_widths(16512)
    widths = [w for w, _ in result]
    assert 256 in widths
    assert 384 not in widths


# ---------------------------------------------------------------------------
# Tests: deepseek_v4_sparse_mla_attention_warmup
# ---------------------------------------------------------------------------


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

        # 3 calls: 1 default (128) + 2 extra (256, 384)
        assert mock_autotune.call_count == 3

        # First call: no profile_seq_lens (default short-sequence)
        first_kwargs = mock_autotune.call_args_list[0].kwargs
        assert first_kwargs.get("profile_seq_lens") is None

        # Subsequent calls: profile_seq_lens set for widths > 128
        for call in mock_autotune.call_args_list[1:]:
            assert call.kwargs.get("profile_seq_lens") is not None
            assert call.kwargs["profile_seq_lens"] >= 16512


def test_warmup_skips_extra_passes_for_short_model(mock_worker):
    """When max_model_len is small (only 128 reachable), no extra passes."""
    mock_worker.vllm_config.model_config.max_model_len = 4096
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

        # Only 1 call: default (128), no extra passes
        assert mock_autotune.call_count == 1


def test_warmup_skips_extra_passes_for_v2_runner(mock_worker):
    """V2 model runner should skip extra C128A width passes."""
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
        return_value=True,
    ):
        mock_autotune.return_value = True

        from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
            deepseek_v4_sparse_mla_attention_warmup,
        )

        deepseek_v4_sparse_mla_attention_warmup(mock_worker)

        # Only 1 call: default (128); V2 skips extra passes
        assert mock_autotune.call_count == 1


def test_warmup_pooling_model_early_return(mock_worker):
    """Pooling model should return without any autotune calls."""
    mock_worker.model_runner.is_pooling_model = True
    with patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._deepseek_v4_sparse_mla_decode_autotune"
    ) as mock_autotune:
        from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
            deepseek_v4_sparse_mla_attention_warmup,
        )

        deepseek_v4_sparse_mla_attention_warmup(mock_worker)

        assert mock_autotune.call_count == 0


def test_warmup_no_dsv4_backend_early_return(mock_worker):
    """No DSv4 sparse MLA backend should return without autotune calls."""
    with patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._has_deepseek_v4_sparse_mla_backend",
        return_value=False,
    ), patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._deepseek_v4_sparse_mla_decode_autotune"
    ) as mock_autotune:
        from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
            deepseek_v4_sparse_mla_attention_warmup,
        )

        deepseek_v4_sparse_mla_attention_warmup(mock_worker)

        assert mock_autotune.call_count == 0


def test_warmup_fallback_when_autotune_fails(mock_worker):
    """When autotune returns False (mixed_warmup_done=False), fallback to
    a plain dummy_run without profile_seq_lens."""
    with patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._deepseek_v4_sparse_mla_decode_autotune",
        return_value=False,
    ) as mock_autotune, patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._has_deepseek_v4_sparse_mla_backend",
        return_value=True,
    ), patch(
        "vllm.model_executor.warmup.flashinfer_sparse_mla_warmup"
        "._uses_v2_model_runner",
        return_value=False,
    ), patch.object(
        mock_worker.model_runner, "_dummy_run"
    ) as mock_dummy_run:
        from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
            deepseek_v4_sparse_mla_attention_warmup,
        )

        deepseek_v4_sparse_mla_attention_warmup(mock_worker)

        # Autotune called once (default pass only, no extra since it failed)
        assert mock_autotune.call_count == 1
        # Fallback dummy_run called once
        assert mock_dummy_run.call_count == 1
        # Fallback should not have profile_seq_lens
        dummy_kwargs = mock_dummy_run.call_args.kwargs
        assert "profile_seq_lens" not in dummy_kwargs
