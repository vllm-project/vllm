# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GDN FLA chunk-scan workspace accounting."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.chunk_workspace import (
    _format_workspace_bytes,
    apply_gdn_chunk_workspace_reservation,
    estimate_gdn_chunk_workspace,
    max_gdn_chunk_count,
    resolve_gdn_workspace_spec,
    uncovered_gdn_workspace_bytes,
)
from vllm.third_party.flash_linear_attention.ops.utils import (
    FLA_CHUNK_SIZE,
    gdn_workspace_tracker,
)

# Qwen3.8-27B linear-attn geometry (global heads; divide by TP in tests).
_QWEN38_V_HEADS = 48
_HEAD_DIM = 128
_BF16 = 2
_MIB = 1024 * 1024


def _qwen38_estimate(
    num_tokens: int,
    *,
    tp_size: int = 1,
    num_seqs: int = 1,
    worst_case_chunks: bool = False,
    seq_lens: list[int] | None = None,
):
    return estimate_gdn_chunk_workspace(
        num_tokens=num_tokens,
        num_value_heads_local=_QWEN38_V_HEADS // tp_size,
        key_dim=_HEAD_DIM,
        value_dim=_HEAD_DIM,
        dtype_itemsize=_BF16,
        num_seqs=num_seqs,
        seq_lens=seq_lens,
        worst_case_chunks=worst_case_chunks,
    )


def test_qwen38_h_plus_v_new_matches_36kib_per_token():
    est = _qwen38_estimate(8192)
    assert est.num_chunks == 128
    assert est.hv_only_bytes == 288 * _MIB
    assert est.hv_only_bytes / 8192 == 36 * 1024


def test_qwen38_extended_live_set_is_72kib_per_token():
    est = _qwen38_estimate(8192)
    assert est.peak_live_bytes == 576 * _MIB
    assert est.peak_live_bytes / 8192 == 72 * 1024
    assert est.h_bytes == 192 * _MIB
    assert est.v_new_bytes == est.w_bytes == est.u_bytes == est.a_bytes == 96 * _MIB


@pytest.mark.parametrize(
    ("num_tokens", "hv_mib", "peak_mib"),
    [
        (8192, 288, 576),
        (16384, 576, 1152),
        (32768, 1152, 2304),
        (65536, 2304, 4608),
    ],
)
def test_qwen38_scales_linearly_with_tokens(
    num_tokens: int, hv_mib: int, peak_mib: int
):
    est = _qwen38_estimate(num_tokens)
    assert est.hv_only_bytes == hv_mib * _MIB
    assert est.peak_live_bytes == peak_mib * _MIB


def test_tp_shards_value_heads():
    full = _qwen38_estimate(16384, tp_size=1)
    sharded = _qwen38_estimate(16384, tp_size=2)
    assert sharded.num_value_heads_local == 24
    assert sharded.peak_live_bytes == full.peak_live_bytes // 2
    assert sharded.hv_only_bytes == full.hv_only_bytes // 2


def test_even_split_same_tokens_same_chunks():
    one = _qwen38_estimate(8192, num_seqs=1)
    four = _qwen38_estimate(8192, num_seqs=4)
    assert one.num_chunks == four.num_chunks == 128
    assert one.peak_live_bytes == four.peak_live_bytes


def test_worst_case_chunks_inflates_h_only():
    typical = _qwen38_estimate(16384, num_seqs=8, worst_case_chunks=False)
    worst = _qwen38_estimate(16384, num_seqs=8, worst_case_chunks=True)
    assert worst.num_chunks == max_gdn_chunk_count(16384, 8)
    assert worst.num_chunks == 263
    assert typical.num_chunks == 256
    assert worst.v_new_bytes == typical.v_new_bytes
    assert worst.h_bytes > typical.h_bytes
    assert worst.peak_live_bytes - typical.peak_live_bytes == (
        worst.h_bytes - typical.h_bytes
    )


def test_seq_lens_override_uses_varlen_chunk_sum():
    # Eight length-1 sequences plus the remainder would be worst-case.
    rem = 16384 - 7
    est = _qwen38_estimate(16384, seq_lens=[1] * 7 + [rem])
    assert est.num_chunks == 7 + math_ceil(rem, FLA_CHUNK_SIZE)
    assert est.num_chunks == 263


def math_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


def test_format_workspace_bytes_uses_mib_below_one_gib():
    assert _format_workspace_bytes(2 * _MIB).endswith("MiB")
    assert _format_workspace_bytes(2 * 1024 * _MIB).endswith("GiB")


def test_uncovered_delta_is_fail_closed():
    assert uncovered_gdn_workspace_bytes(100, 10) == 90
    assert uncovered_gdn_workspace_bytes(100, 100) == 0
    assert uncovered_gdn_workspace_bytes(100, 150) == 0
    assert uncovered_gdn_workspace_bytes(100, -5) == 100


def test_tracker_disabled_is_noop():
    gdn_workspace_tracker.peak_bytes = 0
    gdn_workspace_tracker.record(1024)
    with gdn_workspace_tracker.track_call():
        gdn_workspace_tracker.record(2048)
    assert gdn_workspace_tracker.peak_bytes == 0


def test_tracker_peak_is_per_call_not_sum_across_layers():
    with gdn_workspace_tracker.collecting():
        with gdn_workspace_tracker.track_call():
            gdn_workspace_tracker.record(100)
            gdn_workspace_tracker.record(50)
        with gdn_workspace_tracker.track_call():
            gdn_workspace_tracker.record(80)
        assert gdn_workspace_tracker.peak_bytes == 150


def test_tracker_without_track_call_does_not_commit_peak():
    with gdn_workspace_tracker.collecting():
        gdn_workspace_tracker.record(4096)
        assert gdn_workspace_tracker.peak_bytes == 0


def _dummy_vllm_config(
    *,
    max_num_batched_tokens: int = 16384,
    max_num_seqs: int = 8,
    tp_size: int = 2,
    dtype=torch.bfloat16,
    linear_heads: int | None = _QWEN38_V_HEADS,
):
    text = SimpleNamespace(
        linear_num_value_heads=linear_heads,
        linear_key_head_dim=_HEAD_DIM,
        linear_value_head_dim=_HEAD_DIM,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text, dtype=dtype),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
    )


class _GdnLayer:
    def __init__(self, backend: str = "triton", tp_size: int = 2):
        self.num_v_heads = _QWEN38_V_HEADS
        self.tp_size = tp_size
        self.head_k_dim = _HEAD_DIM
        self.head_v_dim = _HEAD_DIM
        self.gdn_prefill_backend = backend
        self.model_config = SimpleNamespace(dtype=torch.bfloat16)

    def modules(self):
        yield self


def test_resolve_spec_from_layer_applies_tp():
    spec = resolve_gdn_workspace_spec(_dummy_vllm_config(), _GdnLayer(tp_size=2))
    assert spec is not None
    assert spec.num_value_heads_local == 24
    assert spec.backend == "triton"


def test_apply_reservation_subtracts_uncovered_delta():
    cfg = _dummy_vllm_config()
    upper = _qwen38_estimate(
        16384, tp_size=2, num_seqs=8, worst_case_chunks=True
    ).peak_live_bytes
    covered = _qwen38_estimate(FLA_CHUNK_SIZE, tp_size=2).peak_live_bytes
    available = 8 * 1024**3
    remaining = apply_gdn_chunk_workspace_reservation(
        available, cfg, _GdnLayer(), covered
    )
    assert remaining == available - (upper - covered)
    assert 0 < (upper - covered) < available


def test_apply_reservation_skips_flashinfer():
    available = 8 * 1024**3
    remaining = apply_gdn_chunk_workspace_reservation(
        available,
        _dummy_vllm_config(),
        _GdnLayer(backend="flashinfer"),
        covered_bytes=0,
    )
    assert remaining == available


def test_apply_reservation_skips_non_gdn_models():
    cfg = _dummy_vllm_config(linear_heads=None)
    cfg.model_config.hf_text_config = SimpleNamespace()
    remaining = apply_gdn_chunk_workspace_reservation(
        1024, cfg, model=None, covered_bytes=0
    )
    assert remaining == 1024


def test_apply_reservation_errors_when_kv_cannot_fit():
    cfg = _dummy_vllm_config()
    with pytest.raises(RuntimeError, match="GDN chunk-scan workspace"):
        apply_gdn_chunk_workspace_reservation(1, cfg, _GdnLayer(), covered_bytes=0)


def test_manual_kv_path_logs_but_does_not_subtract():
    available = 4 * 1024**3
    remaining = apply_gdn_chunk_workspace_reservation(
        available,
        _dummy_vllm_config(),
        _GdnLayer(),
        covered_bytes=0,
        apply_reservation=False,
    )
    assert remaining == available
