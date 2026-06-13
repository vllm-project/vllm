# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from vllm.config.attention import AttentionConfig
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.fa_utils import (
    compute_fa3_num_splits,
    resolve_fa_num_splits,
)


@pytest.mark.parametrize(
    "kv, seqs, sms, expected",
    [
        (1, 1, 132, 132),  # 1 KV head → use all SMs
        (1, 32, 132, 32),  # enough CTAs, stays at floor
        (8, 32, 132, 32),  # way more CTAs than SMs
        (1, 4, 132, 33),  # ceil(132/4)
        (4, 1, 132, 33),  # ceil(132/4)
        (1, 1, 300, 256),  # capped at FA3 max
        (1, 1, 192, 192),  # B200-like
    ],
)
def test_compute_fa3_num_splits(kv, seqs, sms, expected):
    assert compute_fa3_num_splits(kv, seqs, sms) == expected


@contextmanager
def _mock_hopper(num_sms=132):
    m = MagicMock()
    m.is_cuda.return_value = True
    m.is_device_capability.side_effect = lambda cap: cap == 90
    m.get_device_capability.return_value = DeviceCapability(9, 0)
    m.num_compute_units.return_value = num_sms
    with patch("vllm.v1.attention.backends.fa_utils.current_platform", m):
        yield


def _stub(kv_heads=1, seqs=1, fa_version=None, max_model_len=262144):
    cfg = MagicMock()
    cfg.model_config.get_num_kv_heads.return_value = kv_heads
    cfg.model_config.max_model_len = max_model_len
    cfg.scheduler_config.max_num_seqs = seqs
    cfg.attention_config = AttentionConfig(flash_attn_version=fa_version)
    return cfg


def test_resolve_h100_1kv():
    with _mock_hopper(132):
        assert resolve_fa_num_splits(_stub(kv_heads=1)) == 132


def test_resolve_many_kv_heads_stays_default():
    with _mock_hopper(132):
        assert resolve_fa_num_splits(_stub(kv_heads=8, seqs=32)) == 32


def test_resolve_fa2_falls_back():
    with _mock_hopper(132):
        assert resolve_fa_num_splits(_stub(fa_version=2)) == 32


def test_resolve_non_cuda_falls_back():
    m = MagicMock()
    m.is_cuda.return_value = False
    with patch("vllm.v1.attention.backends.fa_utils.current_platform", m):
        assert resolve_fa_num_splits(_stub()) == 32


def test_resolve_short_context_falls_back():
    """max_model_len below threshold should fall back to default."""
    with _mock_hopper(132):
        assert resolve_fa_num_splits(_stub(kv_heads=1, max_model_len=32768)) == 32


def test_resolve_user_override_wins():
    """If the user sets the config explicitly, the heuristic is bypassed."""
    cfg = _stub(kv_heads=1)
    cfg.attention_config = AttentionConfig(flash_attn_max_num_splits_for_cuda_graph=7)
    with _mock_hopper(132):
        assert resolve_fa_num_splits(cfg) == 7
