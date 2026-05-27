# SPDX-License-Identifier: Apache-2.0
"""Verify QuestSparseOffloadImpl delegates forward to FlashAttentionImpl."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch


def _impl_kwargs():
    return dict(
        num_heads=8,
        head_size=64,
        scale=1.0 / (64 ** 0.5),
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
    )


def test_init_constructs_fa_impl_with_same_args():
    fa_target = "vllm.v1.attention.backends.quest.impl.FlashAttentionImpl"
    with patch(fa_target) as fa_mock:
        from vllm.v1.attention.backends.quest.impl import QuestSparseOffloadImpl

        QuestSparseOffloadImpl(**_impl_kwargs())
        fa_mock.assert_called_once()
        # Ensure the kwargs propagated.
        kwargs = fa_mock.call_args.kwargs
        assert kwargs["num_heads"] == 8
        assert kwargs["head_size"] == 64
        assert kwargs["num_kv_heads"] == 8


def test_forward_passes_through_to_fa_forward():
    fa_target = "vllm.v1.attention.backends.quest.impl.FlashAttentionImpl"
    fa_mock_cls = MagicMock()
    fa_mock_instance = MagicMock()
    fa_mock_cls.return_value = fa_mock_instance
    fa_mock_instance.forward.return_value = torch.zeros(1)

    with patch(fa_target, fa_mock_cls):
        from vllm.v1.attention.backends.quest.impl import QuestSparseOffloadImpl

        impl = QuestSparseOffloadImpl(**_impl_kwargs())
        layer = MagicMock()
        q = torch.zeros(2, 8, 64)
        k = torch.zeros(2, 8, 64)
        v = torch.zeros(2, 8, 64)
        kv_cache = torch.zeros(1)
        meta = MagicMock()
        out = torch.zeros(2, 8, 64)

        result = impl.forward(layer, q, k, v, kv_cache, meta, out)

    fa_mock_instance.forward.assert_called_once()
    args, kwargs = fa_mock_instance.forward.call_args
    assert args[0] is layer
    assert torch.equal(args[1], q)
    assert torch.equal(args[2], k)
    assert torch.equal(args[3], v)
    assert torch.equal(args[4], kv_cache)
    assert args[5] is meta
    assert torch.equal(args[6], out)
    assert kwargs["output_scale"] is None
    assert kwargs["output_block_scale"] is None
    assert torch.equal(result, fa_mock_instance.forward.return_value)


def test_kv_cache_dtype_attribute_set():
    fa_target = "vllm.v1.attention.backends.quest.impl.FlashAttentionImpl"
    with patch(fa_target):
        from vllm.v1.attention.backends.quest.impl import QuestSparseOffloadImpl

        impl = QuestSparseOffloadImpl(**_impl_kwargs())
    assert impl.kv_cache_dtype == "auto"
