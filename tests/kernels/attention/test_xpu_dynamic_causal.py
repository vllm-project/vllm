# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import torch


def test_xpu_flash_attn_forwards_dynamic_causal():
    from vllm import _xpu_ops

    q = torch.randn(3, 2, 8)
    k = torch.randn(3, 1, 8)
    v = torch.randn_like(k)
    cu_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    dynamic_causal = torch.tensor([1, 0], dtype=torch.int32)
    sentinel = torch.empty_like(q)

    with patch.object(_xpu_ops, "flash_attn_varlen_func", return_value=sentinel) as op:
        result = _xpu_ops.xpu_ops.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=2,
            max_seqlen_k=2,
            dynamic_causal=dynamic_causal,
        )

    assert result is sentinel
    assert op.call_args.kwargs["dynamic_causal"] is dynamic_causal


def test_xpu_fa2_supports_dynamic_causal(monkeypatch):
    from vllm.v1.attention.backends import flash_attn

    monkeypatch.setattr(flash_attn.current_platform, "is_xpu", lambda: True)
    assert flash_attn._supports_dynamic_causal(2)
    assert flash_attn._supports_dynamic_causal(4)
    assert not flash_attn._supports_dynamic_causal(3)

    monkeypatch.setattr(flash_attn.current_platform, "is_xpu", lambda: False)
    assert not flash_attn._supports_dynamic_causal(2)
    assert flash_attn._supports_dynamic_causal(4)


def test_dynamic_causal_symmetrizes_causal_window():
    from vllm.v1.attention.backends.flash_attn import _maybe_symmetrize_window

    dynamic_causal = torch.tensor([1, 0], dtype=torch.int32)
    assert _maybe_symmetrize_window((32, 0), dynamic_causal) == (32, 32)
    assert _maybe_symmetrize_window((-1, -1), dynamic_causal) == (-1, -1)
    assert _maybe_symmetrize_window((32, 0), True) == (32, 0)
