# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention import Attention, static_sink_attention
from vllm.model_executor.layers.attention.static_sink_attention import (
    StaticSinkAttention,
)

pytestmark = pytest.mark.cpu_test


def test_static_sink_attention_initializes_custom_op_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_op_init_calls = 0

    def custom_op_init(self, *args, **kwargs) -> None:
        nonlocal custom_op_init_calls
        custom_op_init_calls += 1
        nn.Module.__init__(self)
        self._forward_method = self.forward_native

    def attention_init(self, *args, **kwargs) -> None:
        # Mirror Attention's cooperative super() call through the actual MRO.
        CustomOp.__init__(self)
        self.impl = nn.Identity()
        for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
            self.register_buffer(name, torch.tensor(1.0))

    monkeypatch.setattr(CustomOp, "__init__", custom_op_init)
    monkeypatch.setattr(Attention, "__init__", attention_init)
    monkeypatch.setattr(
        static_sink_attention,
        "create_static_sink_attention_backend",
        lambda *args, **kwargs: object,
    )

    attention = StaticSinkAttention(
        num_heads=1,
        head_size=1,
        scale=1.0,
        sink_len=1,
        attn_backend=object,
    )

    assert custom_op_init_calls == 1
    assert isinstance(attention.impl, nn.Identity)
    assert set(dict(attention.named_buffers())) == {
        "_k_scale",
        "_v_scale",
        "_q_scale",
        "_prob_scale",
    }
