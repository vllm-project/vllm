# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which attention layer DeepSeek-V4.1 picks on SM90, and what the SM90 layer
changes in the FlashMLA layer: query padding and the decode tile scheduler."""

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v41.nvidia import model
from vllm.models.deepseek_v41.nvidia.flashmla import (
    DeepseekSparseSWAFlashMLAMetadataBuilder,
    DeepseekV4FlashMLAAttention,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.attention import (
    DeepseekV41HopperAttention,
    HopperSWAMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _config(backend=None, pcp=1):
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
    )


@pytest.fixture
def sm90(monkeypatch):
    monkeypatch.setattr(model, "is_ampere", lambda: False)
    monkeypatch.setattr(model, "is_hopper", lambda: True)


def test_sm90_takes_the_hopper_layer(sm90):
    assert model._select_dsv4_attn_cls(_config()) is DeepseekV41HopperAttention


@pytest.mark.parametrize(
    "config",
    [
        # An explicit FlashMLA backend keeps the stock decode.
        _config(backend=AttentionBackendEnum.FLASHMLA_SPARSE_DSV41),
        # Prefill context parallelism keeps the stock layer.
        _config(pcp=2),
    ],
)
def test_sm90_keeps_the_flashmla_layer(sm90, config):
    assert model._select_dsv4_attn_cls(config) is DeepseekV4FlashMLAAttention


@pytest.mark.parametrize("heads", [8, 16, 32, 20, 64, 128])
def test_query_padding(heads):
    stock = DeepseekV4FlashMLAAttention.get_padded_num_q_heads(heads)
    got = DeepseekV41HopperAttention.get_padded_num_q_heads(heads)
    assert got == (heads if heads in (8, 16, 32) else stock)


def test_no_decode_tile_scheduler(monkeypatch):
    planned = []

    def build_tile_scheduler(self, num_decode_tokens):
        planned.append(num_decode_tokens)
        return {}

    monkeypatch.setattr(
        DeepseekSparseSWAFlashMLAMetadataBuilder,
        "build_tile_scheduler",
        build_tile_scheduler,
    )
    builder = object.__new__(HopperSWAMetadataBuilder)
    assert builder.build_tile_scheduler(48) == {}
    assert planned == [0]
