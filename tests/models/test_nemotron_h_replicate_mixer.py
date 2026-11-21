# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-H with VLLM_NEMOTRON_H_REPLICATE_MIXER under a fake TP=2.

Rank 1 must build every Mamba2 / attention mixer whole (all heads, no
all-reduce) and size its Mamba state and attention KV cache for all heads.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import NemotronHConfig

from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.mamba import mamba_mixer2
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.models import nemotron_h
from vllm.platforms import interface

TP_SIZE = 2
HIDDEN, HEADS, HEAD_DIM, GROUPS, STATE, CONV = 16, 4, 4, 2, 8, 4
INTERMEDIATE = HEADS * HEAD_DIM
CONV_DIM = INTERMEDIATE + 2 * GROUPS * STATE
ATTN_HEADS, KV_HEADS, ATTN_HEAD_DIM = 4, 2, 4
ENV = "VLLM_NEMOTRON_H_REPLICATE_MIXER"


def _tiny_config() -> NemotronHConfig:
    return NemotronHConfig(
        vocab_size=32,
        hidden_size=HIDDEN,
        mamba_num_heads=HEADS,
        mamba_head_dim=HEAD_DIM,
        n_groups=GROUPS,
        ssm_state_size=STATE,
        conv_kernel=CONV,
        num_attention_heads=ATTN_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=ATTN_HEAD_DIM,
    )


class _Attention(nn.Module):
    """Records what the attention mixer asks the KV cache to hold."""

    def __init__(self, num_heads, head_size, scale, num_kv_heads, **kwargs):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads


@pytest.fixture
def fake_tp_rank1(monkeypatch, default_vllm_config):
    """Pretend to be rank 1 of a TP=2 group whose all-reduce must never run."""

    def _forbidden_all_reduce(*args, **kwargs):
        raise AssertionError("tensor-parallel all-reduce in a replicated mixer")

    for module in (linear, parameter, mamba_mixer2, weight_utils):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: TP_SIZE
        )
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        nemotron_h, "get_tensor_model_parallel_world_size", lambda: TP_SIZE
    )
    monkeypatch.setattr(
        linear, "tensor_model_parallel_all_reduce", _forbidden_all_reduce
    )
    monkeypatch.setattr(nemotron_h, "Attention", _Attention)


def _mamba_layer(monkeypatch, replicate: bool, prefix: str):
    monkeypatch.setenv(ENV, "1" if replicate else "0")
    return nemotron_h.NemotronHMambaDecoderLayer(_tiny_config(), 0, prefix=prefix)


def _state_shape_from_config(monkeypatch, replicate: bool):
    monkeypatch.setenv(ENV, "1" if replicate else "0")
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=TP_SIZE),
        cache_config=SimpleNamespace(use_replayssm=False),
        model_config=SimpleNamespace(hf_config=_tiny_config()),
        num_speculative_tokens=0,
    )
    return nemotron_h.NemotronHForCausalLM.get_mamba_state_shape_from_config(
        vllm_config
    )


def _full_state_shape(tp_world_size: int):
    return MambaStateShapeCalculator.mamba2_state_shape(
        tp_world_size=tp_world_size,
        intermediate_size=INTERMEDIATE,
        n_groups=GROUPS,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        state_size=STATE,
        conv_kernel=CONV,
    )


def test_replicated_mamba_mixer_holds_all_heads(monkeypatch, fake_tp_rank1):
    # Control: without the flag, rank 1 holds half of every head-sharded tensor.
    sharded = _mamba_layer(monkeypatch, False, "sharded").mixer
    assert sharded.A.shape == (HEADS // TP_SIZE,)
    assert sharded.get_state_shape() == _full_state_shape(TP_SIZE)

    mixer = _mamba_layer(monkeypatch, True, "replicated").mixer
    assert mixer.replicate and mixer.tp_size == 1
    assert mixer.in_proj.weight.shape == (
        2 * INTERMEDIATE + 2 * GROUPS * STATE + HEADS,
        HIDDEN,
    )
    assert mixer.conv1d.weight.shape == (CONV_DIM, 1, CONV)
    assert mixer.out_proj.weight.shape == (HIDDEN, INTERMEDIATE)
    assert mixer.norm.weight.shape == (INTERMEDIATE,)
    for name in ("A", "D", "dt_bias"):
        assert getattr(mixer, name).shape == (HEADS,), name

    # Rank 1 must load the whole checkpoint tensor, not narrow to its shard.
    full = torch.arange(1, HEADS + 1, dtype=torch.float32)
    for name in ("D", "dt_bias"):
        getattr(mixer, name).weight_loader(getattr(mixer, name), full)
        torch.testing.assert_close(getattr(mixer, name).data, full)
    mixer.A.weight_loader(mixer.A, full)
    torch.testing.assert_close(mixer.A.data, -torch.exp(full))

    # Mixer state and the cache NemotronHForCausalLM allocates: all heads.
    assert mixer.get_state_shape() == _full_state_shape(1)
    assert mixer.get_state_shape()[1] == (HEADS, HEAD_DIM, STATE)
    assert _state_shape_from_config(monkeypatch, True) == mixer.get_state_shape()
    assert _state_shape_from_config(monkeypatch, False) == sharded.get_state_shape()


def test_replicated_out_projections_do_not_reduce(monkeypatch, fake_tp_rank1):
    x = torch.randn(3, INTERMEDIATE)
    sharded = _mamba_layer(monkeypatch, False, "sharded").mixer
    with pytest.raises(AssertionError, match="all-reduce"):
        sharded.out_proj(x[:, : INTERMEDIATE // TP_SIZE])

    mixer = _mamba_layer(monkeypatch, True, "replicated").mixer
    mixer.out_proj.weight.data.normal_()
    out, _ = mixer.out_proj(x)
    torch.testing.assert_close(out, x @ mixer.out_proj.weight.T)

    attn = nemotron_h.NemotronHAttentionDecoderLayer(
        _tiny_config(), 0, prefix="attn"
    ).mixer
    attn.o_proj.weight.data.normal_()
    y = torch.randn(3, ATTN_HEADS * ATTN_HEAD_DIM)
    out, _ = attn.o_proj(y)
    torch.testing.assert_close(out, y @ attn.o_proj.weight.T)


def test_replicated_attention_caches_all_kv_heads(monkeypatch, fake_tp_rank1):
    monkeypatch.setenv(ENV, "0")
    sharded = nemotron_h.NemotronHAttentionDecoderLayer(_tiny_config(), 0).mixer
    assert sharded.attn.num_kv_heads == KV_HEADS // TP_SIZE

    monkeypatch.setenv(ENV, "1")
    attn = nemotron_h.NemotronHAttentionDecoderLayer(_tiny_config(), 0).mixer
    assert (attn.num_heads, attn.num_kv_heads) == (ATTN_HEADS, KV_HEADS)
    assert (attn.attn.num_heads, attn.attn.num_kv_heads) == (ATTN_HEADS, KV_HEADS)
    qkv_rows = (ATTN_HEADS + 2 * KV_HEADS) * ATTN_HEAD_DIM
    assert attn.qkv_proj.weight.shape == (qkv_rows, HIDDEN)

    # The platform's KV page sizing must agree with the layer, for Nemotron-H only.
    def vllm_config(model_type: str):
        model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type),
            get_total_num_kv_heads=lambda: KV_HEADS,
            get_num_kv_heads=lambda parallel_config: KV_HEADS // TP_SIZE,
        )
        return SimpleNamespace(model_config=model_config, parallel_config=None)

    assert interface._attn_kv_heads_per_rank(vllm_config("nemotron_h")) == KV_HEADS
    assert interface._attn_kv_heads_per_rank(vllm_config("llama")) == 1
    monkeypatch.setenv(ENV, "0")
    assert interface._attn_kv_heads_per_rank(vllm_config("nemotron_h")) == 1


@pytest.mark.parametrize(
    "parallel",
    [
        {"enable_expert_parallel": True},
        {"use_sequence_parallel_moe": True},
        {"decode_context_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
    ],
)
def test_replication_refuses_ep_sp_cp(monkeypatch, parallel):
    monkeypatch.setenv(ENV, "1")
    parallel_config = SimpleNamespace(
        **{
            "enable_expert_parallel": False,
            "use_sequence_parallel_moe": False,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            **parallel,
        }
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=_tiny_config()),
        cache_config=None,
        quant_config=None,
        parallel_config=parallel_config,
    )
    with pytest.raises(NotImplementedError, match=ENV):
        nemotron_h.NemotronHModel(vllm_config=vllm_config)
