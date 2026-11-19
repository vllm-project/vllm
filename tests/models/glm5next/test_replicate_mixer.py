# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.x-Next mixers kept whole on every TP rank (VLLM_GLM5NEXT_REPLICATE_MIXER).

A replicated mixer holds all heads, the full weights and the full recurrent state
on each rank, and its output projection is complete, so it must never all-reduce.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Glm5NextTextConfig

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor import parameter
from vllm.model_executor.layers import canonical_topk, linear
from vllm.model_executor.layers.mamba.gdn import base as gdn_base
from vllm.model_executor.model_loader import weight_utils
from vllm.models.glm5next.common import model as glm_model
from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.triton_utils import tp_synced_autotune

HIDDEN, HEADS, HEAD_DIM, TP_SIZE = 32, 4, 16, 2
PROJ = HEADS * HEAD_DIM


@pytest.mark.parametrize(
    ("value", "kinds"),
    [
        ("", set()),
        ("0", set()),
        ("off", set()),
        ("1", {"kda", "mla"}),
        ("all", {"kda", "mla"}),
        ("ALL", {"kda", "mla"}),
        ("mla", {"mla"}),
        ("kda", {"kda"}),
        ("mla, kda", {"kda", "mla"}),
    ],
)
def test_replicated_mixer_kinds(monkeypatch, value: str, kinds: set[str]):
    monkeypatch.setenv("VLLM_GLM5NEXT_REPLICATE_MIXER", value)
    assert glm_model._replicated_mixer_kinds() == kinds


@pytest.mark.parametrize("value", ["attention", "kda,moe"])
def test_replicated_mixer_kinds_rejects_unknown(monkeypatch, value: str):
    monkeypatch.setenv("VLLM_GLM5NEXT_REPLICATE_MIXER", value)
    with pytest.raises(ValueError, match="use all, mla or kda"):
        glm_model._replicated_mixer_kinds()


def _fake_tensor_parallel_world(monkeypatch) -> None:
    """Pretend to be rank 1 of a TP=2 group whose all-reduce must never run."""

    def _forbidden_all_reduce(*args, **kwargs):
        raise AssertionError("tensor-parallel all-reduce after a replicated mixer")

    for module in (linear, parameter, gdn_base, weight_utils):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: TP_SIZE
        )
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        linear, "tensor_model_parallel_all_reduce", _forbidden_all_reduce
    )


class _Mixer(nn.Module):
    """Stands in for both mixers and records the `replicate` it was built with."""

    def __init__(self, *args, replicate: bool, **kwargs) -> None:
        super().__init__()
        self.replicate = replicate
        self.o_proj = SimpleNamespace(reduce_results=True)


def _decoder_layers(monkeypatch, value: str) -> list[glm_model.Glm5NextDecoderLayer]:
    _fake_tensor_parallel_world(monkeypatch)
    monkeypatch.setenv("VLLM_GLM5NEXT_REPLICATE_MIXER", value)
    monkeypatch.setattr(glm_model, "Glm5NextLinearAttention", _Mixer)
    monkeypatch.setattr(glm_model, "Glm5NextMLAAttention", _Mixer)
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=2,
        first_k_dense_replace=2,
        n_routed_experts=4,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mhc=False,
        mla_use_nope=False,
    )
    vllm_config = SimpleNamespace(
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        kernel_config=SimpleNamespace(enable_jit_warmup=False),
    )
    return [
        glm_model.Glm5NextDecoderLayer(vllm_config, config, layer_idx=i)
        for i in range(2)
    ]


@pytest.mark.parametrize(
    ("value", "kda", "mla"),
    [
        ("0", False, False),
        ("kda", True, False),
        ("mla", False, True),
        ("all", True, True),
    ],
)
def test_decoder_layer_builds_replicated_mixers(
    monkeypatch, default_vllm_config, value: str, kda: bool, mla: bool
):
    # Both hooks are process-wide: record them instead of installing them.
    installs: list[int] = []
    topk_reasons: list[str] = []
    monkeypatch.setattr(tp_synced_autotune, "install", lambda: installs.append(1))
    monkeypatch.setattr(
        canonical_topk, "require_deterministic_sparse_topk", topk_reasons.append
    )

    kda_layer, mla_layer = _decoder_layers(monkeypatch, value)

    assert (kda_layer.layer_kind, mla_layer.layer_kind) == ("kda", "mla")
    assert (kda_layer.self_attn.replicate, mla_layer.self_attn.replicate) == (kda, mla)
    assert len(installs) == kda + mla
    assert topk_reasons == (["VLLM_GLM5NEXT_REPLICATE_MIXER"] if mla else [])


@pytest.mark.parametrize(
    "parallel",
    [
        {"enable_expert_parallel": True},
        {"decode_context_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"use_sequence_parallel_moe": True},
    ],
)
def test_replicated_mixer_refuses_ep_sp_and_cp(
    monkeypatch, default_vllm_config, parallel: dict
):
    monkeypatch.setattr(tp_synced_autotune, "install", lambda: None)
    current = dict(
        enable_expert_parallel=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    current.update((k, v) for k, v in parallel.items() if k in current)
    monkeypatch.setattr(
        glm_model,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(**current)),
    )
    vllm_config = SimpleNamespace(
        cache_config=None,
        quant_config=None,
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=parallel.get("use_sequence_parallel_moe", False)
        ),
    )
    monkeypatch.setenv("VLLM_GLM5NEXT_REPLICATE_MIXER", "kda")
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        mhc=False,
    )
    with pytest.raises(NotImplementedError, match="REPLICATE_MIXER"):
        glm_model.Glm5NextDecoderLayer(vllm_config, config, layer_idx=0)


def _kda_layer(replicate: bool) -> Glm5NextLinearAttention:
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        linear_num_heads=HEADS,
        linear_head_dim=HEAD_DIM,
        num_hidden_layers=1,
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    with set_current_vllm_config(vllm_config):
        return Glm5NextLinearAttention(
            config, vllm_config, prefix="model.layers.0.self_attn", replicate=replicate
        )


def test_sharded_kda_follows_the_fake_tp_group(monkeypatch, dist_init):
    # Control arm: without replication rank 1 holds the second half of the heads.
    _fake_tensor_parallel_world(monkeypatch)
    layer = _kda_layer(replicate=False)

    local_heads = HEADS // TP_SIZE
    assert (layer.tp_size, layer.tp_rank) == (TP_SIZE, 1)
    assert layer.A_log.shape == (1, 1, local_heads, 1)
    assert layer.dt_bias.shape == (PROJ // TP_SIZE,)
    assert layer.o_proj.weight.shape == (HIDDEN, PROJ // TP_SIZE)
    assert layer.get_state_shape()[1] == (local_heads, HEAD_DIM, HEAD_DIM)
    a_log = torch.arange(HEADS, dtype=torch.float32)
    layer.A_log.weight_loader(layer.A_log, a_log)
    assert torch.equal(layer.A_log.flatten(), a_log[local_heads:])


def test_replicated_kda_holds_every_head_on_rank_1(monkeypatch, dist_init):
    _fake_tensor_parallel_world(monkeypatch)
    layer = _kda_layer(replicate=True)

    assert (layer.tp_size, layer.tp_rank) == (1, 0)
    assert layer.local_num_heads == HEADS
    assert layer.A_log.shape == (1, 1, HEADS, 1)
    assert layer.dt_bias.shape == (PROJ,)
    # q|k|v|b|f_a|g_a, all full width.
    assert layer.in_proj_qkvbfg_a.weight.shape == (
        3 * PROJ + HEADS + 2 * HEAD_DIM,
        HIDDEN,
    )
    assert layer.f_b_proj.weight.shape == (PROJ, HEAD_DIM)
    assert layer.g_b_proj.weight.shape == (PROJ, HEAD_DIM)
    assert layer.conv1d.weight.shape[0] == 3 * PROJ
    assert layer.o_proj.weight.shape == (HIDDEN, PROJ)

    conv_shape, recurrent_shape = layer.get_state_shape()
    assert recurrent_shape == (HEADS, HEAD_DIM, HEAD_DIM)
    assert 3 * PROJ in conv_shape

    # Checkpoint tensors load whole, not as rank 1's slice.
    a_log = torch.arange(HEADS, dtype=torch.float32)
    layer.A_log.weight_loader(layer.A_log, a_log)
    assert torch.equal(layer.A_log.flatten(), a_log)
    dt_bias = torch.arange(PROJ, dtype=torch.float32)
    layer.dt_bias.weight_loader(layer.dt_bias, dt_bias)
    assert torch.equal(layer.dt_bias, dt_bias)
    o_proj = torch.randn(HIDDEN, PROJ, generator=torch.Generator().manual_seed(0))
    layer.o_proj.weight_loader(layer.o_proj.weight, o_proj)
    assert torch.equal(layer.o_proj.weight, o_proj.to(layer.o_proj.weight.dtype))

    # The output projection is complete on its own: no all-reduce.
    x = torch.randn(3, PROJ, generator=torch.Generator().manual_seed(1))
    x = x.to(layer.o_proj.weight.dtype)
    out, _ = layer.o_proj(x)
    torch.testing.assert_close(out, x @ layer.o_proj.weight.T)


def test_replicated_kda_state_shape_from_config(monkeypatch):
    hf_config = SimpleNamespace(
        linear_num_heads=HEADS, linear_head_dim=HEAD_DIM, linear_conv_kernel_dim=4
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=TP_SIZE),
        model_config=SimpleNamespace(hf_text_config=hf_config),
        speculative_config=None,
    )
    shape = glm_model.Glm5NextForCausalLM.get_mamba_state_shape_from_config
    for value, heads in (("mla", HEADS // TP_SIZE), ("kda", HEADS)):
        monkeypatch.setenv("VLLM_GLM5NEXT_REPLICATE_MIXER", value)
        assert shape(vllm_config)[1] == (heads, HEAD_DIM, HEAD_DIM)
