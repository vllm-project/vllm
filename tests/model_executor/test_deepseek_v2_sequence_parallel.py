# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout tracking for MoE sequence parallelism in ``DeepseekV2Model``.

``hidden_states.shape[0]`` cannot tell a full input apart from a padded
sequence-parallel shard: with TP=2 and a single decode token, both have one
row. These tests pin the explicit layout state that replaced the shape check.
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config.compilation import CompilationMode
from vllm.model_executor.models import deepseek_v2 as deepseek_mod

TP_SIZE = 2
HIDDEN_SIZE = 8


def sp_rows(num_tokens: int) -> int:
    """Row count of one shard after padding to a multiple of TP_SIZE."""
    return math.ceil(num_tokens / TP_SIZE)


class DummyPPGroup:
    world_size = 1
    rank_in_group = 0
    is_first_rank = True
    is_last_rank = True


class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, *args, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        return torch.zeros((*input_ids.shape, self.hidden_size))


class DummyNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, hidden_states, residual=None):
        return hidden_states, residual


class RecordingLayer(nn.Module):
    """Records the layout it is handed and reproduces a real layer's shapes.

    A sequence-parallel layer all-gathers its input, runs attention on the
    full sequence and reduce-scatters, so its output always has ``sp_rows``
    rows no matter which layout came in.
    """

    def __init__(self, use_sequence_parallel_moe, calls):
        super().__init__()
        self.use_sequence_parallel_moe = use_sequence_parallel_moe
        self.calls = calls

    def forward(
        self,
        positions,
        hidden_states,
        residual,
        llama_4_scaling=None,
        input_is_sequence_parallel=False,
    ):
        full_num_tokens = positions.shape[0]
        self.calls.append(
            SimpleNamespace(
                input_is_sequence_parallel=input_is_sequence_parallel,
                rows=hidden_states.shape[0],
                expected_rows=(
                    sp_rows(full_num_tokens)
                    if input_is_sequence_parallel
                    else full_num_tokens
                ),
            )
        )
        out_rows = (
            sp_rows(full_num_tokens)
            if self.use_sequence_parallel_moe
            else full_num_tokens
        )
        shape = (out_rows, HIDDEN_SIZE)
        return hidden_states.new_zeros(shape), hidden_states.new_zeros(shape)


def make_vllm_config(num_hidden_layers):
    hf_config = SimpleNamespace(
        model_type="deepseek_v3",
        first_k_dense_replace=0,
        vocab_size=32,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=num_hidden_layers,
        rms_norm_eps=1e-5,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        cache_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


def build_model(monkeypatch, sp_flags, calls, gathered_rows):
    layer_flags = iter(sp_flags)

    def fake_layer(*args, **kwargs):
        return RecordingLayer(next(layer_flags), calls)

    def fake_make_layers(num_hidden_layers, layer_fn, prefix):
        layers = [layer_fn(prefix=f"{prefix}.{i}") for i in range(num_hidden_layers)]
        return 0, num_hidden_layers, nn.ModuleList(layers)

    def fake_all_gather(x, dim):
        gathered_rows.append(x.shape[0])
        return torch.cat([x] * TP_SIZE, dim=dim)

    monkeypatch.setattr(deepseek_mod, "get_pp_group", lambda: DummyPPGroup())
    monkeypatch.setattr(deepseek_mod, "VocabParallelEmbedding", DummyEmbedding)
    monkeypatch.setattr(deepseek_mod, "RMSNorm", DummyNorm)
    monkeypatch.setattr(deepseek_mod, "DeepseekV2DecoderLayer", fake_layer)
    monkeypatch.setattr(deepseek_mod, "make_layers", fake_make_layers)
    monkeypatch.setattr(
        deepseek_mod, "tensor_model_parallel_all_gather", fake_all_gather
    )

    return deepseek_mod.DeepseekV2Model(vllm_config=make_vllm_config(len(sp_flags)))


def run_forward(monkeypatch, num_tokens, sp_flags, aux_layers=()):
    calls: list = []
    gathered_rows: list = []
    model = build_model(monkeypatch, sp_flags, calls, gathered_rows)
    model.aux_hidden_state_layers = aux_layers

    output = model(
        torch.zeros(num_tokens, dtype=torch.long),
        torch.arange(num_tokens),
        None,
    )
    return output, calls, gathered_rows


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_tokens", [1, 2, 7, 8])
@pytest.mark.parametrize(
    "sp_flags",
    [
        # Pure dense stack.
        [False, False],
        # first_k_dense_replace: dense prefix then MoE layers.
        [False, True, True],
        # MoE layers only.
        [True, True, True],
        # moe_layer_freq > 1: a dense layer between MoE layers must restore
        # the full layout.
        [True, False, True],
        # Ends on a dense layer, so no trailing all-gather is needed.
        [True, True, False],
    ],
)
def test_layer_inputs_match_the_declared_layout(monkeypatch, num_tokens, sp_flags):
    output, calls, _ = run_forward(monkeypatch, num_tokens, sp_flags)

    assert len(calls) == len(sp_flags)
    for idx, call in enumerate(calls):
        assert call.rows == call.expected_rows, (
            f"layer {idx} was told input_is_sequence_parallel="
            f"{call.input_is_sequence_parallel} but received {call.rows} rows"
        )
    # The model must hand back the full sequence.
    assert output.shape == (num_tokens, HIDDEN_SIZE)


@pytest.mark.cpu_test
def test_single_token_decode_keeps_tracking_sequence_parallel_layout(monkeypatch):
    """Regression test for the shape-inference bug.

    With one token and TP=2 every tensor has a single row, so the previous
    ``hidden_states.shape[0] != full_num_tokens`` check reported "not
    sequence parallel" for shards and skipped the all-gathers.
    """
    output, calls, gathered_rows = run_forward(
        monkeypatch, num_tokens=1, sp_flags=[False, True, True]
    )

    assert all(call.rows == 1 for call in calls), (
        "shapes are ambiguous here, which is the whole point of the test"
    )
    # The dense layer and the first MoE layer see the full input; only the
    # second MoE layer receives a shard produced by the first one.
    assert [call.input_is_sequence_parallel for call in calls] == [False, False, True]
    # The trailing shard must still be gathered before the final norm.
    assert gathered_rows == [1]
    assert output.shape == (1, HIDDEN_SIZE)


@pytest.mark.cpu_test
def test_dense_layer_after_moe_restores_full_layout(monkeypatch):
    _, calls, gathered_rows = run_forward(
        monkeypatch, num_tokens=1, sp_flags=[True, False, True]
    )

    assert [call.input_is_sequence_parallel for call in calls] == [False, False, False]
    # One gather before the dense layer, one after the trailing MoE layer.
    assert gathered_rows == [1, 1]


@pytest.mark.cpu_test
@pytest.mark.parametrize("num_tokens", [1, 8])
def test_aux_hidden_states_are_gathered(monkeypatch, num_tokens):
    """EAGLE aux hidden states must be full-length even when captured
    between two sequence-parallel layers."""
    _, _, gathered_rows = run_forward(
        monkeypatch,
        num_tokens,
        sp_flags=[True, True, True],
        aux_layers=(1,),
    )

    # Layer 1 is fed a shard, so its aux capture needs its own all-gather on
    # top of the trailing one.
    assert gathered_rows == [sp_rows(num_tokens), sp_rows(num_tokens)]
