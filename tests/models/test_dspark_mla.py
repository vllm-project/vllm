# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_speculator


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    assert set(K3DSparkForCausalLM.checkpoint_skip_substrs) == {
        "confidence_head",
        "embed_tokens",
        "lm_head",
    }


@pytest.mark.cpu_test
def test_dspark_enables_sequence_sharded_target_aux_after_draft_load(monkeypatch):
    class TargetInner:
        def __init__(self):
            self.enable_calls = 0
            self.gather_calls = 0

        def enable_sequence_sharded_raw_aux_hidden_states(self) -> bool:
            self.enable_calls += 1
            return True

        def gather_sequence_sharded_aux_hidden_states(
            self, hidden_states: torch.Tensor
        ) -> torch.Tensor:
            self.gather_calls += 1
            return hidden_states

    target_inner = TargetInner()
    target_model = SimpleNamespace(model=target_inner)
    draft_model = SimpleNamespace(
        combine_hidden_states=Mock(),
        draft_id_to_target_id=None,
    )
    monkeypatch.setattr(
        dspark_speculator,
        "load_dspark_model",
        Mock(return_value=draft_model),
    )

    speculator = object.__new__(dspark_speculator.DSparkSpeculator)
    speculator.vllm_config = SimpleNamespace()
    speculator.draft_logits = None
    speculator._sequence_sharded_aux_gather = None

    loaded_model = speculator.load_draft_model(target_model, set())

    assert loaded_model is draft_model
    assert target_inner.enable_calls == 1
    hidden_states = torch.ones(2, 3)
    assert speculator._sequence_sharded_aux_gather is not None
    torch.testing.assert_close(
        speculator._sequence_sharded_aux_gather(hidden_states),
        hidden_states,
    )
    assert target_inner.gather_calls == 1


@pytest.mark.cpu_test
def test_dspark_projects_local_aux_before_single_gather_and_crops_padding():
    local_aux_0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    local_aux_1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    local_projected = local_aux_0 + local_aux_1
    gathered_with_padding = torch.cat(
        [local_projected, local_projected + 10],
        dim=0,
    )

    combine_hidden_states = Mock(
        side_effect=lambda hidden_states: hidden_states[:, :2] + hidden_states[:, 2:]
    )
    gather = Mock(return_value=gathered_with_padding)
    speculator = object.__new__(dspark_speculator.DSparkSpeculator)
    speculator.model = SimpleNamespace(combine_hidden_states=combine_hidden_states)
    speculator._sequence_sharded_aux_gather = gather

    output = speculator._prepare_target_hidden_states(
        last_hidden_states=torch.full((3, 2), -1.0),
        aux_hidden_states=[local_aux_0, local_aux_1],
        num_target_tokens=3,
    )

    torch.testing.assert_close(output, gathered_with_padding[:3])
    combine_hidden_states.assert_called_once()
    torch.testing.assert_close(
        combine_hidden_states.call_args.args[0],
        torch.cat([local_aux_0, local_aux_1], dim=-1),
    )
    gather.assert_called_once()
    torch.testing.assert_close(gather.call_args.args[0], local_projected)


@pytest.mark.cpu_test
def test_dspark_target_aux_preparation_is_noop_without_aux():
    gather = Mock(side_effect=AssertionError("unexpected gather"))
    combine_hidden_states = Mock(side_effect=AssertionError("unexpected projection"))
    speculator = object.__new__(dspark_speculator.DSparkSpeculator)
    speculator.model = SimpleNamespace(combine_hidden_states=combine_hidden_states)
    speculator._sequence_sharded_aux_gather = gather
    last_hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)

    output = speculator._prepare_target_hidden_states(
        last_hidden_states=last_hidden_states,
        aux_hidden_states=None,
        num_target_tokens=3,
    )

    assert output is last_hidden_states
    combine_hidden_states.assert_not_called()
    gather.assert_not_called()


@pytest.mark.cpu_test
def test_dspark_target_aux_preparation_keeps_unsharded_path():
    aux_hidden_states = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
    ]
    expected = torch.tensor([[4.0, 6.0]])
    combine_hidden_states = Mock(return_value=expected)
    speculator = object.__new__(dspark_speculator.DSparkSpeculator)
    speculator.model = SimpleNamespace(combine_hidden_states=combine_hidden_states)
    speculator._sequence_sharded_aux_gather = None

    output = speculator._prepare_target_hidden_states(
        last_hidden_states=torch.zeros_like(expected),
        aux_hidden_states=aux_hidden_states,
        num_target_tokens=1,
    )

    torch.testing.assert_close(output, expected)
    combine_hidden_states.assert_called_once()


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
