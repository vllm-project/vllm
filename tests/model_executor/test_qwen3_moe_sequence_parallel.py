# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, local
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.models import qwen3_moe as model


class _TPCollectives:
    """CPU collectives with distinct data from every simulated TP rank."""

    def __init__(self, size):
        self.size = size
        self.state = local()
        self.barrier = Barrier(size, timeout=15)
        self.inputs = [None] * size

    def chunk(self, tensor):
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, -len(tensor) % self.size))
        return tensor.chunk(self.size, dim=0)[self.state.rank]

    def collective(self, tensor, operation):
        self.inputs[self.state.rank] = tensor
        self.barrier.wait()
        if operation == "gather":
            result = torch.cat(self.inputs, dim=0)
        else:
            result = torch.stack(self.inputs).sum(dim=0)
            if operation == "scatter":
                result = result.chunk(self.size, dim=0)[self.state.rank]
        self.barrier.wait()
        return result

    def all_gather(self, tensor, dim):
        assert dim == 0
        return self.collective(tensor, "gather")

    def reduce_scatter(self, tensor, dim):
        assert dim == 0
        return self.collective(tensor, "scatter")


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_counts = []

    def forward(self, hidden_states, residual=None):
        self.token_counts.append(len(hidden_states))
        if residual is not None:
            assert hidden_states.shape == residual.shape
            hidden_states = hidden_states + residual
        output = hidden_states * torch.rsqrt(
            hidden_states.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return output if residual is None else (output, hidden_states)


class _Attention(nn.Module):
    def __init__(self, world, reduce_results):
        super().__init__()
        self.world = world
        self.reduce_results = reduce_results
        self.token_counts = []

    def forward(self, positions, hidden_states):
        assert len(hidden_states) == positions.shape[-1]
        self.token_counts.append(len(hidden_states))
        # Include cross-token dependence and unequal TP partial contributions.
        partial = (self.world.state.rank + 1) * (
            hidden_states + hidden_states.mean(dim=0) / 4
        )
        if self.reduce_results:
            return self.world.collective(partial, "reduce")
        return partial


class _Experts(nn.Module):
    def forward(self, hidden_states, router_logits):
        return hidden_states.tanh()


def _make_model(world, use_sp, collect_aux):
    result = model.Qwen3MoeModel.__new__(model.Qwen3MoeModel)
    nn.Module.__init__(result)
    result.start_layer, result.end_layer = 0, 2
    result.aux_hidden_state_layers = (0, 1, 2) if collect_aux else ()
    result.norm = _Norm()
    result.layers = nn.ModuleList()
    for _ in range(result.end_layer):
        layer = model.Qwen3MoeDecoderLayer.__new__(model.Qwen3MoeDecoderLayer)
        nn.Module.__init__(layer)
        layer.use_attn_reduce_scatter_for_moe = use_sp
        layer.input_layernorm = _Norm()
        layer.post_attention_layernorm = _Norm()
        layer.self_attn = _Attention(world, reduce_results=not use_sp)
        layer.mlp = model.Qwen3MoeSparseMoeBlock.__new__(model.Qwen3MoeSparseMoeBlock)
        nn.Module.__init__(layer.mlp)
        layer.mlp.is_sequence_parallel = True
        layer.mlp.experts = _Experts()
        result.layers.append(layer)
    return result


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("num_tokens", [1, 3, 16])
@pytest.mark.parametrize("collect_aux", [False, True])
def test_sp_preserves_outputs_and_shards_all_norms(
    monkeypatch, tp_size, num_tokens, collect_aux
):
    world = _TPCollectives(tp_size)
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(model, "sequence_parallel_chunk", world.chunk)
    monkeypatch.setattr(model, "tensor_model_parallel_all_gather", world.all_gather)
    monkeypatch.setattr(
        model, "tensor_model_parallel_reduce_scatter", world.reduce_scatter
    )
    monkeypatch.setattr(
        model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    inputs = (
        torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4) / 7
    )

    def run_rank(rank, use_sp):
        world.state.rank = rank
        instance = _make_model(world, use_sp, collect_aux)
        output = instance.forward(None, torch.arange(num_tokens), inputs_embeds=inputs)
        expected_tokens = (
            (num_tokens + tp_size - 1) // tp_size if use_sp else num_tokens
        )
        for layer in instance.layers:
            assert layer.input_layernorm.token_counts == [expected_tokens]
            assert layer.post_attention_layernorm.token_counts == [expected_tokens]
            assert layer.self_attn.token_counts == [num_tokens]
        assert instance.norm.token_counts == [expected_tokens]
        return output

    with ThreadPoolExecutor(max_workers=tp_size) as pool:
        reference = list(pool.map(lambda rank: run_rank(rank, False), range(tp_size)))
        actual = list(pool.map(lambda rank: run_rank(rank, True), range(tp_size)))
    for expected, output in zip(reference, actual):
        torch.testing.assert_close(output, expected)
        if collect_aux:
            output, auxiliary = output
            assert all(tensor.shape == inputs.shape for tensor in auxiliary)
        assert output.shape == inputs.shape


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_text_config=SimpleNamespace(
                num_experts=8,
                mlp_only_layers=[],
                decoder_sparse_step=1,
                hidden_size=4,
                num_attention_heads=2,
                num_key_value_heads=2,
                rope_parameters={},
                rms_norm_eps=1e-6,
                moe_intermediate_size=8,
                intermediate_size=8,
                hidden_act="silu",
            ),
        ),
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=True, pipeline_parallel_size=1
        ),
        cache_config=None,
        quant_config=None,
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("parallel_config", "use_sequence_parallel_moe", False),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("model_config", "is_multimodal_model", True),
        ("hf_text_config", "num_experts", 0),
        ("hf_text_config", "mlp_only_layers", [1]),
        ("hf_text_config", "decoder_sparse_step", 2),
    ],
)
def test_sp_keeps_unsupported_layouts_on_existing_path(section, key, value):
    config = _config()
    assert model._should_use_sequence_parallel(config)
    target = (
        config.model_config.hf_text_config
        if section == "hf_text_config"
        else getattr(config, section)
    )
    setattr(target, key, value)
    assert not model._should_use_sequence_parallel(config)


@pytest.mark.parametrize("use_sp", [False, True])
def test_decoder_disables_oproj_allreduce_only_with_sp(monkeypatch, use_sp):
    config = _config()
    config.parallel_config.use_sequence_parallel_moe = use_sp
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 2)
    row_linear = Mock(side_effect=lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(model, "RowParallelLinear", row_linear)
    for name in (
        "QKVParallelLinear",
        "Attention",
        "RMSNorm",
        "get_rope",
        "Qwen3MoeSparseMoeBlock",
    ):
        monkeypatch.setattr(model, name, lambda *args, **kwargs: nn.Identity())
    model.Qwen3MoeDecoderLayer(config, prefix="model.layers.0")
    assert row_linear.call_args.kwargs["reduce_results"] is (not use_sp)


def test_decoder_subclass_without_base_init_retains_replicated_layout():
    # Mellum inherits forward but initializes its attention independently.
    class Decoder(model.Qwen3MoeDecoderLayer):
        def __init__(self):
            nn.Module.__init__(self)

    assert not Decoder().use_attn_reduce_scatter_for_moe
