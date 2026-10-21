# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

from vllm.config import (
    AttentionConfig,
    DeviceConfig,
    KernelConfig,
    LoadConfig,
    ModelConfig,
    VllmConfig,
)
from vllm.model_executor.layers.fused_moe.expert_substitution import (
    ConstantExpertSubstitution,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.expert_substitution import as_expert_substitution_model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MODEL_CONFIG = SimpleNamespace(
    hf_config=SimpleNamespace(approximate_experts={"0": [1, 2], "1": [1, 2]})
)


def _compression_config(value_names: dict[int, dict[int, str]]) -> dict:
    """Explicitly named constants given as ``{layer: {expert: tensor name}}``."""
    targets = {
        f"model.layers.{layer}.mlp.experts": {
            "num_logical_experts": 3,
            "weight_layout": "compact_retained_experts",
            "replacements": {
                str(expert): {"format": "constant-v1", "tensors": {"value": name}}
                for expert, name in names.items()
            },
        }
        for layer, names in value_names.items()
    }
    router_semantics = {
        "preserve_logical_expert_ids": True,
        "preserve_router_weights": True,
        "renormalize_after_substitution": False,
    }
    return {
        "transform_config": {
            "expert_substitution": {
                "version": 1,
                "router_semantics": router_semantics,
                "targets": targets,
            }
        }
    }


# Expert 1 of both layers shares one explicitly named constant.
EXPLICIT_MODEL_CONFIG = SimpleNamespace(
    hf_config=SimpleNamespace(
        compression_config=_compression_config(
            {
                layer: {1: "constants.shared", 2: f"constants.layer_{layer}"}
                for layer in (0, 1)
            }
        )
    )
)


def _approx_value(layer: int, expert: int) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.approx_value"


def _cpu_model(local_layers=(0, 1), tracked=True, model_config=MODEL_CONFIG):
    """Two decoder layers; the others are placeholders as on another PP stage."""

    class Model(torch.nn.Module):
        def __init__(self, prefix=""):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            for index in range(2):
                if index not in local_layers:
                    self.model.layers.append(PPMissingLayer())
                    continue
                layer = torch.nn.Module()
                layer.mlp = torch.nn.Module()
                layer.mlp.experts = ConstantExpertSubstitution(
                    index, 3, (1, 2), 2, torch.float32
                )
                self.model.layers.append(layer)

        def load_weights(self, weights):
            loaded = AutoWeightsLoader(self).load_weights(weights)
            return loaded if tracked else None

    return as_expert_substitution_model(Model, model_config)()


@pytest.mark.parametrize("local_layers", [(0, 1), (0,), (1,), ()])
def test_loads_local_approx_values_and_passes_through_others(local_layers):
    """Other PP stages and MTP drafts own none or some of the configured layers."""
    model = _cpu_model(local_layers)
    loaded = model.load_weights(
        [
            ("weight", torch.tensor([1.0, 2.0])),
            *(
                (_approx_value(layer, expert), torch.full((2,), 10.0 * layer + expert))
                for layer in range(2)
                for expert in (1, 2)
            ),
        ]
    )
    model.process_weights_after_loading()

    assert loaded == {"weight"} | {
        f"model.layers.{layer}.mlp.experts.values" for layer in local_layers
    }
    for layer in local_layers:
        torch.testing.assert_close(
            model.model.layers[layer].mlp.experts.values,
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]) + 10.0 * layer,
        )


@pytest.mark.parametrize("local_layers", [(0, 1), (1,), ()])
def test_loads_explicitly_named_values(local_layers):
    """Named values are consumed even when their layer is built elsewhere."""
    model = _cpu_model(local_layers, model_config=EXPLICIT_MODEL_CONFIG)
    loaded = model.load_weights(
        [
            ("weight", torch.tensor([1.0, 2.0])),
            ("constants.shared", torch.full((2,), 1.0)),
            ("constants.layer_0", torch.full((2,), 2.0)),
            ("constants.layer_1", torch.full((2,), 3.0)),
        ]
    )
    model.process_weights_after_loading()

    assert loaded == {"weight"} | {
        f"model.layers.{layer}.mlp.experts.values" for layer in local_layers
    }
    for layer in local_layers:
        torch.testing.assert_close(
            model.model.layers[layer].mlp.experts.values,
            torch.tensor([[1.0, 1.0], [2.0 + layer, 2.0 + layer]]),
        )


def test_direct_reload_updates_values_incrementally():
    model = _cpu_model(local_layers=(0,), tracked=False)
    values = model.model.layers[0].mlp.experts.values
    model.load_weights([(_approx_value(0, 2), torch.ones(2))])
    with pytest.raises(ValueError, match="missing approx_value .* \\[1\\]"):
        model.process_weights_after_loading()

    assert model.load_weights([(_approx_value(0, 1), torch.zeros(2))]) is None
    model.process_weights_after_loading()
    model.load_weights([(_approx_value(0, 2), torch.full((2,), 3.0))])
    torch.testing.assert_close(values, torch.tensor([[0.0, 0.0], [3.0, 3.0]]))

    with pytest.raises(ValueError, match="not listed"):
        model.load_weights([(_approx_value(0, 0), torch.ones(2))])


def test_layerwise_reload_is_rejected_and_keeps_weights():
    from vllm.model_executor.model_loader.reload import (
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    model = _cpu_model(local_layers=(0,))
    record_metadata_for_reloading(model)
    loaded = [
        ("weight", torch.tensor([3.0, 4.0])),
        (_approx_value(0, 1), torch.zeros(2)),
        (_approx_value(0, 2), torch.ones(2)),
    ]
    model.load_weights(loaded)
    model.process_weights_after_loading()
    substitution = model.model.layers[0].mlp.experts
    values = substitution.values

    initialize_layerwise_reload(model)
    with pytest.raises(NotImplementedError, match="layerwise weight reload"):
        model.load_weights([(name, weight + 1) for name, weight in loaded])

    assert substitution.values is values
    torch.testing.assert_close(values, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    torch.testing.assert_close(model.weight, torch.tensor([3.0, 4.0]))
    assert substitution.expert_substitution_routes.tolist() == [0, -1, -2]


@pytest.mark.parametrize("missing_value", [False, True])
def test_tensorizer_finalizes_local_substitution_values(monkeypatch, missing_value):
    """Raw Tensorizer loading must reject an unloaded local constant row."""
    from vllm.model_executor.model_loader import tensorizer_loader

    model = _cpu_model(local_layers=(0,))
    weights = [(_approx_value(0, 1), torch.zeros(2))]
    if not missing_value:
        weights.append((_approx_value(0, 2), torch.ones(2)))
    finalize = Mock(wraps=model.process_weights_after_loading)
    monkeypatch.setattr(model, "process_weights_after_loading", finalize)
    monkeypatch.setattr(tensorizer_loader, "initialize_model", lambda **kwargs: model)
    loader = object.__new__(tensorizer_loader.TensorizerLoader)
    monkeypatch.setattr(loader, "_get_weights_iterator", lambda: iter(weights))
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )

    if missing_value:
        with pytest.raises(ValueError, match=r"missing approx_value .* \[2\]"):
            loader._load_model_serialized_cpu(config)
    else:
        assert loader._load_model_serialized_cpu(config) is model
        torch.testing.assert_close(
            model.model.layers[0].mlp.experts.values,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        )
    finalize.assert_called_once_with()


def _write_tiny_deepseek_mone_checkpoint(
    model_dir: Path, explicit_names: bool
) -> tuple[dict[str, torch.Tensor], str]:
    config = DeepseekV2Config(
        architectures=["DeepseekV2ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        first_k_dense_replace=0,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        norm_topk_prob=False,
        q_lora_rank=None,
        kv_lora_rank=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        tie_word_embeddings=False,
        dtype="float16",
    )
    experts = "model.layers.0.mlp.experts"
    if explicit_names:
        value_name = "model.layers.0.mlp.expert_replacements.1.value"
        config.compression_config = _compression_config({0: {1: value_name}})
    else:
        value_name = f"{experts}.1.approx_value"
        config.approximate_experts = {"0": [1]}

    hf_weights = DeepseekV2ForCausalLM(config).to(torch.float16).state_dict()
    gate_up = hf_weights.pop(f"{experts}.gate_up_proj")
    down = hf_weights.pop(f"{experts}.down_proj")
    weights = {name: value.contiguous() for name, value in hf_weights.items()}
    for expert in (0, 2):
        gate, up = gate_up[expert].chunk(2, dim=0)
        weights[f"{experts}.{expert}.gate_proj.weight"] = gate.contiguous()
        weights[f"{experts}.{expert}.up_proj.weight"] = up.contiguous()
        weights[f"{experts}.{expert}.down_proj.weight"] = down[expert].contiguous()
    weights[value_name] = torch.arange(16, dtype=torch.float16)
    config.save_pretrained(model_dir)
    save_file(weights, model_dir / "model.safetensors")
    return weights, value_name


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
@pytest.mark.parametrize(
    "explicit_names", [False, True], ids=["approximate_experts", "expert_substitution"]
)
def test_loads_mone_checkpoint_into_compact_experts(
    tmp_path: Path, explicit_names: bool
):
    weights, value_name = _write_tiny_deepseek_mone_checkpoint(tmp_path, explicit_names)
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=32,
        enforce_eager=True,
    )
    load_config = LoadConfig(load_format="safetensors", use_tqdm_on_load=False)
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cuda"),
        load_config=load_config,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_MLA),
        kernel_config=KernelConfig(moe_backend="triton"),
    )
    model = get_model_loader(load_config).load_model(vllm_config, model_config)

    experts = "model.layers.0.mlp.experts"
    routed_experts = model.model.layers[0].mlp.experts.routed_experts
    # Logical expert 2 is stored in physical row 1.
    torch.testing.assert_close(
        routed_experts.w13_weight[1].cpu(),
        torch.cat(
            [
                weights[f"{experts}.2.gate_proj.weight"],
                weights[f"{experts}.2.up_proj.weight"],
            ]
        ),
    )
    torch.testing.assert_close(
        routed_experts.w2_weight[1].cpu(), weights[f"{experts}.2.down_proj.weight"]
    )
    substitution = routed_experts.expert_substitution
    torch.testing.assert_close(
        substitution.values.cpu(), weights[value_name].unsqueeze(0)
    )

    updated = weights[value_name] + 1
    assert model.load_weights([(value_name, updated)]) == {
        f"{experts}.routed_experts.expert_substitution.values"
    }
    torch.testing.assert_close(substitution.values.cpu(), updated.unsqueeze(0))
