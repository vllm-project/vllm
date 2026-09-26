# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
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
    make_expert_substitution,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.expert_substitution import as_expert_substitution_model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _cpu_config(prefix: str = ""):
    prefix = f"{prefix}." if prefix else ""
    return SimpleNamespace(
        compression_config={
            "transform_config": {
                "expert_substitution": {
                    "version": 1,
                    "router_semantics": {
                        "preserve_logical_expert_ids": True,
                        "preserve_router_weights": True,
                        "renormalize_after_substitution": False,
                    },
                    "targets": {
                        f"{prefix}model.layers.{layer}.experts": {
                            "num_logical_experts": 3,
                            "weight_layout": "compact_retained_experts",
                            "replacements": {
                                str(expert): {
                                    "format": "constant-v1",
                                    "tensors": {"value": name},
                                }
                                for expert, name in (
                                    (1, "constants.shared"),
                                    (2, f"constants.layer_{layer}"),
                                )
                            },
                        }
                        for layer in range(2)
                    },
                }
            }
        }
    )


def _cpu_model(config, *, local_layer=0, loader="auto", prefix=""):
    class Model(torch.nn.Module):
        def __init__(self, prefix=""):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList()
            self.finalize_calls = 0
            for index in range(2):
                if index != local_layer:
                    self.model.layers.append(PPMissingLayer())
                    continue
                layer = torch.nn.Module()
                layer.experts = torch.nn.Module()
                path = f"model.layers.{index}.experts"
                path = f"{prefix}.{path}" if prefix else path
                layer.experts.substitution = make_expert_substitution(
                    config, path, 3, 2, torch.float32
                )
                self.model.layers.append(layer)

        def load_weights(self, weights):
            if loader == "auto":
                return AutoWeightsLoader(self).load_weights(weights)
            loaded = set()
            for name, value in weights:
                self.get_parameter(name).data.copy_(value)
                loaded.add(name)
            return None if loader == "untracked" else loaded

        def process_weights_after_loading(self):
            self.finalize_calls += 1

    return as_expert_substitution_model(Model, config)(prefix=prefix)


@pytest.mark.parametrize("loader", ["auto", "legacy", "untracked"])
def test_direct_substitution_loading_preserves_incremental_updates(loader):
    """A direct reload must consume metadata names before the model's loader."""
    model = _cpu_model(_cpu_config(), loader=loader)
    substitution = model.model.layers[0].experts.substitution
    model.load_weights([("constants.shared", torch.tensor([1.0, 2.0]))])
    with pytest.raises(ValueError, match="missing 1 constant expert value"):
        model.process_weights_after_loading()
    model.load_weights([("constants.layer_0", torch.tensor([3.0, 4.0]))])
    model.process_weights_after_loading()

    loaded = model.load_weights(
        [
            ("weight", torch.tensor([5.0, 6.0])),
            ("constants.shared", torch.tensor([7.0, 8.0])),
            ("constants.layer_0", torch.tensor([9.0, 10.0])),
        ]
    )
    if loader == "untracked":
        assert loaded is None
    else:
        assert loaded == {"weight", "model.layers.0.experts.substitution.values"}
    torch.testing.assert_close(model.weight, torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(
        substitution.values, torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    )

    model.load_weights([("constants.shared", torch.tensor([11.0, 12.0]))])
    model.process_weights_after_loading()
    torch.testing.assert_close(
        substitution.values, torch.tensor([[11.0, 12.0], [9.0, 10.0]])
    )
    assert model.finalize_calls == 2
    with pytest.raises(ValueError, match="has shape"):
        model.load_weights([("constants.shared", torch.ones(3))])


@pytest.mark.parametrize("local_layer", [0, 1])
@pytest.mark.parametrize("prefix", ["", "language_model"])
@pytest.mark.parametrize("suffix_paths", [False, True])
def test_pipeline_stage_loads_local_constants_with_external_names(
    local_layer, prefix, suffix_paths
):
    """Remote-only tensors are skipped; a shared local/remote tensor still loads."""
    config = _cpu_config(prefix)
    if suffix_paths:
        schema = config.compression_config["transform_config"]["expert_substitution"]
        schema["targets"] = {
            "layers." + path.split(".layers.", 1)[1]: target
            for path, target in schema["targets"].items()
        }
    model = _cpu_model(config, local_layer=local_layer, prefix=prefix)
    model.load_weights(
        [
            ("constants.shared", torch.tensor([1.0, 2.0])),
            ("constants.layer_0", torch.tensor([3.0, 4.0])),
            ("constants.layer_1", torch.tensor([5.0, 6.0])),
        ]
    )
    model.process_weights_after_loading()
    torch.testing.assert_close(
        model.model.layers[local_layer].experts.substitution.values,
        torch.tensor([[1.0, 2.0], [3.0 + 2 * local_layer, 4.0 + 2 * local_layer]]),
    )


@pytest.mark.parametrize("bad_path", ["model.layers.0.typo", "model.layers.2.experts"])
def test_pipeline_validation_rejects_unbound_local_targets(bad_path):
    config = _cpu_config()
    targets = config.compression_config["transform_config"]["expert_substitution"][
        "targets"
    ]
    targets[bad_path] = targets.pop("model.layers.0.experts")
    with pytest.raises(ValueError, match="unmatched targets"):
        _cpu_model(config)


@pytest.mark.parametrize("missing_value", [False, True])
def test_tensorizer_finalizes_model_loading_on_cpu(monkeypatch, missing_value):
    from vllm.model_executor.model_loader import tensorizer_loader

    model = _cpu_model(_cpu_config())
    weights = [("constants.shared", torch.ones(2))]
    if not missing_value:
        weights.append(("constants.layer_0", torch.zeros(2)))
    monkeypatch.setattr(tensorizer_loader, "initialize_model", lambda **kwargs: model)
    loader = object.__new__(tensorizer_loader.TensorizerLoader)
    monkeypatch.setattr(loader, "_get_weights_iterator", lambda: iter(weights))
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    if missing_value:
        with pytest.raises(ValueError, match="missing 1 constant expert value"):
            loader._load_model_serialized_cpu(config)
    else:
        assert loader._load_model_serialized_cpu(config) is model
        assert model.finalize_calls == 1


@pytest.mark.parametrize("partial", [False, True])
def test_layerwise_reload_updates_constants_and_preserves_routing(partial):
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    model = _cpu_model(_cpu_config())
    record_metadata_for_reloading(model)
    model.load_weights(
        [("constants.shared", torch.zeros(2)), ("constants.layer_0", torch.ones(2))]
    )
    model.process_weights_after_loading()
    substitution = model.model.layers[0].experts.substitution
    original_values = substitution.values
    initialize_layerwise_reload(model)
    weights = [
        ("weight", torch.tensor([3.0, 4.0])),
        ("constants.shared", torch.tensor([5.0, 6.0])),
    ]
    if not partial:
        weights.append(("constants.layer_0", torch.tensor([7.0, 8.0])))
    model.load_weights(weights)
    if partial:
        with pytest.raises(ValueError, match="layerwise reload requires all constant"):
            finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))
        torch.testing.assert_close(
            original_values, torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        )
        return
    finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))
    torch.testing.assert_close(model.weight, torch.tensor([3.0, 4.0]))
    assert substitution.values is original_values
    torch.testing.assert_close(
        substitution.values, torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    )
    assert substitution.logical_to_physical.tolist() == [0, -1, -1]
    assert substitution.substitution_index.tolist() == [-1, 0, 1]

    initialize_layerwise_reload(model)
    model.load_weights([("constants.shared", torch.zeros(2))])
    with pytest.raises(ValueError, match="layerwise reload requires all constant"):
        finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.float32))
    torch.testing.assert_close(original_values, torch.tensor([[5.0, 6.0], [7.0, 8.0]]))


def _write_tiny_deepseek_substitution_checkpoint(
    model_dir: Path, *, include_value: bool = True
) -> torch.Tensor:
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
        n_routed_experts=2,
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
    value_name = "model.layers.0.mlp.expert_replacements.1.value"
    config.compression_config = {
        "producer": {"name": "llm-compressor"},
        "transform_config": {
            "expert_substitution": {
                "version": 1,
                "router_semantics": {
                    "preserve_logical_expert_ids": True,
                    "preserve_router_weights": True,
                    "renormalize_after_substitution": False,
                },
                "targets": {
                    "model.layers.0.mlp.experts": {
                        "num_logical_experts": 2,
                        "weight_layout": "compact_retained_experts",
                        "replacements": {
                            "1": {
                                "format": "constant-v1",
                                "tensors": {"value": value_name},
                            }
                        },
                    }
                },
            }
        },
    }

    hf_model = DeepseekV2ForCausalLM(config).to(torch.float16)
    hf_weights = hf_model.state_dict()
    fused_gate_up_name = "model.layers.0.mlp.experts.gate_up_proj"
    fused_down_name = "model.layers.0.mlp.experts.down_proj"
    weights = {
        name: value.contiguous()
        for name, value in hf_weights.items()
        if name not in (fused_gate_up_name, fused_down_name)
    }
    retained_gate_up = hf_weights[fused_gate_up_name][0]
    retained_gate, retained_up = retained_gate_up.chunk(2, dim=0)
    retained_prefix = "model.layers.0.mlp.experts.0"
    weights[f"{retained_prefix}.gate_proj.weight"] = retained_gate.contiguous()
    weights[f"{retained_prefix}.up_proj.weight"] = retained_up.contiguous()
    weights[f"{retained_prefix}.down_proj.weight"] = hf_weights[fused_down_name][
        0
    ].contiguous()
    substitution_value = torch.arange(config.hidden_size, dtype=torch.float16)
    if include_value:
        weights[value_name] = substitution_value
    config.save_pretrained(model_dir)
    save_file(weights, model_dir / "model.safetensors")
    return substitution_value


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
@pytest.mark.parametrize("load_format", ["safetensors", "runai_streamer", "tensorizer"])
@pytest.mark.parametrize(
    "missing_value", [False, True], ids=["complete", "missing-value"]
)
def test_transform_only_config_loads_substituted_expert_checkpoint(
    tmp_path: Path, load_format: str, missing_value: bool
):
    if load_format == "runai_streamer":
        pytest.importorskip("runai_model_streamer")
    expected_value = _write_tiny_deepseek_substitution_checkpoint(
        tmp_path, include_value=not missing_value
    )
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="float16",
        max_model_len=32,
        enforce_eager=True,
    )

    assert model_config.quantization is None
    assert model_config.model_arch_config.quantization_config is None
    assert "quant_method" not in model_config.hf_config.compression_config

    extra_config = {}
    if load_format == "tensorizer":
        tensorizer = pytest.importorskip("tensorizer")
        tensorizer_path = tmp_path / "model.tensors"
        serializer = tensorizer.TensorSerializer(tensorizer_path)
        serializer.write_state_dict(load_file(tmp_path / "model.safetensors"))
        serializer.close()
        extra_config = {"tensorizer_config": {"tensorizer_uri": str(tensorizer_path)}}
    load_config = LoadConfig(
        load_format=load_format,
        use_tqdm_on_load=False,
        model_loader_extra_config=extra_config,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cuda"),
        load_config=load_config,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_MLA),
        kernel_config=KernelConfig(moe_backend="triton"),
    )
    loader = get_model_loader(load_config)
    if missing_value:
        with pytest.raises(
            ValueError, match="constant expert value|expert_substitution"
        ):
            loader.load_model(vllm_config, model_config)
        return
    model = loader.load_model(vllm_config, model_config)

    substitution = model.model.layers[0].mlp.experts.routed_experts.expert_substitution
    assert substitution is not None
    assert substitution.num_compute_experts == 1
    assert substitution.logical_to_physical.tolist() == [0, -1]
    torch.testing.assert_close(substitution.values.cpu(), expected_value.unsqueeze(0))

    updated_value = expected_value + 1
    loaded = model.load_weights(
        [("model.layers.0.mlp.expert_replacements.1.value", updated_value)]
    )
    assert loaded == {
        "model.layers.0.mlp.experts.routed_experts.expert_substitution.values"
    }
    torch.testing.assert_close(substitution.values.cpu(), updated_value.unsqueeze(0))
