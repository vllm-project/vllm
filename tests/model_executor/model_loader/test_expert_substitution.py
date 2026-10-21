# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from vllm.config import (
    AttentionConfig,
    DeviceConfig,
    KernelConfig,
    LoadConfig,
    ModelConfig,
    VllmConfig,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe.expert_substitution import (
    ConstantExpertSubstitution,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    cast_to_fp4,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.expert_substitution import as_expert_substitution_model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _moe_sparsity_config(retained_per_layer):
    """The ``quantization_config.moe_sparsity_config`` novice format."""
    return {
        "config_groups": {},
        "format": "fakequant",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "moe_sparsity_config": {
            "method": "novice",
            "num_experts": retained_per_layer,
        },
    }


# Both layers retain experts 0 and 1; experts 2 and 3 are novices.
MODEL_CONFIG = SimpleNamespace(
    hf_config=SimpleNamespace(quantization_config=_moe_sparsity_config([2, 2]))
)


def _nvfp4_quantization_config(retained_per_layer):
    """A compressed-tensors NVFP4 config that targets the stacked experts."""
    nvfp4_weights = {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": 16,
        "num_bits": 4,
        "observer": "memoryless_minmax",
        "observer_kwargs": {},
        "scale_dtype": "torch.float8_e4m3fn",
        "strategy": "tensor_group",
        "symmetric": True,
        "type": "float",
        "zp_dtype": None,
    }
    nvfp4_input = dict(nvfp4_weights, dynamic="local", observer="static_minmax")
    return {
        "config_groups": {
            "group_0": {
                "input_activations": nvfp4_input,
                "targets": [
                    "model.layers.0.mlp.experts.gate_up_proj",
                    "model.layers.0.mlp.experts.down_proj",
                ],
                "weights": nvfp4_weights,
            }
        },
        "format": "nvfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "moe_sparsity_config": {
            "method": "novice",
            "num_experts": retained_per_layer,
        },
    }


_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pack_nvfp4(weight: torch.Tensor, global_scale: float):
    """Quantize one [N, K] projection to packed FP4 and per-group scales.

    Mirrors ``ref_nvfp4_quant`` from ``nvfp4_emulation_utils`` plus the
    two-values-per-byte packing of ``break_fp4_bytes``.
    """
    num_rows, num_cols = weight.shape
    blocks = weight.float().reshape(num_rows, num_cols // 16, 16)
    vec_max = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (global_scale * vec_max / 6.0).clamp(-448, 448)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    scaled = blocks * global_scale / scale.clamp_min(1e-30)
    fp4 = cast_to_fp4(scaled.clamp(-6, 6))
    magnitudes = torch.tensor(_FP4_MAGNITUDES)
    nibbles = torch.searchsorted(magnitudes, fp4.abs())
    nibbles = nibbles + (fp4 < 0).to(torch.uint8) * 8
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    return (
        packed.reshape(num_rows, num_cols // 2).to(torch.uint8),
        scale.reshape(num_rows, num_cols // 16).to(torch.float8_e4m3fn),
    )


def _novices(layer: int) -> str:
    return f"model.layers.{layer}.mlp.experts.novices"


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
                    index, 4, (2, 3), 2, torch.float32
                )
                self.model.layers.append(layer)

        def load_weights(self, weights):
            loaded = AutoWeightsLoader(self).load_weights(weights)
            return loaded if tracked else None

    return as_expert_substitution_model(Model, model_config)()


@pytest.mark.parametrize("local_layers", [(0, 1), (0,), (1,), ()])
def test_loads_local_novices_and_passes_through_others(local_layers):
    """Other PP stages and MTP drafts own none or some of the configured layers."""
    model = _cpu_model(local_layers)
    loaded = model.load_weights(
        [
            ("weight", torch.tensor([1.0, 2.0])),
            *(
                (_novices(layer), torch.full((2, 2), 10.0 * layer))
                for layer in range(2)
            ),
        ]
    )
    model.process_weights_after_loading()

    assert loaded == {"weight"} | {
        f"model.layers.{layer}.mlp.experts.novices" for layer in local_layers
    }
    for layer in local_layers:
        torch.testing.assert_close(
            model.model.layers[layer].mlp.experts.novices,
            torch.full((2, 2), 10.0 * layer),
        )


def test_direct_reload_updates_novices_incrementally():
    model = _cpu_model(local_layers=(0,), tracked=False)
    novices = model.model.layers[0].mlp.experts.novices
    model.load_weights([(_novices(0), torch.ones(2, 2))])

    assert model.load_weights([(_novices(1), torch.ones(2, 2))]) is None
    torch.testing.assert_close(novices, torch.ones(2, 2))
    model.load_weights([(_novices(0), torch.zeros(2, 2))])
    torch.testing.assert_close(novices, torch.zeros(2, 2))

    with pytest.raises(ValueError, match="has shape"):
        model.load_weights([(_novices(0), torch.ones(3, 2))])


def test_layerwise_reload_is_rejected_and_keeps_weights():
    from vllm.model_executor.model_loader.reload import (
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    model = _cpu_model(local_layers=(0,))
    record_metadata_for_reloading(model)
    loaded = [
        ("weight", torch.tensor([3.0, 4.0])),
        (_novices(0), torch.zeros(2, 2)),
    ]
    model.load_weights(loaded)
    model.process_weights_after_loading()
    substitution = model.model.layers[0].mlp.experts
    novices = substitution.novices

    initialize_layerwise_reload(model)
    with pytest.raises(NotImplementedError, match="layerwise weight reload"):
        model.load_weights([(name, weight + 1) for name, weight in loaded])

    assert substitution.novices is novices
    torch.testing.assert_close(novices, torch.zeros(2, 2))
    torch.testing.assert_close(model.weight, torch.tensor([3.0, 4.0]))
    assert substitution.expert_substitution_routes.tolist() == [0, 1, -1, -2]


def test_missing_novices_tensor_is_rejected():
    model = _cpu_model(local_layers=(0, 1))
    model.load_weights([("weight", torch.tensor([1.0, 2.0]))])
    with pytest.raises(ValueError, match="missing or incomplete"):
        model.process_weights_after_loading()


@pytest.mark.parametrize("missing_novices", [False, True])
def test_tensorizer_finalizes_local_novices(monkeypatch, missing_novices):
    """Raw Tensorizer loading must reject an unloaded local novices tensor."""
    from vllm.model_executor.model_loader import tensorizer_loader

    model = _cpu_model(local_layers=(0,))
    weights = [("weight", torch.tensor([1.0, 2.0]))]
    if not missing_novices:
        weights.append((_novices(0), torch.ones(2, 2)))
    finalize = Mock(wraps=model.process_weights_after_loading)
    monkeypatch.setattr(model, "process_weights_after_loading", finalize)
    monkeypatch.setattr(tensorizer_loader, "initialize_model", lambda **kwargs: model)
    loader = object.__new__(tensorizer_loader.TensorizerLoader)
    monkeypatch.setattr(loader, "_get_weights_iterator", lambda: iter(weights))
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )

    if missing_novices:
        with pytest.raises(ValueError, match="missing or incomplete"):
            loader._load_model_serialized_cpu(config)
    else:
        assert loader._load_model_serialized_cpu(config) is model
        torch.testing.assert_close(
            model.model.layers[0].mlp.experts.novices,
            torch.ones(2, 2),
        )
    finalize.assert_called_once_with()


def _write_tiny_qwen3_mone_checkpoint(model_dir: Path) -> dict[str, torch.Tensor]:
    """One MoE layer of 4 experts that retains experts 0-2 and collapses 3."""
    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        decoder_sparse_step=1,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        dtype="float16",
    )
    config.quantization_config = _moe_sparsity_config([3])

    experts = "model.layers.0.mlp.experts"
    hf_weights = Qwen3MoeForCausalLM(config).to(torch.float16).state_dict()
    weights = {}
    for name, value in hf_weights.items():
        if name.startswith(f"{experts}."):
            continue
        weights[name] = value.contiguous()
    # Retained experts 0-2 are left-shifted into stacked tensors; the router
    # keeps its full width and expert 3 collapses into one novice row.
    weights[f"{experts}.gate_up_proj"] = hf_weights[f"{experts}.gate_up_proj"][
        :3
    ].contiguous()
    weights[f"{experts}.down_proj"] = hf_weights[f"{experts}.down_proj"][
        :3
    ].contiguous()
    weights[f"{experts}.novices"] = torch.arange(16, dtype=torch.float16).reshape(1, 16)

    config.save_pretrained(model_dir)
    save_file(weights, model_dir / "model.safetensors")
    return weights


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
def test_loads_mone_checkpoint_into_compact_experts(tmp_path: Path):
    weights = _write_tiny_qwen3_mone_checkpoint(tmp_path)
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
        attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
        kernel_config=KernelConfig(moe_backend="triton"),
    )
    model = get_model_loader(load_config).load_model(vllm_config, model_config)

    experts = "model.layers.0.mlp.experts"
    moe = model.model.layers[0].mlp
    routed_experts = moe.experts.routed_experts
    assert isinstance(routed_experts.expert_substitution, ConstantExpertSubstitution)
    assert routed_experts.expert_substitution.substituted_expert_ids == (3,)
    assert routed_experts.w13_weight.shape[0] == 3
    # Stacked rows load unchanged into the compact physical rows.
    torch.testing.assert_close(
        routed_experts.w13_weight.cpu(), weights[f"{experts}.gate_up_proj"]
    )
    torch.testing.assert_close(
        routed_experts.w2_weight.cpu(), weights[f"{experts}.down_proj"]
    )
    # The router keeps the full logical expert width.
    torch.testing.assert_close(
        moe.gate.weight.cpu(), weights["model.layers.0.mlp.gate.weight"]
    )
    substitution = routed_experts.expert_substitution
    torch.testing.assert_close(
        substitution.novices.cpu(), weights[f"{experts}.novices"]
    )

    updated = weights[f"{experts}.novices"] + 1
    assert model.load_weights([(f"{experts}.novices", updated)]) == {
        f"{experts}.routed_experts.expert_substitution.novices"
    }
    torch.testing.assert_close(substitution.novices.cpu(), updated)


def _write_tiny_qwen3_mone_nvfp4_checkpoint(model_dir: Path) -> dict[str, torch.Tensor]:
    """One MoE layer of 4 NVFP4 experts that retains experts 0-2, collapses 3.

    Expert tensors use the stacked (fused) names of compact MoNE exports:
    ``experts.gate_up_proj.weight_packed`` and friends.
    """
    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        decoder_sparse_step=1,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        dtype="float32",
    )
    config.quantization_config = _nvfp4_quantization_config([3])

    experts = "model.layers.0.mlp.experts"
    hf_weights = Qwen3MoeForCausalLM(config).to(torch.float32).state_dict()
    weights = {}
    for name, value in hf_weights.items():
        if name.startswith(f"{experts}."):
            continue
        weights[name] = value.contiguous()

    gate_up = hf_weights[f"{experts}.gate_up_proj"][:3]
    down = hf_weights[f"{experts}.down_proj"][:3]
    num_retained, fused_intermediate, hidden = gate_up.shape
    intermediate = fused_intermediate // 2

    def _quantize_stack(stacked: torch.Tensor, shard_size: int):
        packed, scales, globals_ = [], [], []
        for expert in stacked:
            # One encode scale per expert row, shared by the gate/up shards.
            global_scale = 448.0 / 6.0 / expert.abs().amax().clamp_min(1e-6)
            shards = (
                (expert[:shard_size], expert[shard_size:])
                if shard_size != expert.shape[0]
                else (expert,)
            )
            expert_packed, expert_scales = [], []
            for shard in shards:
                shard_packed, shard_scales = _pack_nvfp4(shard, global_scale)
                expert_packed.append(shard_packed)
                expert_scales.append(shard_scales)
            packed.append(torch.cat(expert_packed, dim=0))
            scales.append(torch.cat(expert_scales, dim=0))
            globals_.append(torch.full((len(shards),), float(global_scale)))
        return (
            torch.stack(packed),
            torch.stack(scales),
            torch.stack(globals_),
        )

    gate_up_packed, gate_up_scales, gate_up_globals = _quantize_stack(
        gate_up, intermediate
    )
    down_packed, down_scales, down_globals = _quantize_stack(down, down.shape[1])
    weights[f"{experts}.gate_up_proj.weight_packed"] = gate_up_packed.contiguous()
    weights[f"{experts}.gate_up_proj.weight_scale"] = gate_up_scales.contiguous()
    weights[f"{experts}.gate_up_proj.weight_global_scale"] = gate_up_globals
    weights[f"{experts}.gate_up_proj.input_global_scale"] = torch.full(
        (num_retained, 2), 1e-2
    )
    weights[f"{experts}.down_proj.weight_packed"] = down_packed.contiguous()
    weights[f"{experts}.down_proj.weight_scale"] = down_scales.contiguous()
    weights[f"{experts}.down_proj.weight_global_scale"] = down_globals
    weights[f"{experts}.down_proj.input_global_scale"] = torch.full(
        (num_retained, 1), 1e-2
    )
    weights[f"{experts}.novices"] = torch.arange(32, dtype=torch.float32).reshape(1, 32)

    config.save_pretrained(model_dir)
    save_file(weights, model_dir / "model.safetensors")
    return weights


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
def test_loads_nvfp4_mone_checkpoint_into_compact_experts(tmp_path: Path):
    weights = _write_tiny_qwen3_mone_nvfp4_checkpoint(tmp_path)
    model_config = ModelConfig(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=32,
        enforce_eager=True,
    )
    load_config = LoadConfig(load_format="safetensors", use_tqdm_on_load=False)
    vllm_config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cuda"),
        load_config=load_config,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
        kernel_config=KernelConfig(moe_backend="emulation"),
    )
    model = get_model_loader(load_config).load_model(vllm_config, model_config)

    moe = model.model.layers[0].mlp
    routed_experts = moe.experts.routed_experts
    substitution = routed_experts.expert_substitution
    assert isinstance(substitution, ConstantExpertSubstitution)
    assert substitution.substituted_expert_ids == (3,)
    assert isinstance(routed_experts.quant_method, CompressedTensorsW4A4Nvfp4MoEMethod)
    # The emulation backend keeps the serialized layout, so the compact
    # rows load bit-exactly.
    experts = "model.layers.0.mlp.experts"
    torch.testing.assert_close(
        routed_experts.w13_weight.cpu(),
        weights[f"{experts}.gate_up_proj.weight_packed"],
    )
    torch.testing.assert_close(
        routed_experts.w2_weight.cpu(),
        weights[f"{experts}.down_proj.weight_packed"],
    )
    torch.testing.assert_close(
        routed_experts.w13_weight_scale.cpu(),
        weights[f"{experts}.gate_up_proj.weight_scale"],
    )
    torch.testing.assert_close(
        routed_experts.w2_weight_scale.cpu(),
        weights[f"{experts}.down_proj.weight_scale"],
    )
    torch.testing.assert_close(
        routed_experts.w13_weight_scale_2.cpu(),
        1.0 / weights[f"{experts}.gate_up_proj.weight_global_scale"][:, 0],
    )
    torch.testing.assert_close(
        routed_experts.w2_weight_scale_2.cpu(),
        1.0 / weights[f"{experts}.down_proj.weight_global_scale"][:, 0],
    )
    # The router keeps the full logical expert width and the novice loads.
    torch.testing.assert_close(
        moe.gate.weight.cpu(),
        weights["model.layers.0.mlp.gate.weight"].to(torch.bfloat16),
    )
    torch.testing.assert_close(
        substitution.novices.cpu(), weights[f"{experts}.novices"].to(torch.bfloat16)
    )

    # A route to the collapsed expert contributes exactly the novice
    # constant; the redirected physical route carries zero weight.
    with set_forward_context(None, vllm_config):
        get_forward_context().all_moe_layers = None
        hidden_states = torch.randn(2, 32, dtype=torch.bfloat16, device="cuda")
        topk_weights = torch.ones(2, 1, dtype=torch.float32, device="cuda")
        topk_ids = torch.tensor([[3], [3]], device="cuda")
        output = routed_experts.forward_modular(
            hidden_states.clone(), topk_weights, topk_ids
        )
    torch.testing.assert_close(
        output,
        weights[f"{experts}.novices"].to(torch.bfloat16).cuda().expand(2, -1),
    )
