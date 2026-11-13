# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    DeepseekV32Config,
    Glm5NextTextConfig,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from vllm.config import (
    AttentionConfig,
    DeviceConfig,
    KernelConfig,
    LoadConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe.expert_substitution import (
    ConstantExpertSubstitution,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_nvfp4 import (  # noqa: E501
    CompressedTensorsW4A4Nvfp4MoEMethod,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.expert_substitution import as_expert_substitution_model
from vllm.model_executor.models.utils import AutoWeightsLoader, PPMissingLayer
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _moe_sparsity_config(retained_per_layer):
    """The `quantization_config.moe_sparsity_config` novice format."""
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
    nvfp4_weights: dict[str, object] = {
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


def _novices(layer: int) -> str:
    return f"model.layers.{layer}.mlp.experts.novices"


def _load_mone_model(
    model_dir: Path, *, backend="triton", dtype="bfloat16", load_config=None
):
    model_config = ModelConfig(
        model=str(model_dir),
        skip_tokenizer_init=True,
        dtype=dtype,
        max_model_len=32,
        enforce_eager=True,
    )
    load_config = load_config or LoadConfig(
        load_format="safetensors", use_tqdm_on_load=False
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cuda"),
        load_config=load_config,
        attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
        kernel_config=KernelConfig(moe_backend=backend),
    )
    return config, get_model_loader(load_config).load_model(config, model_config)


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
                layer.mlp.experts = torch.nn.Module()
                layer.mlp.experts.routed_experts = torch.nn.Module()
                layer.mlp.experts.routed_experts.expert_substitution = (
                    ConstantExpertSubstitution(index, 4, 2, 2, torch.float32)
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

    assert loaded == {"weight"} | {
        f"model.layers.{layer}.mlp.experts.routed_experts.expert_substitution.novices"
        for layer in local_layers
    }
    for layer in local_layers:
        experts = model.model.layers[layer].mlp.experts.routed_experts
        torch.testing.assert_close(
            experts.expert_substitution.novices,
            torch.full((2, 2), 10.0 * layer),
        )


def test_direct_reload_updates_novices_incrementally():
    model = _cpu_model(local_layers=(0,), tracked=False)
    experts = model.model.layers[0].mlp.experts.routed_experts
    novices = experts.expert_substitution.novices
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
    substitution = model.model.layers[0].mlp.experts.routed_experts.expert_substitution
    novices = substitution.novices

    initialize_layerwise_reload(model)
    with pytest.raises(NotImplementedError, match="layerwise weight reload"):
        model.load_weights([(name, weight + 1) for name, weight in loaded])

    assert substitution.novices is novices
    torch.testing.assert_close(novices, torch.zeros(2, 2))
    torch.testing.assert_close(model.weight, torch.tensor([3.0, 4.0]))


@pytest.mark.usefixtures("dist_init", "workspace_init")
def test_tensorizer_rejects_missing_novices(tmp_path: Path):
    """Raw Tensorizer loading must reject an unloaded local novices tensor."""
    from vllm.model_executor.model_loader.tensorizer import TensorSerializer

    weights = _write_tiny_qwen3_mone_checkpoint(tmp_path)
    del weights[_novices(0)]
    checkpoint = tmp_path / "model.tensors"
    with checkpoint.open("wb+") as stream:
        serializer = TensorSerializer(stream)
        serializer.write_state_dict(weights)
        serializer.close()

    load_config = LoadConfig(
        load_format="tensorizer",
        model_loader_extra_config={
            "tensorizer_config": {"tensorizer_uri": str(checkpoint)}
        },
    )
    with pytest.raises(ValueError, match="missing or incomplete"):
        _load_mone_model(tmp_path, dtype="float16", load_config=load_config)


def _write_tiny_qwen3_mone_checkpoint(
    model_dir: Path,
    *,
    head_dim: int = 8,
    hidden_size: int = 16,
    intermediate_size: int = 8,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """One MoE layer of 4 experts that retains experts 0-2 and collapses 3."""
    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=32,
        moe_intermediate_size=intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=32,
        decoder_sparse_step=1,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        tie_word_embeddings=False,
        dtype=str(dtype).removeprefix("torch."),
    )
    config.quantization_config = _moe_sparsity_config([3])

    experts = "model.layers.0.mlp.experts"
    hf_weights = Qwen3MoeForCausalLM(config).to(dtype).state_dict()
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
    weights[f"{experts}.novices"] = torch.arange(hidden_size, dtype=dtype).reshape(
        1, hidden_size
    )

    config.save_pretrained(model_dir)
    save_file(weights, model_dir / "model.safetensors")
    return weights


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
def test_loads_mone_checkpoint_into_compact_experts(tmp_path: Path):
    weights = _write_tiny_qwen3_mone_checkpoint(tmp_path)
    _, model = _load_mone_model(tmp_path, dtype="float16")

    experts = "model.layers.0.mlp.experts"
    moe = model.model.layers[0].mlp
    routed_experts = moe.experts.routed_experts
    assert isinstance(routed_experts.expert_substitution, ConstantExpertSubstitution)
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


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="DSA model initialization requires Hopper or newer",
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
@pytest.mark.parametrize("architecture", ["DeepseekV32", "Glm5Next"])
@pytest.mark.parametrize("retained", [None, 2])
def test_expert_loading_streams_tensors(tmp_path: Path, architecture, retained):
    from vllm.models.deepseek_v32.nvidia.model import DeepseekV32Model
    from vllm.models.glm5next.common.model import Glm5NextModel

    hf_cls, model_cls = (
        (DeepseekV32Config, DeepseekV32Model)
        if architecture == "DeepseekV32"
        else (Glm5NextTextConfig, Glm5NextModel)
    )
    config = hf_cls(
        architectures=[f"{architecture}ForCausalLM"],
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        q_lora_rank=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        n_group=1,
        topk_group=1,
        index_n_heads=32,
        index_topk=512,
        indexer_rope_interleave=False,
        max_position_embeddings=8192,
        first_k_dense_replace=0,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        layer_types=["deepseek_sparse_attention"] * 2,
        mlp_layer_types=["sparse"] * 2,
        mhc=False,
        mla_use_nope=architecture == "Glm5Next",
        pad_token_id=0,
    )
    if retained is not None:
        config.quantization_config = _moe_sparsity_config([retained, retained])
    config.save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=32,
            enforce_eager=True,
        ),
        kernel_config=KernelConfig(moe_backend="triton", enable_jit_warmup=False),
    )
    with (
        set_current_vllm_config(vllm_config),
        torch.device("cuda"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        model = model_cls(vllm_config=vllm_config, prefix="model")
        first = model.layers[0].mlp.experts.routed_experts
        second = model.layers[1].mlp.experts.routed_experts
        first.w13_weight.data.zero_()
        second.w2_weight.data.zero_()

        def weights():
            # Check before requesting the next tensor, then revisit layer 0.
            yield "layers.0.mlp.experts.gate_up_proj", torch.ones_like(first.w13_weight)
            assert (first.w13_weight == 1).all()
            yield "layers.1.mlp.experts.down_proj", torch.full_like(second.w2_weight, 2)
            assert (second.w2_weight == 2).all()
            yield "layers.0.mlp.experts.0.up_proj.weight", torch.full((256, 256), 3.0)
            assert (first.w13_weight[0, 256:] == 3).all()

        loaded = model.load_weights(weights())
        assert loaded == {
            "layers.0.mlp.experts.routed_experts.w13_weight",
            "layers.1.mlp.experts.routed_experts.w2_weight",
        }


def _write_tiny_qwen3_mone_nvfp4_checkpoint(
    model_dir: Path,
    *,
    hidden_size: int = 32,
    intermediate_size: int = 16,
    head_dim: int = 8,
) -> dict[str, torch.Tensor]:
    """One MoE layer of 4 NVFP4 experts that retains experts 0-2, collapses 3.

    Expert tensors use the stacked (fused) names of compact MoNE exports:
    `experts.gate_up_proj.weight_packed` and friends.
    """
    weights = _write_tiny_qwen3_mone_checkpoint(
        model_dir,
        head_dim=head_dim,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=torch.float32,
    )
    config = Qwen3MoeConfig.from_pretrained(model_dir)
    config.quantization_config = _nvfp4_quantization_config([3])

    experts = "model.layers.0.mlp.experts"
    generator = torch.Generator().manual_seed(42)
    for projection, shards in (("gate_up_proj", 2), ("down_proj", 1)):
        num_retained, rows, cols = weights.pop(f"{experts}.{projection}").shape
        prefix = f"{experts}.{projection}"
        # Every FP4 nibble is finite. Seeded patterns distinguish expert rows,
        # gate/up shards and orientations without implementing a quantizer.
        weights[f"{prefix}.weight_packed"] = torch.randint(
            0,
            256,
            (num_retained, rows, cols // 2),
            generator=generator,
            dtype=torch.uint8,
        )
        weights[f"{prefix}.weight_scale"] = torch.randint(
            1, 5, (num_retained, rows, cols // 16), generator=generator
        ).to(torch.float8_e4m3fn)
        global_scales = torch.arange(1, num_retained + 1, dtype=torch.float32)
        if projection == "down_proj":
            global_scales += num_retained
        weights[f"{prefix}.weight_global_scale"] = (
            global_scales[:, None].expand(-1, shards).contiguous()
        )
        weights[f"{prefix}.input_global_scale"] = torch.full(
            (num_retained, shards), 1e-2
        )
    config.save_pretrained(model_dir)
    save_file(weights, model_dir / "model.safetensors")
    return weights


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
def test_rejects_mixed_stacked_quantization(tmp_path: Path, vllm_runner, monkeypatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _write_tiny_qwen3_mone_nvfp4_checkpoint(tmp_path, head_dim=16)
    config = Qwen3MoeConfig.from_pretrained(tmp_path)
    config.quantization_config["ignore"] = ["model.layers.0.mlp.experts.down_proj"]
    config.save_pretrained(tmp_path)
    with (
        pytest.raises(ValueError, match="All MoE projections need to have same"),
        vllm_runner(
            str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=32,
            kv_cache_memory_bytes=16 * 1024 * 1024,
            enforce_eager=True,
            distributed_executor_backend="uni",
            attention_config={"backend": "FLASH_ATTN"},
            kernel_config={"moe_backend": "emulation", "enable_jit_warmup": False},
        ),
    ):
        pytest.fail("mixed stacked quantization was accepted")


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
def test_loads_nvfp4_mone_checkpoint_into_compact_experts(tmp_path: Path):
    weights = _write_tiny_qwen3_mone_nvfp4_checkpoint(tmp_path)
    vllm_config, model = _load_mone_model(tmp_path, backend="emulation")

    moe = model.model.layers[0].mlp
    routed_experts = moe.experts.routed_experts
    substitution = routed_experts.expert_substitution
    assert isinstance(substitution, ConstantExpertSubstitution)
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

    # Quantized retained experts must still include the novice's constant output.
    with set_forward_context(None, vllm_config):
        get_forward_context().all_moe_layers = None
        hidden_states = torch.randn(2, 32, dtype=torch.bfloat16, device="cuda")
        topk_weights = torch.ones(2, 1, dtype=torch.float32, device="cuda")
        topk_ids = torch.tensor([[3], [3]], device="cuda")
        output = routed_experts.forward_modular(
            hidden_states.clone(), topk_weights, topk_ids
        )
    torch.testing.assert_close(output, substitution.novices.expand_as(output))


@pytest.mark.skipif(
    not TrtLlmNvFp4ExpertsMonolithic._supports_current_device(),
    reason="requires FlashInfer TRTLLM NVFP4",
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
def test_native_nvfp4_mone_preserves_mixed_route_weights(tmp_path: Path):
    """Native routing must retain the novice's share of normalized weights."""
    weights = _write_tiny_qwen3_mone_nvfp4_checkpoint(
        tmp_path, hidden_size=256, intermediate_size=512
    )
    for name, weight in weights.items():
        if name.endswith("weight_packed"):
            weight.zero_()
    logits = torch.tensor([[0.0, -8.0, -8.0, 1.0], [1.0, 0.0, -8.0, -8.0]])
    # The Qwen runner owns the gate and computes logits from its input.
    gate = weights["model.layers.0.mlp.gate.weight"]
    gate.zero_()
    gate[:, :2] = logits.T
    save_file(weights, tmp_path / "model.safetensors")
    config, model = _load_mone_model(tmp_path, backend="flashinfer_trtllm")
    layer = model.model.layers[0].mlp.experts
    routed = layer.routed_experts
    assert routed.quant_method.is_monolithic
    assert routed.w13_weight.shape[0] == routed.w2_weight.shape[0] == 3
    hidden = torch.zeros(2, 256, dtype=torch.bfloat16, device="cuda")
    hidden[:, :2] = torch.eye(2, device="cuda")
    logits = logits.cuda()
    # Retained MLPs are zero. Compute full logical softmax/top-k explicitly;
    # the first row selects a novice, the second selects only retained experts.
    probabilities, ids = logits.double().softmax(-1).topk(2, dim=-1)
    probabilities /= probabilities.sum(-1, keepdim=True)
    # Round the novice's routing weight to the model dtype before multiplication.
    novice_weight = (probabilities * (ids == 3)).sum(-1).to(hidden.dtype)
    expected = (novice_weight[:, None] * weights[_novices(0)].double().cuda()).to(
        torch.bfloat16
    )
    with set_current_vllm_config(config), set_forward_context(None, config):
        get_forward_context().all_moe_layers = None
        actual = layer(hidden, logits)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _write_quantized_mone_checkpoint(model_dir: Path, quant_format: str):
    """Compact CT export with exactly representable, nonzero retained MLPs."""
    weights = _write_tiny_qwen3_mone_checkpoint(
        model_dir, head_dim=64, hidden_size=1024, intermediate_size=1024
    )
    config = Qwen3MoeConfig.from_pretrained(model_dir)
    experts = "model.layers.0.mlp.experts"
    quant = _moe_sparsity_config([2])
    weight_args: dict[str, object] = dict(
        num_bits=8, type="float", strategy="tensor", symmetric=True
    )
    input_args: dict[str, object] | None = None
    if quant_format.startswith("fp8"):
        input_args = dict(weight_args, dynamic=quant_format == "fp8_block")
    quant["format"] = "float-quantized"
    if quant_format == "fp8_block":
        weight_args.update(strategy="block", block_structure=[128, 128])
    elif quant_format in ("mxfp4", "mxint4"):
        weight_args.update(
            num_bits=4,
            strategy="group",
            group_size=32,
            type="int" if quant_format == "mxint4" else "float",
        )
        input_args = None
        quant["format"] = "pack-quantized"
    quant["config_groups"] = {
        "group_0": {
            "weights": weight_args,
            "input_activations": input_args,
            "targets": [f"{experts}.gate_up_proj", f"{experts}.down_proj"],
        }
    }
    config.quantization_config = quant
    config.save_pretrained(model_dir)
    gate_up = torch.zeros(2, 2048, 1024, dtype=torch.bfloat16)
    down = torch.zeros(2, 1024, 1024, dtype=torch.bfloat16)
    gate_up[:, 0, 0], gate_up[:, 1024, 0] = 16, 1
    down[0, :, 0], down[1, :, 0] = 1 / 16, 1 / 8
    for projection, canonical in (("gate_up_proj", gate_up), ("down_proj", down)):
        del weights[f"{experts}.{projection}"]
        prefix = f"{experts}.{projection}"
        if quant_format.startswith("fp8"):
            if quant_format == "fp8_block":
                scales = torch.ones(2, canonical.shape[1] // 128, 8)
                scales[1] = 2
                expanded = scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
            else:
                scales = torch.tensor([0.5, 2.0])
                expanded = scales[:, None, None]
                weights[f"{prefix}.input_scale"] = (
                    torch.ones(2, 2) if projection == "gate_up_proj" else torch.ones(2)
                )
            weights[f"{prefix}.weight"] = (canonical / expanded).to(torch.float8_e4m3fn)
            weights[f"{prefix}.weight_scale"] = (
                scales[:, None].expand(-1, 2).contiguous()
                if quant_format == "fp8_tensor" and projection == "gate_up_proj"
                else scales
            )
        elif quant_format == "mxfp4":
            from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
                mxfp4_quantize,
            )

            packed, scales = mxfp4_quantize(canonical.cuda())
            weights[f"{prefix}.weight_packed"] = packed.cpu()
            weights[f"{prefix}.weight_scale"] = scales.cpu().view(torch.uint8)
        else:
            # CT stores eight biased INT4 values per int32, with group scales.
            scales = torch.ones(2, canonical.shape[1], 32, dtype=torch.bfloat16)
            if projection == "gate_up_proj":
                scales[:, 0] = 8
                scales[:, 1024] = 0.5
            else:
                scales[0] = 1 / 32
                scales[1] = 1 / 16
            codes = (canonical / scales.repeat_interleave(32, -1)).to(torch.int64) + 8
            packed = (
                (codes.reshape(2, canonical.shape[1], -1, 8) << (4 * torch.arange(8)))
                .sum(-1)
                .to(torch.int32)
            )
            weights[f"{prefix}.weight_packed"] = packed
            weights[f"{prefix}.weight_scale"] = scales
    weights[_novices(0)] = torch.stack(
        [torch.full((1024,), 4), torch.full((1024,), 8)]
    ).to(torch.bfloat16)
    save_file(
        {name: value.contiguous() for name, value in weights.items()},
        model_dir / "model.safetensors",
    )
    return weights


@pytest.mark.skipif(
    not TrtLlmBf16ExpertsMonolithic._supports_current_device(),
    reason="requires FlashInfer TRTLLM on Blackwell",
)
@pytest.mark.usefixtures("dist_init", "workspace_init")
@pytest.mark.parametrize("quant_format", ["fp8_block", "fp8_tensor", "mxfp4", "mxint4"])
def test_loads_quantized_mone_and_preserves_logical_outputs(
    tmp_path: Path, quant_format, monkeypatch
):
    """Real compact weights/scales must survive loading, repacking and routing."""
    if quant_format == "mxint4":
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_INT4", "1")
    weights = _write_quantized_mone_checkpoint(tmp_path, quant_format)
    # Gate inputs select retained-only, mixed, novice-only and mixed routes.
    pairs = torch.tensor([[0, 1], [0, 2], [2, 3], [3, 1]])
    logits = torch.full((4, 4), -1000.0).scatter_(1, pairs, 0)
    gate = weights["model.layers.0.mlp.gate.weight"]
    gate.zero_()
    gate[:, 1:5] = logits.T
    save_file(weights, tmp_path / "model.safetensors")
    config, model = _load_mone_model(tmp_path, backend="flashinfer_trtllm")
    layer = model.model.layers[0].mlp.experts
    routed = layer.routed_experts
    assert routed.quant_method.is_monolithic
    assert routed.w13_weight.shape[0] == routed.w2_weight.shape[0] == 2
    assert routed.w13_weight_scale.shape[0] == routed.w2_weight_scale.shape[0] == 2
    assert model.model.layers[0].mlp.gate.weight.shape[0] == 4
    torch.testing.assert_close(
        routed.expert_substitution.novices.cpu(), weights[_novices(0)]
    )
    hidden = torch.zeros(4, 1024, dtype=torch.bfloat16, device="cuda")
    hidden[:, 0] = 1
    hidden[:, 1:5] = torch.eye(4, device="cuda")
    # The stored MLPs return 1 and 2; novices return 4 and 8. Equal selected
    # logits give half of each contribution, including the novice's share.
    expected = torch.tensor([1.5, 2.5, 6, 5], device="cuda", dtype=torch.bfloat16)
    expected = expected[:, None].expand_as(hidden)
    scores = logits.cuda()
    with set_current_vllm_config(config), set_forward_context(None, config):
        get_forward_context().all_moe_layers = None
        torch.testing.assert_close(layer(hidden, scores), expected, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(hidden, scores)
        hidden[:, 1:5] = hidden[:, 1:5].flip(0)
        graph.replay()
        torch.testing.assert_close(output, expected.flip(0), atol=0, rtol=0)


@pytest.mark.parametrize(
    "backend",
    [
        "triton",
        pytest.param(
            "flashinfer_trtllm",
            marks=pytest.mark.skipif(
                not TrtLlmBf16ExpertsMonolithic._supports_current_device(),
                reason="requires FlashInfer TRTLLM BF16",
            ),
        ),
    ],
)
def test_generation_includes_loaded_novices(tmp_path: Path, vllm_runner, backend):
    """Check novice magnitude as well as its effect on the winning token.

    Duplicating the novice contribution still selects token 15; the full
    log-probability reference must catch that error independently.
    """
    from vllm import SamplingParams

    monolithic = backend == "flashinfer_trtllm"
    weights = _write_tiny_qwen3_mone_checkpoint(
        tmp_path,
        head_dim=16,
        hidden_size=256 if monolithic else 16,
        intermediate_size=512 if monolithic else 8,
    )
    dtype = torch.bfloat16 if monolithic else torch.float16
    config = Qwen3MoeConfig.from_pretrained(tmp_path)
    config.num_experts_per_tok = 1
    config.save_pretrained(tmp_path)
    # Zero attention and retained MLPs leave only the embedding and novice
    # residual. The readout selects dimension 15 only when the novice is added.
    for weight in weights.values():
        weight.zero_()
    weights["model.embed_tokens.weight"].fill_(1)
    for name, weight in weights.items():
        if name.endswith("layernorm.weight") or name == "model.norm.weight":
            weight.fill_(1)
    weights["model.layers.0.mlp.gate.weight"][3].fill_(1)
    novice_row = weights[_novices(0)].shape[0] - 1
    weights[_novices(0)][novice_row, 15] = 4
    weights["lm_head.weight"][:16, :16] = torch.eye(16, dtype=torch.float16)
    weights["lm_head.weight"][0, 0] = 2
    save_file(weights, tmp_path / "model.safetensors")

    # Attention and retained MLPs are zero: compute the residual, final RMSNorm
    # and readout directly from the checkpoint, without invoking a vLLM model.
    residual = (
        weights["model.embed_tokens.weight"][1].double()
        + weights[_novices(0)][novice_row].double()
    )
    normalized = (
        residual * torch.rsqrt(residual.square().mean() + config.rms_norm_eps)
    ).to(dtype)
    expected_logprobs = torch.nn.functional.linear(
        normalized.float(), weights["lm_head.weight"].float()
    ).log_softmax(dim=-1)

    with vllm_runner(
        str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16" if monolithic else "float16",
        max_model_len=32,
        max_num_seqs=1,
        max_logprobs=config.vocab_size,
        enforce_eager=not monolithic,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1],
        }
        if monolithic
        else None,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        attention_config={"backend": "FLASH_ATTN"},
        kernel_config={"moe_backend": backend, "enable_jit_warmup": False},
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [1, 2]}],
            SamplingParams(
                temperature=0, max_tokens=2, ignore_eos=True, logprobs=config.vocab_size
            ),
            use_tqdm=False,
        )
    assert list(outputs[0].outputs[0].token_ids) == [15, 15]
    logprobs = outputs[0].outputs[0].logprobs
    assert logprobs is not None
    assert len(logprobs) == 2
    for step in logprobs:
        actual = torch.tensor(
            [step[token].logprob for token in range(config.vocab_size)]
        )
        torch.testing.assert_close(actual, expected_logprobs, atol=1e-6, rtol=0)
