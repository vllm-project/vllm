# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import inspect
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torch import nn

import vllm.config as vllm_config_module
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import utils as fused_moe_utils
from vllm.model_executor.layers.fused_moe.layer import determine_expert_counts
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    is_shared_expert_quant_fse_compatible,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.models.deepseek_v4 import quant_config as deepseek_v4_quant_config
from vllm.models.minimax_m3.amd import model as minimax_m3_model
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig

pytestmark = pytest.mark.skipif(
    current_platform.is_xpu(),
    reason="ROCm-specific aiter ops are not supported on XPU",
)

_QUARK_FSE_CONFIG: dict[str, Any] = {
    "global_quant_config": {
        "input_tensors": {
            "dtype": "fp4",
            "is_dynamic": True,
            "qscheme": "per_group",
            "ch_axis": -1,
            "group_size": 32,
            "block_size": None,
            "symmetric": None,
            "round_method": "half_even",
            "scale_type": "float",
            "scale_format": "e8m0",
            "scale_calculation_mode": "even",
            "mx_element_dtype": None,
            "observer_cls": "PerBlockMXObserver",
            "is_scale_quant": False,
            "enable_buffer_reuse": False,
            "max_input_numel": 4194304,
        },
        "output_tensors": None,
        "weight": {
            "dtype": "fp4",
            "is_dynamic": False,
            "qscheme": "per_group",
            "ch_axis": -1,
            "group_size": 32,
            "block_size": None,
            "symmetric": None,
            "round_method": "half_even",
            "scale_type": "float",
            "scale_format": "e8m0",
            "scale_calculation_mode": "even",
            "mx_element_dtype": None,
            "observer_cls": "PerBlockMXObserver",
            "is_scale_quant": False,
            "enable_buffer_reuse": False,
            "max_input_numel": 4194304,
        },
        "bias": None,
        "target_device": None,
    },
    "algo_config": None,
    "softmax_quant_spec": None,
    "quant_method": "quark",
    "layer_type_quant_config": {},
    "layer_quant_config": {},
    "kv_cache_quant_config": {},
    "kv_cache_post_rope": False,
    "quant_mode": "eager_mode",
    "version": "0.12+9d3d471cdf1",
    "export": {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    },
}


def get_deepseek_v4_quark_config(exclude: list[str]) -> dict[str, Any]:
    """Return the DeepSeek-V4-Pro-MXFP4 FSE quantization layout."""
    # Mimics https://huggingface.co/amd/DeepSeek-V4-Pro-MXFP4.
    quantization_config: dict[str, Any] = deepcopy(_QUARK_FSE_CONFIG)
    quantization_config["exclude"] = exclude
    mxfp4_config = cast(
        dict[str, Any], deepcopy(quantization_config["global_quant_config"])
    )
    fp8_config = deepcopy(mxfp4_config)
    fp8_config["input_tensors"].update(
        {
            "dtype": "fp8_e4m3",
            "group_size": 128,
            "symmetric": True,
            "scale_type": None,
            "scale_format": None,
            "scale_calculation_mode": None,
            "observer_cls": "PerGroupMinMaxObserver",
        }
    )
    fp8_config["weight"].update(
        {
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "ch_axis": None,
            "group_size": None,
            "block_size": [128, 128],
            "symmetric": True,
            "scale_type": "float8_e8m0fnu",
            "scale_format": None,
            "scale_calculation_mode": None,
            "observer_cls": "PerBlock2DMinMaxObserver",
        }
    )
    quantization_config["layer_quant_config"] = {
        r"re:mtp\.0\.ffn\.experts\.\d+\.w1": mxfp4_config,
        r"re:mtp\.0\.ffn\.shared_experts\.w1": fp8_config,
    }
    return quantization_config


def get_fse_test_model_config(
    model_type: str,
    quantization_config: dict[str, Any],
) -> tuple[object, type[nn.Module]]:
    if model_type == "minimax_m3":
        config = MiniMaxM3TextConfig(
            hidden_size=128,
            intermediate_size=32,
            dense_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=128,
            num_local_experts=2,
            num_experts_per_tok=1,
            moe_layer_freq=[0, 1],
            sparse_attention_config=None,
            rotary_dim=64,
            quantization_config=quantization_config,
        )
        return config, minimax_m3_model.MiniMaxM3Model
    if model_type == "deepseek_v4":
        from vllm.models.deepseek_v4.amd.model import DeepseekV4Model

        config = SimpleNamespace(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=1,
            head_dim=128,
            max_position_embeddings=128,
            sliding_window=None,
            compress_ratios=[1],
            rope_theta=10000.0,
            compress_rope_theta=10000.0,
            rope_parameters={"rope_type": "default"},
            rms_norm_eps=1e-6,
            hidden_act="silu",
            q_lora_rank=0,
            o_lora_rank=0,
            o_groups=1,
            qk_rope_head_dim=64,
            index_head_dim=64,
            index_n_heads=1,
            index_topk=1,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            moe_intermediate_size=32,
            norm_topk_prob=False,
            num_hash_layers=0,
            swiglu_limit=0.0,
            hc_eps=1e-6,
            hc_mult=1,
            hc_sinkhorn_iters=1,
            expert_dtype="fp4",
            num_nextn_predict_layers=1,
            quantization_config=quantization_config,
        )
        return config, DeepseekV4Model
    if model_type == "qwen3_5":
        from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

        config = Qwen3_5MoeTextConfig(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=128,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts_per_tok=1,
            num_experts=2,
            layer_types=["full_attention"],
            quantization_config=quantization_config,
        )
        return config, Qwen3_5Model
    if model_type == "deepseek_v2":
        from transformers import DeepseekV2Config

        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

        config = DeepseekV2Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=128,
            first_k_dense_replace=0,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            moe_intermediate_size=32,
            q_lora_rank=None,
            kv_lora_rank=0,
            qk_nope_head_dim=0,
            qk_rope_head_dim=0,
            v_head_dim=0,
            quantization_config=quantization_config,
        )
        return config, DeepseekV2Model
    if model_type == "glm4_moe":
        from transformers.models.glm4_moe import Glm4MoeConfig

        from vllm.model_executor.models.glm4_moe import Glm4MoeModel

        config = Glm4MoeConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=128,
            moe_intermediate_size=32,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            first_k_dense_replace=0,
            quantization_config=quantization_config,
        )
        return config, Glm4MoeModel
    raise ValueError(f"Unsupported FSE test model: {model_type}")


def test_determine_expert_counts_fuse_shared_experts_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common_args = (8, 0, 2)
    assert determine_expert_counts(*common_args, True)[2] == 2
    assert determine_expert_counts(*common_args, False)[2] == 0


def test_resolve_layer_fused_shared_expert_skips_compatibility_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "is_shared_expert_quant_fse_compatible",
        lambda *_: pytest.fail(
            "compatibility must not be checked when FSE is disabled"
        ),
    )

    assert not fused_moe_utils.resolve_layer_fused_shared_expert(
        object(), "model.layers.0.mlp"
    )


def test_resolve_layer_fused_shared_expert_normalizes_unavailable_aiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: None,
    )

    assert (
        fused_moe_utils.resolve_layer_fused_shared_expert(
            object(), "model.layers.0.mlp"
        )
        is False
    )


def test_resolve_layer_fused_shared_expert_passes_module_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_config = object()
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )

    calls: list[tuple[object, str, str]] = []

    def check_compatibility(
        config: object, expert_prefix: str, shared_expert_prefix: str
    ) -> tuple[bool, str | None]:
        calls.append((config, expert_prefix, shared_expert_prefix))
        return True, None

    monkeypatch.setattr(
        fused_moe_utils, "is_shared_expert_quant_fse_compatible", check_compatibility
    )

    assert fused_moe_utils.resolve_layer_fused_shared_expert(
        quant_config,
        "model.layers.0.mlp",
        shared_expert_name="shared_expert",
    )
    assert calls == [
        (
            quant_config,
            "model.layers.0.mlp.experts",
            "model.layers.0.mlp.shared_expert",
        )
    ]


def test_resolve_layer_fused_shared_expert_rejects_incompatible_quantization(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "is_shared_expert_quant_fse_compatible",
        lambda *_: (False, "shared experts are excluded"),
    )

    assert not fused_moe_utils.resolve_layer_fused_shared_expert(
        object(), "model.layers.0.mlp"
    )
    assert "shared experts are excluded" in caplog.text


def test_deepseek_v4_shared_expert_fse_uses_mtp_quantization_config_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeepseekV4Config:
        expert_dtype = "fp4"

        def _is_quark_mxfp4_ocp(self, hf_config: object) -> bool:
            return True

    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        quantization_config={
            "layer_quant_config": {
                r"re:mtp\.0\.ffn\.shared_experts\.w1": {"weight": {"dtype": "fp4"}}
            },
            "global_quant_config": {"weight": {"dtype": "fp8"}},
        },
    )
    monkeypatch.setattr(
        deepseek_v4_quant_config, "DeepseekV4FP8Config", DeepseekV4Config
    )
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config)),
    )

    compatible, reason = is_shared_expert_quant_fse_compatible(
        DeepseekV4Config(),
        "model.layers.2.ffn.experts",
        "model.layers.2.ffn.shared_experts",
    )

    assert compatible
    assert reason is None


def test_deepseek_v4_heterogeneous_fhmoe_keeps_native_intermediate_width() -> None:
    from vllm.models.deepseek_v4.amd.model import _prepare_native_fp8_shared_expert

    hidden_size = 7168
    intermediate_size = 384
    w13 = torch.empty((2 * intermediate_size, hidden_size), dtype=torch.float8_e4m3fn)
    w2 = torch.empty((hidden_size, intermediate_size), dtype=torch.float8_e4m3fn)
    w13_scale_bytes = (torch.arange(6 * 56).reshape(6, 56).remainder(20) + 0x60).to(
        torch.uint8
    )
    w2_scale_bytes = (torch.arange(56 * 3).reshape(56, 3).remainder(30) + 0x50).to(
        torch.uint8
    )
    w13_scale = w13_scale_bytes.view(torch.float8_e8m0fnu)
    w2_scale = w2_scale_bytes.view(torch.float8_e8m0fnu)

    prepared = _prepare_native_fp8_shared_expert(
        w13, w2, w13_scale, w2_scale, intermediate_size
    )

    assert prepared[0].shape == (1, 768, 7168)
    assert prepared[1].shape == (1, 7168, 384)
    assert prepared[2].shape == (768, 224)
    assert prepared[3].shape == (7168, 16)
    expected_w13_scale = w13_scale_bytes.repeat_interleave(
        128, dim=0
    ).repeat_interleave(4, dim=1)
    expected_w2_scale = w2_scale_bytes.repeat_interleave(128, dim=0).repeat_interleave(
        4, dim=1
    )
    assert torch.equal(prepared[2].view(torch.uint8), expected_w13_scale)
    assert torch.equal(prepared[3].view(torch.uint8)[:, :12], expected_w2_scale)
    assert torch.all(prepared[3].view(torch.uint8)[:, 12:] == 0x7F)


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (0, False),
        (1, True),
        (2, True),
        (8, True),
        (16, True),
        (1536, True),
        (2048, True),
        (2049, False),
    ],
)
def test_deepseek_v4_heterogeneous_fhmoe_token_policy(
    num_tokens: int, expected: bool
) -> None:
    from vllm.models.deepseek_v4.amd.model import _use_heterogeneous_fhmoe

    assert _use_heterogeneous_fhmoe(num_tokens) is expected


def _supported_fhmoe_signature(
    shared_w1=None,
    shared_w2=None,
    shared_w1_scale=None,
    shared_w2_scale=None,
    shared_expert_id=-1,
) -> None:
    pass


def _incomplete_fhmoe_signature(
    shared_w1=None,
    shared_w2=None,
    shared_w1_scale=None,
    shared_w2_scale=None,
) -> None:
    pass


def _install_fake_aiter_fhmoe(
    monkeypatch: pytest.MonkeyPatch,
    advertised_max_tokens: object,
    fused_moe: object,
) -> None:
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_fhmoe = ModuleType("aiter.fhmoe")
    fake_fhmoe.__dict__["DSV4_I384_FHMOE_MAX_TOKENS"] = advertised_max_tokens
    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.__dict__["fused_moe"] = fused_moe
    fake_aiter.__dict__["fhmoe"] = fake_fhmoe
    fake_aiter.__dict__["fused_moe"] = fake_fused_moe
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fhmoe", fake_fhmoe)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)


@pytest.mark.parametrize(
    ("required_max_tokens", "advertised_max_tokens", "fused_moe", "expected"),
    [
        (0, 2048, _supported_fhmoe_signature, False),
        (True, 2048, _supported_fhmoe_signature, False),
        (2048, None, _supported_fhmoe_signature, False),
        (2048, True, _supported_fhmoe_signature, False),
        (2048, 2047, _supported_fhmoe_signature, False),
        (2048, 2048, _incomplete_fhmoe_signature, False),
        (2048, 2048, _supported_fhmoe_signature, True),
        (2049, 2048, _supported_fhmoe_signature, False),
    ],
)
def test_deepseek_v4_heterogeneous_fhmoe_aiter_capability(
    monkeypatch: pytest.MonkeyPatch,
    required_max_tokens: object,
    advertised_max_tokens: object,
    fused_moe: object,
    expected: bool,
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    _install_fake_aiter_fhmoe(monkeypatch, advertised_max_tokens, fused_moe)

    assert (
        rocm_aiter_ops._probe_dsv4_i384_fhmoe_capability(cast(int, required_max_tokens))
        is expected
    )


def test_deepseek_v4_heterogeneous_fhmoe_aiter_capability_catches_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fhmoe", None)

    assert not rocm_aiter_ops._probe_dsv4_i384_fhmoe_capability(2048)


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_deepseek_v4_heterogeneous_fhmoe_aiter_capability_catches_signature_error(
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    _install_fake_aiter_fhmoe(monkeypatch, 2048, _supported_fhmoe_signature)

    def raise_signature_error(_: object) -> inspect.Signature:
        raise error

    monkeypatch.setattr(inspect, "signature", raise_signature_error)

    assert not rocm_aiter_ops._probe_dsv4_i384_fhmoe_capability(2048)


@pytest.mark.parametrize(
    ("setting", "value", "expected"),
    [
        ("data_parallel_size", 1, True),
        ("data_parallel_size", 2, False),
        ("prefill_context_parallel_size", 2, False),
        ("topk_method", "greedy", False),
    ],
)
def test_deepseek_v4_heterogeneous_fhmoe_compatibility_gates(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: object,
    expected: bool,
) -> None:
    from vllm.models.deepseek_v4.amd import model as deepseek_v4_model

    hf_config = SimpleNamespace(
        n_routed_experts=384,
        num_experts_per_tok=6,
        n_shared_experts=1,
        hidden_size=7168,
        moe_intermediate_size=3072,
        hidden_act="silu",
        expert_dtype="fp4",
        topk_method="noaux_tc",
    )
    parallel_config = SimpleNamespace(
        enable_expert_parallel=False,
        enable_eplb=False,
        tensor_parallel_size=8,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    if setting == "topk_method":
        hf_config.topk_method = value
    else:
        setattr(parallel_config, setting, value)

    quant_config = SimpleNamespace(
        get_name=lambda: "deepseek_v4_fp8",
        moe_quant_algo="",
        weight_block_size=[128, 128],
        is_checkpoint_fp8_serialized=True,
        is_scale_e8m0=True,
        ignored_layers=None,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16),
        quant_config=quant_config,
        parallel_config=parallel_config,
        kernel_config=SimpleNamespace(moe_backend="aiter"),
        offload_config=None,
    )
    monkeypatch.setattr(deepseek_v4_model.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        deepseek_v4_model.envs,
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
        True,
    )
    monkeypatch.setattr(deepseek_v4_model, "on_gfx950", lambda: True)
    monkeypatch.setattr(
        deepseek_v4_model.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        deepseek_v4_model.rocm_aiter_ops,
        "fused_moe_supports_heterogeneous_shared_expert",
        lambda _: True,
    )

    assert (
        deepseek_v4_model._heterogeneous_shared_expert_enabled(
            cast(VllmConfig, vllm_config)
        )
        is expected
    )


@pytest.mark.parametrize(
    ("weights_shape", "ids_shape", "message"),
    [
        ((2, 7, 1), (2, 7, 1), "equal two-dimensional shapes"),
        ((2, 7), (3, 7), "equal two-dimensional shapes"),
        ((2, 6), (2, 6), "exactly 7 columns"),
        ((2, 8), (2, 8), "exactly 7 columns"),
    ],
)
def test_deepseek_v4_heterogeneous_fhmoe_rejects_invalid_routes(
    weights_shape: tuple[int, ...],
    ids_shape: tuple[int, ...],
    message: str,
) -> None:
    from vllm.models.deepseek_v4.amd.model import _validate_heterogeneous_routes

    with pytest.raises(ValueError, match=message):
        _validate_heterogeneous_routes(
            torch.empty(weights_shape),
            torch.empty(ids_shape, dtype=torch.int64),
            experts_per_token=6,
        )


def test_deepseek_v4_heterogeneous_fhmoe_accepts_appended_shared_route() -> None:
    from vllm.models.deepseek_v4.amd.model import _validate_heterogeneous_routes

    _validate_heterogeneous_routes(
        torch.empty((2, 7)),
        torch.empty((2, 7), dtype=torch.int64),
        experts_per_token=6,
    )


def test_aiter_fused_moe_validates_shared_expert_arguments() -> None:
    from vllm._aiter_ops import (
        _validate_rocm_aiter_fused_moe_shared_expert_args,
    )

    validate = _validate_rocm_aiter_fused_moe_shared_expert_args
    shared_tensor = torch.empty(0)

    assert not validate(None, None, None, None, -1)
    with pytest.raises(ValueError, match="requires shared weights and scales"):
        validate(None, None, None, None, 0)
    with pytest.raises(ValueError, match="both shared weights and scales"):
        validate(shared_tensor, None, None, None, 0)
    with pytest.raises(ValueError, match="non-negative shared expert ID"):
        validate(shared_tensor, shared_tensor, shared_tensor, shared_tensor, -1)
    assert validate(shared_tensor, shared_tensor, shared_tensor, shared_tensor, 0)


def test_is_model_fused_shared_expert_compatible() -> None:
    class MoE(nn.Module):
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.is_fused_shared_expert_enabled = enabled

    class Layer(nn.Module):
        def __init__(self, enabled: bool) -> None:
            super().__init__()
            self.mlp = MoE(enabled)

    enabled_layers = nn.ModuleList([Layer(True)])
    disabled_layers = nn.ModuleList([Layer(False)])
    mixed_layers = nn.ModuleList([Layer(True), Layer(False)])
    empty_layers = nn.ModuleList()
    pipeline_layers = nn.ModuleList([Layer(True), PPMissingLayer()])

    assert fused_moe_utils.is_model_fused_shared_expert_compatible(
        enabled_layers, MoE, "mlp"
    )
    assert not fused_moe_utils.is_model_fused_shared_expert_compatible(
        disabled_layers, MoE, "mlp"
    )
    assert not fused_moe_utils.is_model_fused_shared_expert_compatible(
        empty_layers, MoE, "mlp"
    )
    assert fused_moe_utils.is_model_fused_shared_expert_compatible(
        pipeline_layers, MoE, "mlp"
    )
    with pytest.raises(NotImplementedError, match="1 enabled and 1 disabled layers"):
        fused_moe_utils.is_model_fused_shared_expert_compatible(
            mixed_layers, MoE, "mlp"
        )


@pytest.mark.parametrize(
    "model_type",
    ["minimax_m3", "deepseek_v4", "qwen3_5", "glm4_moe", "deepseek_v2"],
)
@pytest.mark.parametrize(
    ("use_fse", "exclude"),
    [
        (False, []),
        (True, []),
        (True, ["*.shared_experts.*"]),
    ],
)
def test_models_fse_init(
    model_type: str,
    use_fse: bool,
    exclude: list[str],
    dist_init: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model construction resolves FSE consistently with Quark quantization."""

    quantization_config: dict[str, Any] = (
        get_deepseek_v4_quark_config(["layers.0.ffn.shared_experts"] if exclude else [])
        if model_type == "deepseek_v4"
        else {**_QUARK_FSE_CONFIG, "exclude": exclude}
    )

    config, model_constructor = get_fse_test_model_config(
        model_type, quantization_config
    )
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        dtype=torch.bfloat16,
        max_model_len=128,
        is_diffusion=False,
        head_dtype=None,
        is_mm_prefix_lm=False,
        multimodal_config=None,
        quantization_config=None,
        runner_type="generate",
        is_moe=True,
    )
    vllm_config.parallel_config.enable_expert_parallel = False
    if model_type == "deepseek_v4":
        from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

        vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
        vllm_config.quant_config = DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
        )
    else:
        vllm_config.quant_config = QuarkConfig(quantization_config)

    import vllm.envs as envs
    from vllm._aiter_ops import rocm_aiter_ops

    if model_type == "deepseek_v4":
        from vllm.models.deepseek_v4.amd import model as deepseek_v4_model

        warning_logger = deepseek_v4_model.logger
    elif model_type == "minimax_m3":
        warning_logger = minimax_m3_model.logger
    else:
        warning_logger = fused_moe_utils.logger

    with monkeypatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", str(use_fse))
        mp.setenv("VLLM_ROCM_USE_AITER_MOE", str(use_fse))
        mp.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", str(use_fse))
        mp.setattr(
            "vllm.model_executor.layers.fused_moe.experts."
            "ocp_mx_emulation_moe.has_quark",
            lambda: True,
        )
        importlib.reload(envs)
        mp.setattr(envs, "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", use_fse)
        rocm_aiter_ops.refresh_env_variables()
        aiter_fse_enabled = bool(rocm_aiter_ops.is_fusion_moe_shared_experts_enabled())
        # These AMD-specific models currently use the raw environment flag.
        # and do not rely on `rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()`.
        fse_enabled = use_fse and (
            aiter_fse_enabled or model_type in {"deepseek_v4", "minimax_m3"}
        )

        with (
            patch.object(warning_logger, "warning") as warning,
            set_current_vllm_config(vllm_config),
        ):
            model = model_constructor(vllm_config=vllm_config)
            mtp = None
            if model_type == "deepseek_v4" and use_fse and not exclude:
                from vllm.models.deepseek_v4.amd.mtp import (
                    DeepSeekV4MTP,
                )

                vllm_config.speculative_config = SimpleNamespace(
                    draft_model_config=SimpleNamespace(hf_config=config),
                    method="mtp",
                )
                mtp = DeepSeekV4MTP(vllm_config=vllm_config)
        assert model.is_fused_shared_expert_enabled is (fse_enabled and not exclude)

        # The dummy quant config here uses mixed mxfp4/fp8 for experts/shared_expert
        # so should just raise a warning.
        if mtp is not None:
            assert not mtp.model.layers[
                "1"
            ].mtp_block.ffn.is_fused_shared_expert_enabled
            warning.assert_called_once()
            assert (
                "DeepSeek-V4 shared experts at mtp.0.ffn.shared_experts"
                in (warning.call_args.args[1])
            )
        if aiter_fse_enabled and exclude:
            warning.assert_called_once()
            assert (
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
                in (warning.call_args.args[0])
            )
            assert "excludes shared experts" in warning.call_args.args[1]

    importlib.reload(envs)
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    ("exclude", "expected"),
    [([], True), (["*.shared_expert.*"], False)],
)
def test_quark_shared_expert_fse_compatibility(
    exclude: list[str], expected: bool
) -> None:
    compatible, reason = is_shared_expert_quant_fse_compatible(
        QuarkConfig(
            {
                "exclude": exclude,
                "global_quant_config": {},
                "layer_quant_config": {},
            }
        ),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert",
    )

    assert compatible is expected
    if expected:
        assert reason is None
    else:
        assert (
            reason
            == "Quark excludes shared experts at model.layers.0.mlp.shared_expert"
        )


def test_quark_shared_expert_fse_requires_matching_layer_quant_configs() -> None:
    global_quant_config = {"weight": {"dtype": "fp4"}}
    quant_config = QuarkConfig(
        {
            "exclude": [],
            "global_quant_config": global_quant_config,
            "layer_quant_config": {
                "model.layers.0.mlp.shared_expert.gate_up_proj": {
                    "weight": {"dtype": "fp8"}
                },
                "model.layers.0.mlp.shared_expert.down_proj": {
                    "weight": {"dtype": "fp8"}
                },
            },
        }
    )

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert",
    )

    assert not compatible
    assert reason == (
        "Quark uses different quantization configurations for routed and "
        "shared experts at model.layers.0.mlp.shared_expert"
    )


def test_quark_shared_expert_fse_rejects_partial_packed_projection_override() -> None:
    fp4_config = {"weight": {"dtype": "fp4"}}
    fp8_config = {"weight": {"dtype": "fp8"}}
    quant_config = QuarkConfig(
        {
            "exclude": [],
            "global_quant_config": fp4_config,
            "layer_quant_config": {
                "model.layers.0.mlp.experts": fp8_config,
                "model.layers.0.mlp.shared_expert.gate_proj": fp8_config,
                "model.layers.0.mlp.shared_expert.down_proj": fp8_config,
            },
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_expert",
    )

    assert not compatible
    assert reason == (
        "Quark uses different quantization configurations for routed and "
        "shared experts at model.layers.0.mlp.shared_expert"
    )


def test_quark_layer_config_from_name_checks_packed_projections() -> None:
    quant_config = QuarkConfig(
        {
            "global_quant_config": {"weight": {"dtype": "fp4"}},
            "layer_quant_config": {
                "model.layers.0.mlp.shared_expert.w1": {"weight": {"dtype": "fp4"}},
                "model.layers.0.mlp.shared_expert.w3": {"weight": {"dtype": "fp8"}},
            },
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}

    with pytest.raises(ValueError, match="requires all to use the same scheme"):
        quant_config.get_layer_quant_config_from_name(
            "model.layers.0.mlp.shared_expert.gate_up_proj"
        )


def test_quark_packed_layer_config_must_match_global_config() -> None:
    quant_config = QuarkConfig(
        {
            "global_quant_config": {"weight": {"dtype": "fp4"}},
            "layer_type_quant_config": {},
            "layer_quant_config": {
                "model.layers.0.mlp.shared_expert.w1": {"weight": {"dtype": "fp8"}},
            },
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["w1", "w3"]}

    with pytest.raises(ValueError, match="requires all to use the same scheme"):
        quant_config._find_matched_config(
            "model.layers.0.mlp.shared_expert.gate_up_proj", nn.Module()
        )


def test_non_quark_shared_expert_fse_is_incompatible() -> None:
    compatible, reason = is_shared_expert_quant_fse_compatible(
        object(),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert not compatible
    assert reason == (
        "shared-expert FSE quantization compatibility is not implemented for object"
    )
