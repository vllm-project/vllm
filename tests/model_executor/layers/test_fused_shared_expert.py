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
from transformers import Qwen3_5MoeTextConfig

import vllm.config as vllm_config_module
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.config.quantization import QuantizationConfigArgs
from vllm.logger import _print_warning_once
from vllm.model_executor.layers.fused_moe import utils as fused_moe_utils
from vllm.model_executor.layers.fused_moe.layer import determine_expert_counts
from vllm.model_executor.layers.quantization.experts_int8 import ExpertsInt8Config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_quark_ocp_mx_group_size,
    is_shared_expert_quant_fse_compatible,
)
from vllm.model_executor.models.qwen3_next import (
    _should_replicate_misaligned_shared_expert,
)
from vllm.model_executor.models.utils import (
    get_fused_shared_expert_layers,
    maybe_fuse_shared_experts,
)
from vllm.models.deepseek_v4 import quant_config as deepseek_v4_quant_config
from vllm.models.minimax_m3.amd import model as minimax_m3_model
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

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


class _StubGlm5NextAttention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.o_proj = SimpleNamespace(reduce_results=True)
        # GLM5Next's MTP predictor reads this attribute from MLA layers.
        self.mla_attn = SimpleNamespace()


def get_fse_test_model_config(
    model_type: str,
    quantization_config: dict[str, Any],
) -> tuple[Any, type[nn.Module]]:
    config: Any
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
            moe_layer_freq=[1, 1],
            sparse_attention_config=None,
            rotary_dim=64,
            quantization_config=quantization_config,
        )
        return config, minimax_m3_model.MiniMaxM3Model
    if model_type in ("deepseek_v4", "deepseek_v4_cpu"):
        if model_type == "deepseek_v4":
            from vllm.models.deepseek_v4.amd.model import DeepseekV4Model
        else:
            from vllm.models.deepseek_v4.cpu.model import DeepseekV4Model

        config = SimpleNamespace(
            model_type="deepseek_v4",
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            head_dim=128,
            max_position_embeddings=128,
            sliding_window=None,
            layer_types=["sliding_attention", "sliding_attention"],
            mlp_layer_types=[],
            compress_rates={},
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
    if model_type in ("qwen3_next", "qwen3_5", "alice_ai"):
        from transformers import Qwen3NextConfig

        common = dict(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=2,
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
            layer_types=["full_attention", "full_attention"],
            quantization_config=quantization_config,
        )
        if model_type == "qwen3_next":
            from vllm.model_executor.models.qwen3_next import Qwen3NextModel

            return Qwen3NextConfig(**common), Qwen3NextModel
        if model_type == "qwen3_5":
            from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

            return Qwen3_5MoeTextConfig(**common), Qwen3_5Model
        from vllm.model_executor.models.alice_ai import AliceAIModel

        config = Qwen3NextConfig(**common, linear_conv_kernel_dim=4)
        # AliceAI requires the AttnRes grouping metadata.
        config.block_attn_res_block_size = 1
        return config, AliceAIModel
    if model_type == "qwen4_exp":
        from transformers import Qwen4ExpTextConfig

        config = Qwen4ExpTextConfig(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=2,
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
            layer_types=["full_attention", "full_attention"],
            eos_token_id=1,
            hc_count=2,
            hc_lowrank=4,
            ple_layer_ids=[],
            quantization_config=quantization_config,
        )
        from vllm.models.qwen4_exp.amd.model import Qwen4ExpModel

        return config, Qwen4ExpModel
    if model_type == "AXK1":
        from transformers import AXK1Config

        from vllm.model_executor.models.AXK1 import AXK1Model

        config = AXK1Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            n_shared_experts=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=0,
            q_lora_rank=0,
            kv_lora_rank=0,
            qk_nope_head_dim=0,
            qk_rope_head_dim=0,
            v_head_dim=0,
            quantization_config=quantization_config,
        )
        return config, AXK1Model
    if model_type == "glm4_moe_lite":
        from transformers.models.glm4_moe_lite import Glm4MoeLiteConfig

        from vllm.model_executor.models.glm4_moe_lite import Glm4MoeLiteModel

        config = Glm4MoeLiteConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            q_lora_rank=0,
            kv_lora_rank=32,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        config.first_k_dense_replace = 0
        config.num_nextn_predict_layers = 1
        return config, Glm4MoeLiteModel
    if model_type in ("deepseek_v2", "deepseek_v32"):
        from transformers import DeepseekV2Config

        common = dict(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=32,
            num_hidden_layers=2,
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
        config = DeepseekV2Config(**common)
        if model_type == "deepseek_v2":
            from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

            # Needed to build the DeepSeek-V2 MTP draft model in this test.
            config.num_nextn_predict_layers = 1
            return config, DeepseekV2Model
        from vllm.models.deepseek_v32.amd.model import DeepseekV32Model

        config.index_topk = 1
        config.num_nextn_predict_layers = 1
        return config, DeepseekV32Model
    if model_type == "glm4_moe":
        from transformers.models.glm4_moe import Glm4MoeConfig

        from vllm.model_executor.models.glm4_moe import Glm4MoeModel

        config = Glm4MoeConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=32,
            num_hidden_layers=2,
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
    if model_type == "glm5_next":
        from transformers import Glm5NextTextConfig

        from vllm.models.glm5next.common.model import Glm5NextModel

        config = Glm5NextTextConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=32,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=1,
            layer_types=["linear_attention", "linear_attention"],
            mlp_layer_types=["sparse", "sparse"],
            index_topk=1,
            index_kpool=1,
            pad_token_id=None,
            scoring_func="sigmoid",
            topk_method="noaux_tc",
            mhc=False,
            quantization_config=quantization_config,
        )
        config.mla_use_nope = False
        config.num_nextn_predict_layers = 1
        return config, Glm5NextModel
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
        ExpertsInt8Config(), "model.layers.0.mlp"
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
            ExpertsInt8Config(), "model.layers.0.mlp"
        )
        is False
    )


def test_resolve_layer_fused_shared_expert_passes_module_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_config = ExpertsInt8Config()
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )

    calls: list[tuple[object, str, str, str | None]] = []

    def check_compatibility(
        config: object,
        expert_prefix: str,
        shared_expert_prefix: str,
        moe_backend: str | None = None,
    ) -> tuple[bool, str | None]:
        calls.append((config, expert_prefix, shared_expert_prefix, moe_backend))
        return True, None

    monkeypatch.setattr(
        fused_moe_utils, "is_shared_expert_quant_fse_compatible", check_compatibility
    )

    assert fused_moe_utils.resolve_layer_fused_shared_expert(
        quant_config,
        "model.layers.0.mlp",
        shared_expert_name="shared_expert",
        moe_backend="aiter_mega_moe",
    )
    assert calls == [
        (
            quant_config,
            "model.layers.0.mlp.experts",
            "model.layers.0.mlp.shared_expert",
            "aiter_mega_moe",
        )
    ]


@pytest.mark.parametrize(
    ("reason", "enable_ep"),
    [
        ("shared experts are excluded", False),
        ("expert parallelism is enabled", True),
    ],
)
def test_resolve_layer_fused_shared_expert_logs_skip_reason(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reason: str,
    enable_ep: bool,
) -> None:
    monkeypatch.setattr(
        fused_moe_utils.rocm_aiter_ops,
        "is_fusion_moe_shared_experts_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        fused_moe_utils,
        "is_shared_expert_quant_fse_compatible",
        lambda *_, **__: (False, reason),
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=enable_ep)
        ),
    )
    _print_warning_once.cache_clear()

    assert not fused_moe_utils.resolve_layer_fused_shared_expert(
        ExpertsInt8Config(), "model.layers.0.mlp", disable_under_ep=enable_ep
    )
    assert reason in caplog.text


def test_deepseek_v4_shared_expert_fse_uses_mtp_quantization_config_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeepseekV4Config:
        expert_dtype = "fp4"
        online_quantization_config = None

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


def test_deepseek_v41_quant_config_rejects_fp8_mtp_shared_expert_fse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.deepseek_v41 import quant_config as deepseek_v41_quant_config

    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        expert_dtype="fp4",
        quantization_config={
            "quant_method": "quark",
            "global_quant_config": {
                "weight": {"dtype": "fp4", "qscheme": "per_group", "group_size": 32}
            },
            "layer_quant_config": {
                r"re:mtp\.0\.ffn\.shared_experts\.w1": {"weight": {"dtype": "fp8_e4m3"}}
            },
            "exclude": [],
        },
    )
    current_vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config)
    )
    monkeypatch.setattr(
        vllm_config_module, "get_current_vllm_config", lambda: current_vllm_config
    )
    monkeypatch.setattr(
        deepseek_v41_quant_config,
        "get_current_vllm_config",
        lambda: current_vllm_config,
    )
    quant_config = deepseek_v41_quant_config.DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.2.ffn.experts",
        "model.layers.2.ffn.shared_experts",
    )

    assert not compatible
    assert (
        reason
        == "DeepSeek-V4/V4.1 shared experts at mtp.0.ffn.shared_experts are not MXFP4"
    )


def test_deepseek_v41_fp8_experts_keep_fp8_fse_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.deepseek_v41 import quant_config as deepseek_v41_quant_config

    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        expert_dtype="fp8",
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    )
    current_vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config)
    )
    monkeypatch.setattr(
        vllm_config_module, "get_current_vllm_config", lambda: current_vllm_config
    )
    monkeypatch.setattr(
        deepseek_v41_quant_config,
        "get_current_vllm_config",
        lambda: current_vllm_config,
    )
    quant_config = deepseek_v41_quant_config.DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )

    assert is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.2.ffn.experts",
        "model.layers.2.ffn.shared_experts",
    ) == (True, None)


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
    ("num_tokens", "supported_through", "expected"),
    [
        (0, 4096, False),
        (1, 4096, True),
        (1536, 4096, True),
        (2048, 4096, True),
        (2049, 4096, True),
        (4096, 4096, True),
        (4097, 4096, False),
        (1536, 0, False),
    ],
)
def test_deepseek_v4_heterogeneous_fhmoe_token_policy(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    supported_through: int,
    expected: bool,
) -> None:
    from vllm.models.deepseek_v4.amd import model as deepseek_v4_model

    checked_tokens: list[int] = []

    def supports(num_tokens: int) -> bool:
        checked_tokens.append(num_tokens)
        return num_tokens <= supported_through

    monkeypatch.setattr(
        deepseek_v4_model.rocm_aiter_ops,
        "fused_moe_supports_heterogeneous_shared_expert",
        supports,
    )

    assert deepseek_v4_model._use_heterogeneous_fhmoe(num_tokens) is expected
    assert checked_tokens == ([] if num_tokens == 0 else [num_tokens])


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
    supports_dsv4_i384_fhmoe: object,
    fused_moe: object,
) -> None:
    fake_aiter = ModuleType("aiter")
    fake_aiter.__path__ = []
    fake_fhmoe = ModuleType("aiter.fhmoe")
    if supports_dsv4_i384_fhmoe is not None:
        fake_fhmoe.__dict__["supports_dsv4_i384_fhmoe"] = supports_dsv4_i384_fhmoe
    fake_fused_moe = ModuleType("aiter.fused_moe")
    fake_fused_moe.__dict__["fused_moe"] = fused_moe
    fake_aiter.__dict__["fhmoe"] = fake_fhmoe
    fake_aiter.__dict__["fused_moe"] = fake_fused_moe
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setitem(sys.modules, "aiter.fhmoe", fake_fhmoe)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fake_fused_moe)


def _supports_fhmoe_through_2047(max_tokens: int) -> bool:
    return max_tokens <= 2047


def _supports_fhmoe_through_2048(max_tokens: int) -> bool:
    return max_tokens <= 2048


def _supports_fhmoe_through_4096(max_tokens: int) -> bool:
    return max_tokens <= 4096


def _raises_fhmoe_config_error(max_tokens: int) -> bool:
    raise OSError


def _returns_truthy_non_bool(max_tokens: int) -> int:
    return 1


@pytest.mark.parametrize(
    ("num_tokens", "capability", "fused_moe", "expected"),
    [
        (0, _supports_fhmoe_through_2048, _supported_fhmoe_signature, False),
        (True, _supports_fhmoe_through_2048, _supported_fhmoe_signature, False),
        (2048, None, _supported_fhmoe_signature, False),
        (2048, True, _supported_fhmoe_signature, False),
        (2048, _raises_fhmoe_config_error, _supported_fhmoe_signature, False),
        (2048, _returns_truthy_non_bool, _supported_fhmoe_signature, False),
        (2048, _supports_fhmoe_through_2047, _supported_fhmoe_signature, False),
        (2048, _supports_fhmoe_through_2048, _incomplete_fhmoe_signature, False),
        (2048, _supports_fhmoe_through_2048, _supported_fhmoe_signature, True),
        (2049, _supports_fhmoe_through_2048, _supported_fhmoe_signature, False),
        (4096, _supports_fhmoe_through_4096, _supported_fhmoe_signature, True),
        (4097, _supports_fhmoe_through_4096, _supported_fhmoe_signature, False),
    ],
)
def test_deepseek_v4_heterogeneous_fhmoe_aiter_capability(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: object,
    capability: object,
    fused_moe: object,
    expected: bool,
) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    _install_fake_aiter_fhmoe(monkeypatch, capability, fused_moe)

    assert (
        rocm_aiter_ops._probe_dsv4_i384_fhmoe_capability(cast(int, num_tokens))
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

    _install_fake_aiter_fhmoe(
        monkeypatch, _supports_fhmoe_through_2048, _supported_fhmoe_signature
    )

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
        ("fhmoe_supported", False, False),
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
    elif setting != "fhmoe_supported":
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
    checked_tokens: list[int] = []
    fhmoe_supported = setting != "fhmoe_supported" or value is True

    def supports(num_tokens: int) -> bool:
        checked_tokens.append(num_tokens)
        return fhmoe_supported

    monkeypatch.setattr(
        deepseek_v4_model.rocm_aiter_ops,
        "fused_moe_supports_heterogeneous_shared_expert",
        supports,
    )

    assert (
        deepseek_v4_model._heterogeneous_shared_expert_enabled(
            cast(VllmConfig, vllm_config)
        )
        is expected
    )
    assert checked_tokens == [1]


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


def test_deepseek_v4_shared_expert_mapper_fallback() -> None:
    """Mixed and all-excluded DSV4 checkpoints map shared experts correctly."""
    from vllm.models.deepseek_v4.amd.model import (
        _make_deepseek_v4_weights_mapper,
        _redirect_shared_expert_name,
    )

    tensor = torch.empty(0)
    fuse_by_layer = frozenset({6})
    # Mixed: any fused layer puts the mapper in fused mode, which leaves the
    # checkpoint .w2 names untouched; the loader redirects per layer.
    mapper = _make_deepseek_v4_weights_mapper("fp8", fuse_shared_experts=True)
    for raw, expected in [
        (
            "layers.5.ffn.shared_experts.w2.weight",
            "model.layers.5.ffn.shared_experts.down_proj.weight",
        ),
        (
            "layers.5.ffn.shared_experts.w2.scale",
            "model.layers.5.ffn.shared_experts.down_proj.weight_scale_inv",
        ),
        (
            "layers.6.ffn.shared_experts.w2.weight",
            "model.layers.6.ffn.experts.4.w2.weight",
        ),
        (
            "layers.6.ffn.shared_experts.w2.scale",
            "model.layers.6.ffn.experts.4.w2.weight_scale_inv",
        ),
    ]:
        [(mapped, _)] = list(mapper.apply([(raw, tensor)]))
        assert _redirect_shared_expert_name(mapped, fuse_by_layer, 4) == expected

    # All layers excluded (fused_shared_expert_layers == frozenset()) selects
    # the unfused mapper, which renames .w2 -> .down_proj itself.
    mapper = _make_deepseek_v4_weights_mapper("fp8", fuse_shared_experts=False)
    for raw, expected in [
        (
            "layers.5.ffn.shared_experts.w2.weight",
            "model.layers.5.ffn.shared_experts.down_proj.weight",
        ),
        (
            "layers.5.ffn.shared_experts.w2.scale",
            "model.layers.5.ffn.shared_experts.down_proj.weight_scale_inv",
        ),
    ]:
        [(mapped, _)] = list(mapper.apply([(raw, tensor)]))
        assert _redirect_shared_expert_name(mapped, frozenset(), 4) == expected


class _FakeMoE(nn.Module):
    def __init__(self, enabled: bool) -> None:
        super().__init__()
        self.is_fused_shared_expert_enabled = enabled


def _make_fake_moe_model(
    layer_names: list[str], enabled: list[bool], use_dict: bool
) -> nn.Module:
    layers = []
    for flag in enabled:
        layer = nn.Module()
        layer.mlp = _FakeMoE(flag)
        layers.append(layer)
    model = nn.Module()
    model.layers = (
        nn.ModuleDict(dict(zip(layer_names, layers, strict=True)))
        if use_dict
        else nn.ModuleList(layers)
    )
    return model


def test_fused_shared_expert_layers_matches_container_index_space() -> None:
    """The fused-layer set follows the module tree, not a hard-coded offset.

    ``nn.ModuleList`` MTP layers (Qwen3 Next/3.5, Qwen4 Exp) are 0-based,
    while ``nn.ModuleDict`` MTP layers (GLM4, DeepSeek) are keyed by absolute
    layer index. Both must redirect without a per-loader offset.
    """
    list_model = _make_fake_moe_model(["0", "1"], [True, False], use_dict=False)
    dict_model = _make_fake_moe_model(["28", "29"], [False, True], use_dict=True)
    list_layers = get_fused_shared_expert_layers(list_model, _FakeMoE)
    dict_layers = get_fused_shared_expert_layers(dict_model, _FakeMoE)
    assert list_layers == frozenset({0})
    assert dict_layers == frozenset({29})

    weight = torch.arange(8.0).reshape(4, 2)

    def redirected(name: str, fused_shared_expert_layers: frozenset[int]) -> bool:
        [(out_name, _)] = list(
            maybe_fuse_shared_experts(
                [(name, weight)],
                n_routed_experts=4,
                n_shared_experts=1,
                fused_shared_expert_layers=fused_shared_expert_layers,
                ckpt_prefix="mlp.shared_expert",
            )
        )
        return ".experts." in out_name

    # ModuleList: 0-based names, including the Qwen4 Exp MTP case where the
    # quant-config prefix uses an absolute index but the weight names do not.
    assert redirected("layers.0.mlp.shared_expert.down_proj.weight", list_layers)
    assert not redirected("layers.1.mlp.shared_expert.down_proj.weight", list_layers)
    # ModuleDict: absolute names, including the MTP layer offset.
    assert redirected("layers.29.mlp.shared_expert.down_proj.weight", dict_layers)
    assert not redirected("layers.28.mlp.shared_expert.down_proj.weight", dict_layers)


@pytest.mark.parametrize(
    ("enable_expert_parallel", "match"),
    [
        (True, "expert parallelism is not tested"),
        (False, "nixl_ep"),
    ],
)
def test_mixed_fused_shared_experts_rejected_with_ep_or_nixl_ep(
    enable_expert_parallel: bool, match: str
) -> None:
    """Mixed FSE must be rejected with EP or a shared-workspace all2all backend.

    With EP, per-layer geometry differences are not validated. ``nixl_ep``
    additionally keeps a single process-wide buffer initialized from the first
    MoE layer's ``num_experts_per_rank`` and ignores later geometries. A mixed
    model has fused layers with ``n_routed + n_shared`` experts and non-fused
    layers with ``n_routed``, so reusing that buffer would silently map experts
    to the wrong slots. Construction must fail cleanly instead of running the
    untested mixed path.
    """
    model = _make_fake_moe_model(["0", "1"], [True, False], use_dict=False)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.enable_expert_parallel = enable_expert_parallel
    vllm_config.parallel_config.data_parallel_size = 2
    vllm_config.parallel_config.all2all_backend = "nixl_ep"

    with (
        set_current_vllm_config(vllm_config),
        pytest.raises(NotImplementedError, match=match),
    ):
        get_fused_shared_expert_layers(model, _FakeMoE)


def _assert_mtp_fused_shared_expert_redirect(
    mtp: nn.Module, ckpt_prefix: str, moe: nn.Module, n_routed_experts: int
) -> None:
    """Load a fused MTP shared expert and verify it lands in the appended slot.

    The MTP loaders see different name/index spaces (0-based ``ModuleList`` or
    absolute ``ModuleDict``); using the live module map must redirect either.
    """
    packed_down = torch.full((128, 16), 3, dtype=torch.uint8)
    down_scale = torch.full((128, 1), 127, dtype=torch.uint8)
    mtp.load_weights(
        [
            (f"{ckpt_prefix}.down_proj.weight", packed_down),
            (f"{ckpt_prefix}.down_proj.weight_scale", down_scale),
        ]
    )
    torch.testing.assert_close(
        moe.experts.routed_experts.w2_weight[n_routed_experts][:128, :16].float(),
        packed_down.float(),
    )


def _assert_mtp_shared_expert_not_fused(
    mtp: nn.Module, ckpt_prefix: str, moe: nn.Module, n_routed_experts: int
) -> None:
    """Load an unfused MTP shared expert and verify no fused slot is appended.

    Without AITER the shared expert stays a separate MLP, so the routed tensor must
    keep exactly ``n_routed_experts`` rows.
    """
    packed_down = torch.full((128, 16), 3, dtype=torch.uint8)
    down_scale = torch.full((128, 1), 127, dtype=torch.uint8)
    mtp.load_weights(
        [
            (f"{ckpt_prefix}.down_proj.weight", packed_down),
            (f"{ckpt_prefix}.down_proj.weight_scale", down_scale),
        ]
    )
    assert moe.experts.routed_experts.w2_weight.shape[0] == n_routed_experts


def _make_deepseek_v4_shared_expert_checkpoint(
    layer: nn.Module, idx: int, fused: bool, exclude: bool
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]]:
    """Build DSV4's FP8 shared-expert / MXFP4 routed-slot checkpoint pairs."""
    ffn = layer.ffn
    if not fused:
        dtype = torch.bfloat16 if exclude else torch.float8_e4m3fn
        hidden = ffn.shared_experts.gate_up_proj.weight.shape[1]
        intermediate = ffn.shared_experts.down_proj.weight.shape[1]
        gate_up = torch.full((2 * intermediate, hidden), idx + 1, dtype=dtype)
        down = torch.full((hidden, intermediate), idx + 1, dtype=dtype)
        return (
            [
                (f"layers.{idx}.ffn.shared_experts.gate_up_proj.weight", gate_up),
                (f"layers.{idx}.ffn.shared_experts.down_proj.weight", down),
            ],
            [
                (ffn.shared_experts.gate_up_proj.weight, gate_up),
                (ffn.shared_experts.down_proj.weight, down),
            ],
        )

    routed = ffn.experts.routed_experts
    slot = routed.w13_weight.shape[0] - 1
    w13_rows, hidden_half = routed.w13_weight[slot].shape
    intermediate = w13_rows // 2
    hidden = hidden_half * 2
    w1 = torch.full((intermediate, hidden_half), idx + 1, dtype=torch.uint8)
    w3 = torch.full((intermediate, hidden_half), idx + 1, dtype=torch.uint8)
    w2 = torch.full((hidden, intermediate // 2), idx + 1, dtype=torch.uint8)
    return (
        [
            (f"layers.{idx}.ffn.shared_experts.w1.weight", w1),
            (f"layers.{idx}.ffn.shared_experts.w3.weight", w3),
            (f"layers.{idx}.ffn.shared_experts.w2.weight", w2),
        ],
        [
            (routed.w13_weight[slot].chunk(2, dim=0)[0], w1),
            (routed.w13_weight[slot].chunk(2, dim=0)[1], w3),
            (routed.w2_weight[slot], w2),
        ],
    )


def _build_mtp_model(
    mtp_cls: Any,
    vllm_config: VllmConfig,
    config: object,
    *,
    with_text_config: bool = False,
    with_target_model: bool = False,
) -> nn.Module:
    draft_model_config: dict[str, object] = {"hf_config": config}
    if with_text_config:
        draft_model_config["hf_text_config"] = config
    speculative_config: dict[str, object] = {
        "draft_model_config": SimpleNamespace(**draft_model_config),
        "method": "mtp",
        "parallel_drafting": False,
        "enable_adaptive_verification": False,
        "use_dspark": lambda: False,
    }
    if with_target_model:
        speculative_config["target_model_config"] = vllm_config.model_config
    vllm_config.speculative_config = SimpleNamespace(**speculative_config)
    return mtp_cls(vllm_config=vllm_config)


_MTP_FACTORIES: dict[str, tuple[str, str, dict[str, bool]]] = {
    "deepseek_v4": ("vllm.models.deepseek_v4.amd.mtp", "DeepSeekV4MTP", {}),
    "glm4_moe": (
        "vllm.model_executor.models.glm4_moe_mtp",
        "Glm4MoeMTP",
        {},
    ),
    "glm4_moe_lite": (
        "vllm.model_executor.models.glm4_moe_lite_mtp",
        "Glm4MoeLiteMTP",
        {},
    ),
    "deepseek_v32": (
        "vllm.models.deepseek_v32.amd.mtp",
        "DeepseekV32MTP",
        {},
    ),
    "glm5_next": (
        "vllm.models.glm5next.common.mtp",
        "Glm5NextMTP",
        {"with_text_config": True},
    ),
    "deepseek_v2": (
        "vllm.model_executor.models.deepseek_mtp",
        "DeepSeekMTP",
        {},
    ),
    "qwen3_next": (
        "vllm.model_executor.models.qwen3_next_mtp",
        "Qwen3NextMTP",
        {"with_text_config": True, "with_target_model": True},
    ),
    "qwen3_5": (
        "vllm.model_executor.models.qwen3_5_mtp",
        "Qwen3_5MoeMTP",
        {"with_text_config": True, "with_target_model": True},
    ),
}


def _build_mtp_for_model(
    model_type: str,
    vllm_config: VllmConfig,
    config: object,
    enabled: bool,
) -> nn.Module | None:
    factory = _MTP_FACTORIES.get(model_type)
    if not enabled or factory is None:
        return None
    module_name, class_name, kwargs = factory
    module = importlib.import_module(module_name)
    return _build_mtp_model(getattr(module, class_name), vllm_config, config, **kwargs)


# TODO: move to conftest.py.
class _SingleRankGroup:
    """No-op single-rank stand-in for a compiled ``GroupCoordinator``."""

    world_size = 1
    rank_in_group = 0
    local_rank = 0
    is_first_rank = True
    is_last_rank = True

    @property
    def device_group(self):
        return self

    @property
    def cpu_group(self):
        return self

    @property
    def device_communicator(self):
        return self

    def rank(self) -> int:
        return 0

    def size(self) -> int:
        return 1

    def all_reduce(self, input_: torch.Tensor, *args: Any, **kwargs: Any):
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = 0):
        return input_

    def all_gatherv(self, inputs: Any, dim: int = 0, sizes: Any = None):
        return inputs

    def reduce_scatter(self, input_: torch.Tensor, dim: int = 0):
        return input_

    def reduce_scatterv(self, input_: Any, dim: int = 0, sizes: Any = None):
        return input_

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        return input_

    def send(self, *args: Any, **kwargs: Any) -> None:
        return None

    def recv(self, *args: Any, **kwargs: Any) -> None:
        return None

    def barrier(self) -> None:
        return None

    def destroy(self) -> None:
        return None


@pytest.fixture
def single_rank_parallel_state(monkeypatch: pytest.MonkeyPatch):
    """Single rank parallel groups without initializing a process group.

    This is faster than the `dist_init` fixture.
    """
    for name in ("_WORLD", "_TP", "_ETP", "_DCP", "_PP", "_DP", "_EP", "_PCP"):
        monkeypatch.setattr(
            f"vllm.distributed.parallel_state.{name}", _SingleRankGroup()
        )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )


@pytest.mark.parametrize(
    "model_type",
    [
        "minimax_m3",
        "deepseek_v4",
        "deepseek_v4_cpu",
        "qwen3_next",
        "qwen3_5",
        "alice_ai",
        "glm4_moe",
        "deepseek_v2",
        "AXK1",
        "glm4_moe_lite",
        "deepseek_v32",
        "glm5_next",
        "qwen4_exp",
    ],
)
@pytest.mark.parametrize("shared_expert_precision", ["all_mxfp4", "mixed"])
@pytest.mark.parametrize(
    ("use_fse", "exclude"),
    [
        (False, []),
        (True, []),
        (True, [r"re:.*\.shared_experts?\..*"]),
    ],
)
def test_models_fse_init(
    model_type: str,
    shared_expert_precision: str,
    use_fse: bool,
    exclude: list[str],
    single_rank_parallel_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction and loading preserve each shared expert's precision."""
    _print_warning_once.cache_clear()
    per_layer_fse = model_type != "deepseek_v4_cpu"
    moe_attr = {
        "minimax_m3": "block_sparse_moe",
        "deepseek_v4": "ffn",
    }.get(model_type, "mlp")
    shared_name = (
        "shared_experts"
        if model_type
        in {
            "minimax_m3",
            "deepseek_v4",
            "glm4_moe",
            "deepseek_v2",
            "AXK1",
            "glm4_moe_lite",
            "deepseek_v32",
            "glm5_next",
        }
        else "shared_expert"
    )
    if shared_expert_precision == "mixed" and not per_layer_fse:
        pytest.skip(
            "The mixed Quark two-layer setup does not apply to this architecture"
        )
    quantization_config: dict[str, Any] = (
        get_deepseek_v4_quark_config(
            [f"layers.{idx}.ffn.shared_experts" for idx in range(2)] if exclude else []
        )
        if model_type == "deepseek_v4"
        else {**deepcopy(_QUARK_FSE_CONFIG), "exclude": exclude}
    )
    if shared_expert_precision == "mixed":
        mxfp8_config = deepcopy(quantization_config["global_quant_config"])
        mxfp8_config["input_tensors"].update(
            dtype="fp8_e4m3",
            symmetric=True,
            scale_type=None,
            scale_format=None,
            scale_calculation_mode=None,
            observer_cls="PerGroupMinMaxObserver",
            zero_point_type="int32",
        )
        mxfp8_config["weight"].update(
            dtype="fp8_e4m3",
            qscheme="per_block",
            ch_axis=None,
            group_size=None,
            block_size=[1, 32],
            symmetric=True,
            scale_type="float8_e8m0fnu",
            scale_format=None,
            scale_calculation_mode=None,
            observer_cls="PerBlock2DMinMaxObserver",
            zero_point_type="int32",
        )
        if model_type == "deepseek_v4":
            quantization_config.setdefault("layer_quant_config", {})[
                "layers.1.ffn.shared_experts.w1"
            ] = deepcopy(mxfp8_config)
        else:
            quantization_config.setdefault("layer_quant_config", {}).update(
                {
                    f"model.layers.1.{moe_attr}.{shared_name}.{projection}": deepcopy(
                        mxfp8_config
                    )
                    for projection in ("gate_proj", "up_proj", "down_proj")
                }
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
        use_mla=False,
        logits_processors=None,
        rswa_window=None,
    )
    vllm_config.parallel_config.enable_expert_parallel = False
    if model_type in ("deepseek_v4", "deepseek_v4_cpu"):
        from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

        vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
        vllm_config.quant_config = DeepseekV4FP8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
        )
    else:
        vllm_config.quant_config = QuarkConfig(quantization_config)
        if per_layer_fse:
            vllm_config.quant_config.packed_modules_mapping = {
                "gate_up_proj": ["gate_proj", "up_proj"]
            }
    if model_type == "glm5_next":
        from vllm.models.glm5next.common import model as glm5_next_model

        for attention in ("Glm5NextLinearAttention", "Glm5NextMLAAttention"):
            monkeypatch.setattr(glm5_next_model, attention, _StubGlm5NextAttention)
        # The gate allows fusion only at TP4/TP8 on gfx950, and this test
        # runs at TP1; the gate has its own test.
        monkeypatch.setattr(
            glm5_next_model, "_fused_shared_experts_tuned", lambda _: True
        )
        from vllm.models.glm5next.common import mtp as glm5_next_mtp_model

        # The MTP predictor asserts on the concrete MLA attention class and
        # reads ``mla_attn``, so patch its imported name to the same stub.
        monkeypatch.setattr(
            glm5_next_mtp_model, "Glm5NextMLAAttention", _StubGlm5NextAttention
        )
    if model_type == "glm4_moe_lite":
        from vllm.model_executor.models import glm4_moe_lite as glm4_moe_lite_model

        # Only the MoE/shared-expert wiring is under test here; the MLA backend
        # does not accept this test's default dtype.
        for attention in ("Glm4MoeLiteAttention", "Glm4MoeLiteMLAAttention"):
            monkeypatch.setattr(glm4_moe_lite_model, attention, _StubGlm5NextAttention)
    if model_type == "deepseek_v32":
        from vllm.models.deepseek_v32.amd import model as deepseek_v32_model

        # The sparse-MLA attention needs a real MLA backend; only the MoE
        # wiring is under test here.
        monkeypatch.setattr(
            deepseek_v32_model, "DeepseekV32MLAAttention", _StubGlm5NextAttention
        )
    if model_type == "qwen4_exp":
        from vllm.models.qwen4_exp.amd import model as qwen4_exp_model

        # Only the MoE/shared-expert wiring is under test here.
        monkeypatch.setattr(
            qwen4_exp_model, "Qwen3NextAttention", _StubGlm5NextAttention
        )

    import vllm.envs as envs
    from vllm._aiter_ops import rocm_aiter_ops

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
        from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe

        mp.setattr(rocm_aiter_moe, "aiter_topK_meta_data", None)
        rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()
        aiter_fse_enabled = bool(rocm_aiter_ops.is_fusion_moe_shared_experts_enabled())
        fse_enabled = use_fse and aiter_fse_enabled

        with (
            patch.object(warning_logger, "warning") as warning,
            set_current_vllm_config(vllm_config),
        ):
            model = model_constructor(
                vllm_config=vllm_config,
                **({"prefix": "model"} if per_layer_fse else {}),
            )
            mtps = {
                model_type: _build_mtp_for_model(
                    model_type, vllm_config, config, use_fse and not exclude
                )
            }
            mtp = mtps.get("deepseek_v4")
        if per_layer_fse:
            expected_policy = {
                f"layers.0.{moe_attr}": bool(fse_enabled and not exclude),
                f"layers.1.{moe_attr}": bool(
                    fse_enabled
                    and not exclude
                    and shared_expert_precision == "all_mxfp4"
                ),
            }
            assert model.fused_shared_expert_layers == frozenset(
                idx for idx in range(2) if expected_policy[f"layers.{idx}.{moe_attr}"]
            )
            if any(expected_policy.values()) and not all(expected_policy.values()):
                with pytest.raises(NotImplementedError, match="EPLB"):
                    get_fused_shared_expert_layers(
                        model,
                        type(getattr(model.layers[0], moe_attr)),
                        enable_eplb=True,
                    )
            for idx, layer in enumerate(model.layers):
                fused = expected_policy[f"layers.{idx}.{moe_attr}"]
                moe = getattr(layer, moe_attr)
                assert (getattr(moe, shared_name) is None) is fused
                assert moe.experts.expert_map_manager.num_fused_shared_experts == int(
                    fused
                )
                assert (
                    moe.experts.moe_config.is_fused_shared_expert_requested
                    is aiter_fse_enabled
                )
                if not fused and not exclude and model_type != "deepseek_v4":
                    scheme = getattr(moe, shared_name).gate_up_proj.scheme
                    expected_dtype = (
                        "mxfp8_e4m3"
                        if idx == 1 and shared_expert_precision == "mixed"
                        else "mxfp4"
                    )
                    assert scheme.weight_dtype == expected_dtype

            checkpoint = []
            expected_weights = []
            for idx, layer in enumerate(model.layers):
                fused = expected_policy[f"layers.{idx}.{moe_attr}"]
                if model_type == "deepseek_v4":
                    dsv4_checkpoint, dsv4_expected = (
                        _make_deepseek_v4_shared_expert_checkpoint(
                            layer, idx, fused, exclude
                        )
                    )
                    checkpoint.extend(dsv4_checkpoint)
                    expected_weights.extend(dsv4_expected)
                    continue
                high_precision = idx == 1 and shared_expert_precision == "mixed"
                dtype = (
                    torch.bfloat16
                    if exclude
                    else torch.float8_e4m3fn
                    if high_precision
                    else torch.uint8
                )
                pack_factor = 1 if exclude or high_precision else 2
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    shape = (
                        (128, 32 // pack_factor)
                        if projection == "down_proj"
                        else (32, 128 // pack_factor)
                    )
                    weight = torch.full(shape, idx + 1).to(dtype)
                    name = f"layers.{idx}.{moe_attr}.{shared_name}.{projection}"
                    checkpoint.append((f"{name}.weight", weight))
                    if not exclude:
                        scale_shape = (128, 1) if projection == "down_proj" else (32, 4)
                        checkpoint.append(
                            (
                                f"{name}.weight_scale",
                                torch.full(scale_shape, 127, dtype=torch.uint8),
                            )
                        )
                    if fused:
                        routed = getattr(layer, moe_attr).experts.routed_experts
                        target = (
                            routed.w2_weight[2]
                            if projection == "down_proj"
                            else routed.w13_weight[2].chunk(2, dim=0)[
                                int(projection == "up_proj")
                            ]
                        )
                    else:
                        shared = getattr(getattr(layer, moe_attr), shared_name)
                        target = (
                            shared.down_proj.weight
                            if projection == "down_proj"
                            else shared.gate_up_proj.weight.chunk(2, dim=0)[
                                int(projection == "up_proj")
                            ]
                        )
                    expected_weights.append((target, weight))
                if shared_name == "shared_expert":
                    checkpoint.append(
                        (
                            f"layers.{idx}.{moe_attr}.shared_expert_gate.weight",
                            torch.full((1, 128), idx + 1, dtype=torch.bfloat16),
                        )
                    )
            with set_current_vllm_config(vllm_config):
                loaded = model.load_weights(checkpoint)
                base = config.num_hidden_layers
                mtp_cases = [
                    (
                        mtp_model,
                        f"model.layers.{base}.mlp.{shared}",
                        mtp_model.model.layers[str(base)].mtp_block.mlp,
                        getattr(config, "n_routed_experts", 0),
                    )
                    for mtp_model, shared in (
                        (mtps.get("glm4_moe"), "shared_experts"),
                        (mtps.get("glm4_moe_lite"), "shared_experts"),
                        (mtps.get("deepseek_v2"), "shared_experts"),
                        (mtps.get("deepseek_v32"), "shared_experts"),
                        (mtps.get("glm5_next"), "shared_experts"),
                    )
                    if mtp_model is not None
                ]
                mtp_cases += [
                    (
                        mtp_model,
                        "mtp.layers.0.mlp.shared_expert",
                        mtp_model.model.layers[0].mlp,
                        getattr(config, "num_experts", 0),
                    )
                    for mtp_model in (
                        mtps.get("qwen3_next"),
                        mtps.get("qwen3_5"),
                    )
                    if mtp_model is not None
                ]
                for mtp_model, ckpt_prefix, moe, n_routed in mtp_cases:
                    if fse_enabled and not exclude:
                        _assert_mtp_fused_shared_expert_redirect(
                            mtp_model, ckpt_prefix, moe, n_routed
                        )
                    else:
                        _assert_mtp_shared_expert_not_fused(
                            mtp_model, ckpt_prefix, moe, n_routed
                        )
            for idx in range(2):
                prefix = f"layers.{idx}.{moe_attr}"
                expected_name = (
                    f"{prefix}.experts.routed_experts.w2_weight"
                    if expected_policy[prefix]
                    else f"{prefix}.{shared_name}.down_proj.weight"
                )
                assert expected_name in loaded
                if shared_name == "shared_expert":
                    assert f"{prefix}.shared_expert_gate.weight" in loaded
            for target, weight in expected_weights:
                torch.testing.assert_close(
                    target[: weight.shape[0], : weight.shape[1]].float(), weight.float()
                )
        elif model_type == "deepseek_v4_cpu":
            # CPU never supports fused shared experts; the separate MLP is
            # always built and no routed slot is appended.
            for layer in model.layers:
                assert layer.ffn.shared_experts is not None
                assert (
                    layer.ffn.experts.expert_map_manager.num_fused_shared_experts == 0
                )
        else:
            assert model.is_fused_shared_expert_enabled is (fse_enabled and not exclude)
        if model_type == "glm5_next":
            assert (model.layers[0].mlp.shared_experts is None) is (
                fse_enabled and not exclude
            )

        # The dummy quant config here uses mixed mxfp4/fp8 for experts/shared_expert
        # so should just raise a warning.
        if mtp is not None:
            assert not mtp.model.layers[
                str(config.num_hidden_layers)
            ].mtp_block.ffn.is_fused_shared_expert_enabled
            if aiter_fse_enabled:
                warning.assert_called_once()
                assert (
                    "DeepSeek-V4/V4.1 shared experts at <shared_expert>"
                    in (warning.call_args.args[1])
                )
            else:
                # Without AITER nothing fuses, so no skip reason is reported.
                warning.assert_not_called()
        if aiter_fse_enabled and shared_expert_precision == "mixed" and not exclude:
            warning.assert_called_once()
        if aiter_fse_enabled and shared_expert_precision == "mixed" and not exclude:
            warning.assert_called_once()
            if model_type == "deepseek_v4":
                assert "are not MXFP4" in warning.call_args.args[1]
            else:
                assert (
                    "different quantization configurations" in warning.call_args.args[1]
                )
        for mtp_model in (
            mtps.get("glm4_moe"),
            mtps.get("deepseek_v2"),
            mtps.get("glm4_moe_lite"),
            mtps.get("deepseek_v32"),
            mtps.get("glm5_next"),
        ):
            if mtp_model is not None:
                assert mtp_model.fused_shared_expert_layers == (
                    frozenset({config.num_hidden_layers})
                    if fse_enabled
                    else frozenset()
                )
        if aiter_fse_enabled and exclude:
            if model_type == "deepseek_v4_cpu":
                # CPU has no FSE path, so the resolver never warns.
                warning.assert_not_called()
            else:
                assert warning.call_count == 1
                assert (
                    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
                    in (warning.call_args.args[0])
                )
                assert "excludes shared experts" in warning.call_args.args[1]

    rocm_aiter_moe.init_aiter_topK_meta_data.cache_clear()
    importlib.reload(envs)
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize(
    ("setting", "value", "reason"),
    [
        ("tensor_parallel_size", 4, None),
        ("tensor_parallel_size", 8, None),
        ("tensor_parallel_size", 2, "tensor_parallel_size is 2"),
        ("data_parallel_size", 2, "data_parallel_size is 2"),
        ("prefill_context_parallel_size", 2, "prefill_context_parallel_size is 2"),
        ("enable_expert_parallel", True, "expert parallelism is enabled"),
        ("on_gfx950", False, "the GPU is not gfx950"),
    ],
)
def test_glm5_next_fuses_shared_experts_only_in_tuned_setups(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    value: object,
    reason: str | None,
) -> None:
    from vllm.models.glm5next.common import model as glm5_next_model

    parallel_config = SimpleNamespace(
        tensor_parallel_size=4,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=False,
    )
    on_gfx950 = value if setting == "on_gfx950" else True
    if setting != "on_gfx950":
        setattr(parallel_config, setting, value)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: on_gfx950)

    with patch.object(glm5_next_model.logger, "warning_once") as warning:
        tuned = glm5_next_model._fused_shared_experts_tuned(
            cast(ParallelConfig, parallel_config)
        )

    assert tuned is (reason is None)
    if reason is None:
        warning.assert_not_called()
    else:
        warning.assert_called_once()
        assert warning.call_args.args[1] == reason


def test_glm5_next_fused_shared_expert_loads_into_the_shared_slot() -> None:
    from vllm.models.glm5next.common import model as glm5_next_model

    assert glm5_next_model._num_fused_shared_experts(1, True) == 1
    assert glm5_next_model._num_fused_shared_experts(1, False) == 0
    assert glm5_next_model._num_fused_shared_experts(None, True) == 0
    with pytest.raises(NotImplementedError, match="only 1 shared expert"):
        glm5_next_model._num_fused_shared_experts(2, True)

    for shared, fused in [
        (
            "model.layers.3.mlp.shared_experts.gate_proj.weight_scale_inv",
            "model.layers.3.mlp.experts.288.gate_proj.weight_scale_inv",
        ),
        (
            "model.layers.45.mtp_block.mlp.shared_experts.down_proj.weight",
            "model.layers.45.mtp_block.mlp.experts.288.down_proj.weight",
        ),
    ]:
        assert glm5_next_model._fused_shared_expert_name(shared, 288) == fused
    routed = "model.layers.3.mlp.experts.7.down_proj.weight"
    assert glm5_next_model._fused_shared_expert_name(routed, 288) == routed


@pytest.mark.parametrize(
    ("exclude", "expected"),
    [([], True), ([r"re:.*\.shared_expert\..*"], False)],
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


@pytest.mark.parametrize("exclude_shared_expert", [False, True])
def test_online_targets_off_shared_expert_defer_to_checkpoint(
    exclude_shared_expert: bool,
) -> None:
    """Unrelated online targets preserve the checkpoint's FSE decision."""
    shared_prefix = "model.layers.0.mlp.shared_expert"
    quant_config = QuarkConfig(
        {
            **_QUARK_FSE_CONFIG,
            "exclude": (
                [f"{shared_prefix}.down_proj"] if exclude_shared_expert else []
            ),
        }
    )
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={"*linear_attn.out_proj": "mxfp4"})
    )

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        shared_prefix,
    )

    assert compatible is not exclude_shared_expert
    assert reason == (
        f"Quark excludes shared experts at {shared_prefix}"
        if exclude_shared_expert
        else None
    )


def test_online_targeting_only_one_shared_projection_rejects_fse() -> None:
    shared_prefix = "model.layers.0.mlp.shared_expert"
    quant_config = QuarkConfig({**_QUARK_FSE_CONFIG, "exclude": []})
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={f"{shared_prefix}.down_proj": "mxfp4"})
    )

    assert is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        shared_prefix,
    ) == (
        False,
        "online quantization targets only part of the shared expert at "
        f"{shared_prefix}",
    )


def test_online_fp8_shared_expert_rejects_quark_mxfp4_fse() -> None:
    shared_prefix = "model.layers.0.mlp.shared_expert"
    quant_config = QuarkConfig({**_QUARK_FSE_CONFIG, "exclude": []})
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={f"{shared_prefix}*": "fp8_per_tensor"})
    )

    assert is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        shared_prefix,
    ) == (
        False,
        "online shared-expert quantization keys do not match the routed expert keys",
    )


def test_online_mxfp4_shared_expert_rejects_quark_weight_only_mxfp4_fse() -> None:
    """Weight-only routed experts keep BF16 activations, unlike an MXFP4 target."""
    shared_prefix = "model.layers.0.mlp.shared_expert"
    quant_config = QuarkConfig(
        {
            **_QUARK_FSE_CONFIG,
            "exclude": [],
            "global_quant_config": {
                **_QUARK_FSE_CONFIG["global_quant_config"],
                "input_tensors": None,
            },
        }
    )
    quant_config.online_quantization_config = OnlineQuantizationConfig(
        QuantizationConfigArgs(targets={f"{shared_prefix}*": "mxfp4"})
    )

    assert is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        shared_prefix,
    ) == (
        False,
        "online shared-expert quantization keys do not match the routed expert keys",
    )


@pytest.mark.parametrize(
    ("exclude", "expected"),
    [
        ([], True),
        (["lm_head", "model.embed_tokens"], True),
        (["*.shared_experts.*"], False),
        (["model.layers.0.mlp.experts"], False),
        (["model.layers.0.mlp.experts", "*.shared_experts.*"], True),
    ],
)
def test_modelopt_mxfp8_shared_expert_fse_compatibility(
    exclude: list[str], expected: bool
) -> None:
    """MiniMax-M3 ships MXFP8, whose exclude list decides FSE compatibility."""
    compatible, reason = is_shared_expert_quant_fse_compatible(
        ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=exclude,
        ),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert compatible is expected
    if expected:
        assert reason is None
    else:
        assert reason == (
            "MXFP8 excludes routed and shared experts inconsistently at "
            "model.layers.0.mlp.shared_experts"
        )


def test_quark_shared_expert_fse_exclude_is_scoped_to_the_layer() -> None:
    """Excluding one layer's shared experts must not disable FSE elsewhere.

    amd/GLM-5.2-MXFP4 excludes its MTP layer (78) wholesale, including
    ``model.layers.78.mlp.shared_experts.{gate,up,down}_proj``, while the 75
    target MoE layers keep MXFP4 shared experts.
    """
    quant_config = QuarkConfig(
        {
            "exclude": [
                f"model.layers.78.mlp.shared_experts.{projection_name}"
                for projection_name in ("gate_proj", "up_proj", "down_proj")
            ],
            "global_quant_config": {},
            "layer_quant_config": {},
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    assert is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.3.mlp.experts",
        "model.layers.3.mlp.shared_experts",
    ) == (True, None)

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.78.mlp.experts",
        "model.layers.78.mlp.shared_experts",
    )
    assert compatible is False
    assert (
        reason == "Quark excludes shared experts at model.layers.78.mlp.shared_experts"
    )


def test_quark_shared_expert_fse_rejects_partially_excluded_packed_projection() -> None:
    """Packed shared-expert projections require consistent shard exclusions."""
    shared_expert_prefix = "model.layers.0.mlp.shared_expert"
    quant_config = QuarkConfig(
        {
            "exclude": [f"{shared_expert_prefix}.gate_proj"],
            "global_quant_config": {},
            "layer_quant_config": {},
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    with pytest.raises(ValueError, match="different quantization schemes"):
        is_shared_expert_quant_fse_compatible(
            quant_config, "model.layers.0.mlp.experts", shared_expert_prefix
        )


def test_quark_shared_expert_fse_ignores_sibling_gate_exclusions() -> None:
    """Gates beside a shared expert must not disable FSE.

    Qwen3.x MoE checkpoints (e.g. amd/Qwen3.5-397B-A17B-MoE-MXFP4) quantize the
    shared expert to MXFP4 but leave the tiny routing gates next to it in BF16:
    ``model.language_model.layers.N.mlp.shared_expert_gate`` and
    ``model.language_model.layers.N.mlp.gate``. Neither is a shared-expert
    projection, so fusion must stay enabled. Matching ``shared_expert`` against
    raw exclude entries also matches ``shared_expert_gate``, which silently
    disables FSE on every such checkpoint.
    """
    expert_prefix = "model.language_model.layers.0.mlp.experts"
    shared_expert_prefix = "model.language_model.layers.0.mlp.shared_expert"
    quant_config = QuarkConfig(
        {
            "exclude": [
                "model.language_model.layers.0.mlp.gate",
                f"{shared_expert_prefix}_gate",
            ],
            "global_quant_config": {},
            "layer_quant_config": {},
        }
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    assert is_shared_expert_quant_fse_compatible(
        quant_config, expert_prefix, shared_expert_prefix
    ) == (True, None)

    # The gate exclusions must not mask a genuinely unquantized shared expert.
    quant_config.quant_config["exclude"].append(f"{shared_expert_prefix}.down_proj")
    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config, expert_prefix, shared_expert_prefix
    )
    assert compatible is False
    assert reason == f"Quark excludes shared experts at {shared_expert_prefix}"


def test_quark_ocp_mx_group_size_for_layer() -> None:
    quant_config = QuarkConfig({**_QUARK_FSE_CONFIG, "exclude": []})
    assert (
        get_quark_ocp_mx_group_size(
            quant_config,
            "model.layers.0.mlp.shared_expert.down_proj",
        )
        == 32
    )

    quant_config.quant_config["exclude"] = ["model.layers.0.mlp.shared_expert_gate"]
    assert (
        get_quark_ocp_mx_group_size(
            quant_config,
            "model.layers.0.mlp.shared_expert.down_proj",
        )
        == 32
    )

    quant_config.quant_config["exclude"] = [
        "model.layers.0.mlp.shared_expert.down_proj"
    ]
    assert (
        get_quark_ocp_mx_group_size(
            quant_config,
            "model.layers.0.mlp.shared_expert.down_proj",
        )
        is None
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_quark_ocp_mx_shared_expert_uses_supported_tp_shards(tp_size: int) -> None:
    assert not _should_replicate_misaligned_shared_expert(
        intermediate_size=640,
        tp_size=tp_size,
        group_size=32,
        enable_expert_parallel=False,
        is_sequence_parallel=False,
    )


def test_quark_ocp_mx_shared_expert_rejects_plain_tp8() -> None:
    with pytest.raises(
        ValueError,
        match="TP size 8 produces a partition of 80",
    ):
        _should_replicate_misaligned_shared_expert(
            intermediate_size=640,
            tp_size=8,
            group_size=32,
            enable_expert_parallel=False,
            is_sequence_parallel=False,
        )


def test_quark_ocp_mx_shared_expert_is_replicated_for_tep8() -> None:
    assert _should_replicate_misaligned_shared_expert(
        intermediate_size=640,
        tp_size=8,
        group_size=32,
        enable_expert_parallel=True,
        is_sequence_parallel=False,
    )


def test_quark_ocp_mx_shared_expert_is_replicated_when_not_tp_divisible() -> None:
    assert _should_replicate_misaligned_shared_expert(
        intermediate_size=672,
        tp_size=8,
        group_size=32,
        enable_expert_parallel=True,
        is_sequence_parallel=False,
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
        ExpertsInt8Config(),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert not compatible
    assert reason == (
        "shared-expert FSE quantization compatibility is not implemented for "
        "ExpertsInt8Config"
    )


def _fp8_config(**kwargs: Any) -> Fp8Config:
    return Fp8Config(
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
        **kwargs,
    )


def test_block_fp8_shared_expert_fse_is_compatible() -> None:
    compatible, reason = is_shared_expert_quant_fse_compatible(
        _fp8_config(),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert compatible
    assert reason is None


def test_per_tensor_fp8_shared_expert_fse_is_incompatible() -> None:
    """Per-tensor scales are 0-D, so the shared-expert chunker cannot slice them."""
    compatible, reason = is_shared_expert_quant_fse_compatible(
        Fp8Config(activation_scheme="dynamic"),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert not compatible
    assert reason == (
        "FP8 shared-expert FSE is only implemented for block-quantized checkpoints"
    )


@pytest.mark.parametrize("store_dtype", ["mxfp4", "nvfp4"])
def test_store_dtype_fp8_shared_expert_fse_is_incompatible(store_dtype: str) -> None:
    """Any routed-expert storage override rules out fusing the shared expert."""
    compatible, reason = is_shared_expert_quant_fse_compatible(
        _fp8_config(store_dtype=store_dtype),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert not compatible
    assert reason == (
        f"FP8 stores routed experts as {store_dtype}, which is not supported "
        "for fused shared experts at model.layers.0.mlp.shared_experts"
    )


@pytest.mark.parametrize(
    "ignored_layers",
    [
        ["model.layers.0.mlp.experts"],
        ["model.layers.0.mlp.shared_experts.gate_up_proj"],
        ["model.layers.0.mlp.shared_experts.down_proj"],
    ],
)
def test_fp8_shared_expert_fse_rejects_asymmetric_ignored_layers(
    ignored_layers: list[str],
) -> None:
    compatible, reason = is_shared_expert_quant_fse_compatible(
        _fp8_config(ignored_layers=ignored_layers),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert not compatible
    assert reason == (
        "FP8 ignores routed and shared experts inconsistently at "
        "model.layers.0.mlp.shared_experts"
    )


def test_fp8_shared_expert_fse_allows_symmetric_ignored_layers() -> None:
    """Both sides unquantized is still a consistent scheme."""
    compatible, reason = is_shared_expert_quant_fse_compatible(
        _fp8_config(
            ignored_layers=[
                "model.layers.0.mlp.experts",
                "model.layers.0.mlp.shared_experts.gate_up_proj",
                "model.layers.0.mlp.shared_experts.down_proj",
            ]
        ),
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert compatible
    assert reason is None


def test_fp8_shared_expert_fse_expands_packed_projections() -> None:
    """Both shards of a fused projection ignored is symmetric within that
    projection, so the mismatch reported is against the routed experts."""
    quant_config = _fp8_config(
        ignored_layers=[
            "model.layers.0.mlp.shared_experts.gate_proj",
            "model.layers.0.mlp.shared_experts.up_proj",
        ]
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    compatible, reason = is_shared_expert_quant_fse_compatible(
        quant_config,
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp.shared_experts",
    )

    assert not compatible
    assert reason == (
        "FP8 ignores routed and shared experts inconsistently at "
        "model.layers.0.mlp.shared_experts"
    )


def test_fp8_shared_expert_fse_propagates_partial_shard_exclusion() -> None:
    """Half a fused projection excluded is rejected by `is_layer_skipped`
    itself, exactly as it is in `Fp8Config.get_quant_method`."""
    quant_config = _fp8_config(
        ignored_layers=["model.layers.0.mlp.shared_experts.gate_proj"]
    )
    quant_config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    with pytest.raises(ValueError, match="some but not all shards"):
        is_shared_expert_quant_fse_compatible(
            quant_config,
            "model.layers.0.mlp.experts",
            "model.layers.0.mlp.shared_experts",
        )
