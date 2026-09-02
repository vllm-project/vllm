# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and weight loading for quark-quantized models.

Run `pytest tests/quantization/test_quark.py`.

See also `tests/kernels/moe/test_ocp_mx_moe.py`.
"""

import importlib.metadata
from dataclasses import dataclass
from importlib.util import find_spec
from types import SimpleNamespace

import huggingface_hub
import lm_eval
import pytest
import torch
from packaging import version

from tests.quantization.utils import load_model_without_vllm_runner
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.config import set_current_vllm_config
from vllm.config.cache import CacheConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.quark.quark import (  # noqa: E501
    QuarkConfig,
    QuarkLinearMethod,
    QuarkNVFP4,
    QuarkOCP_MX,
    QuarkW8A8Fp8,
    QuarkW8A8Fp8PerBlock,
    QuarkW8A8Int8,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (  # noqa: E501
    QuarkMoEMethod,
    QuarkW4A8Fp8MoEMethod,
    QuarkW8A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.quark.utils import QuarkQTensorHint
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    quant_dequant_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockE8M0Sym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt4W4A8StaticChannelSym,
    kInt8DynamicTensorAsym,
    kInt8DynamicTensorSym,
    kInt8DynamicTokenAsym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kInt8StaticTensorAsym,
    kInt8StaticTensorSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx942, on_gfx950
else:

    def on_gfx942() -> bool:
        return False

    def on_gfx950() -> bool:
        return False


from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch

# Minimum amd-quark version for MXFP4/OCP_MX tests (single source of truth).
QUARK_MXFP4_MIN_VERSION = "0.12"

QUARK_MXFP4_AVAILABLE = find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse(QUARK_MXFP4_MIN_VERSION)

AITER_AVAILABLE = is_aiter_found_and_supported()

DEVICE_TYPE = current_platform.device_type


@dataclass(frozen=True)
class QTensorConfig:
    name: str
    weight: QuarkQTensorHint
    input_tensors: QuarkQTensorHint
    weight_quant_key: QuantKey | None = None
    act_quant_key: QuantKey | None = None
    dispatch_cls: type[QuarkScheme] | type[QuarkMoEMethod] | None = None
    expected_error: tuple[type[Exception], str] | None = None


QTENSOR_CONFIGS = [
    QTensorConfig(
        name="fp8_w8a8_static_tensor",
        weight={"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        weight_quant_key=kFp8StaticTensorSym,
        act_quant_key=kFp8StaticTensorSym,
        dispatch_cls=QuarkW8A8Fp8,
    ),
    QTensorConfig(
        name="fp8_w8a8_static_tensor_single_entry_lists",
        weight=[
            {
                "dtype": "fp8_e4m3",
                "qscheme": "per_tensor",
                "is_dynamic": False,
            }
        ],
        input_tensors=[
            {
                "dtype": "fp8_e4m3",
                "qscheme": "per_tensor",
                "is_dynamic": False,
            }
        ],
        weight_quant_key=kFp8StaticTensorSym,
        act_quant_key=kFp8StaticTensorSym,
        dispatch_cls=QuarkW8A8Fp8,
    ),
    QTensorConfig(
        name="fp8_w8a8_dynamic_tensor",
        weight={"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": True,
        },
        weight_quant_key=kFp8StaticTensorSym,
        act_quant_key=kFp8DynamicTensorSym,
        dispatch_cls=QuarkW8A8Fp8,
    ),
    QTensorConfig(
        name="fp8_w8a8_dynamic_token",
        weight={"dtype": "fp8_e4m3", "qscheme": "per_channel", "is_dynamic": False},
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_channel",
            "is_dynamic": True,
        },
        weight_quant_key=kFp8StaticChannelSym,
        act_quant_key=kFp8DynamicTokenSym,
        dispatch_cls=QuarkW8A8Fp8,
    ),
    QTensorConfig(
        name="fp8_w8a8_channel_static_tensor",
        weight={"dtype": "fp8_e4m3", "qscheme": "per_channel", "is_dynamic": False},
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        weight_quant_key=kFp8StaticChannelSym,
        act_quant_key=kFp8StaticTensorSym,
        dispatch_cls=QuarkW8A8Fp8,
    ),
    QTensorConfig(
        name="fp8_w8a8_tensor_dynamic_token",
        weight={"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_channel",
            "is_dynamic": True,
        },
        weight_quant_key=kFp8StaticTensorSym,
        act_quant_key=kFp8DynamicTokenSym,
        dispatch_cls=QuarkW8A8Fp8,
    ),
    QTensorConfig(
        name="fp8_w8a8_dynamic_block_fp32",
        weight={
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "is_dynamic": False,
            "block_size": [128, 128],
            "symmetric": True,
        },
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_group",
            "is_dynamic": True,
            "group_size": 128,
            "symmetric": True,
        },
        weight_quant_key=kFp8Static128BlockSym,
        act_quant_key=kFp8Dynamic128Sym,
        dispatch_cls=QuarkW8A8Fp8PerBlock,
    ),
    QTensorConfig(
        name="fp8_w8a8_dynamic_block_e8m0",
        weight={
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "is_dynamic": False,
            "block_size": [128, 128],
            "symmetric": True,
            "scale_type": "float8_e8m0fnu",
        },
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_group",
            "is_dynamic": True,
            "group_size": 128,
            "symmetric": True,
        },
        weight_quant_key=kFp8Static128BlockE8M0Sym,
        act_quant_key=kFp8Dynamic128Sym,
        dispatch_cls=QuarkW8A8Fp8PerBlock,
    ),
    QTensorConfig(
        name="fp8_w8a8_block_static_input",
        weight={
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "is_dynamic": False,
            "block_size": [128, 128],
            "symmetric": True,
        },
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_group",
            "is_dynamic": False,
            "group_size": 128,
            "symmetric": True,
        },
        expected_error=(NotImplementedError, "No quark compatible scheme"),
    ),
    QTensorConfig(
        name="fp8_w8a8_block_group_size_mismatch",
        weight={
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "is_dynamic": False,
            "block_size": [128, 128],
            "symmetric": True,
        },
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_group",
            "is_dynamic": True,
            "group_size": 64,
            "symmetric": True,
        },
        expected_error=(NotImplementedError, "No quark compatible scheme"),
    ),
    QTensorConfig(
        name="fp8_w8a8_block_missing_block_size",
        weight={
            "dtype": "fp8_e4m3",
            "qscheme": "per_block",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_group",
            "is_dynamic": True,
            "group_size": 128,
            "symmetric": True,
        },
        expected_error=(ValueError, "requires `block_size`"),
    ),
    QTensorConfig(
        name="int8_w8a8_static_symmetric",
        weight={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        weight_quant_key=kInt8StaticTensorSym,
        act_quant_key=kInt8StaticTensorSym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="int8_w8a8_static_asymmetric",
        weight={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": False,
        },
        weight_quant_key=kInt8StaticTensorSym,
        act_quant_key=kInt8StaticTensorAsym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="int8_w8a8_channel_static_symmetric",
        weight={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        weight_quant_key=kInt8StaticChannelSym,
        act_quant_key=kInt8StaticTensorSym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="int8_w8a8_channel_static_asymmetric",
        weight={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": False,
        },
        weight_quant_key=kInt8StaticChannelSym,
        act_quant_key=kInt8StaticTensorAsym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="int8_w8a8_dynamic_tensor_symmetric",
        weight={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": True,
            "symmetric": True,
        },
        weight_quant_key=kInt8StaticTensorSym,
        act_quant_key=kInt8DynamicTensorSym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="int8_w8a8_dynamic_tensor_asymmetric",
        weight={
            "dtype": "int8",
            "qscheme": "per_tensor",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": True,
            "symmetric": False,
        },
        weight_quant_key=kInt8StaticTensorSym,
        act_quant_key=kInt8DynamicTensorAsym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="int8_w8a8_dynamic_token",
        weight={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": True,
            "symmetric": True,
        },
        weight_quant_key=kInt8StaticChannelSym,
        act_quant_key=kInt8DynamicTokenSym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="int8_w8a8_dynamic_token_asymmetric",
        weight={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": False,
            "symmetric": True,
        },
        input_tensors={
            "dtype": "int8",
            "qscheme": "per_channel",
            "is_dynamic": True,
            "symmetric": False,
        },
        weight_quant_key=kInt8StaticChannelSym,
        act_quant_key=kInt8DynamicTokenAsym,
        dispatch_cls=QuarkW8A8Int8,
    ),
    QTensorConfig(
        name="ocp_mx_mxfp4_weight_only",
        weight={
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        input_tensors=None,
        weight_quant_key=kMxfp4Static,
        act_quant_key=None,
        dispatch_cls=QuarkOCP_MX,
    ),
    QTensorConfig(
        name="ocp_mx_mxfp4_activation",
        weight={
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        input_tensors={
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": True,
        },
        weight_quant_key=kMxfp4Static,
        act_quant_key=kMxfp4Dynamic,
        dispatch_cls=QuarkOCP_MX,
    ),
    QTensorConfig(
        name="ocp_mx_mxfp6_e3m2",
        weight={
            "dtype": "fp6_e3m2",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        input_tensors={
            "dtype": "fp6_e3m2",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": True,
        },
        weight_quant_key=kMxfp6E3M2Static,
        act_quant_key=kMxfp6E3M2Dynamic,
        dispatch_cls=QuarkOCP_MX,
    ),
    QTensorConfig(
        name="ocp_mx_mxfp4_mxfp6_e3m2_activation",
        weight={
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        input_tensors={
            "dtype": "fp6_e3m2",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": True,
        },
        weight_quant_key=kMxfp4Static,
        act_quant_key=kMxfp6E3M2Dynamic,
        dispatch_cls=QuarkOCP_MX,
    ),
    QTensorConfig(
        name="ocp_mx_mxfp4_mxfp6_e2m3_activation",
        weight={
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        input_tensors={
            "dtype": "fp6_e2m3",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": True,
        },
        weight_quant_key=kMxfp4Static,
        act_quant_key=kMxfp6E2M3Dynamic,
        dispatch_cls=QuarkOCP_MX,
    ),
    QTensorConfig(
        name="ocp_mx_mxfp6_e2m3",
        weight={
            "dtype": "fp6_e2m3",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        input_tensors={
            "dtype": "fp6_e2m3",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": True,
        },
        weight_quant_key=kMxfp6E2M3Static,
        act_quant_key=kMxfp6E2M3Dynamic,
        dispatch_cls=QuarkOCP_MX,
    ),
    QTensorConfig(
        name="nvfp4",
        weight=[
            {
                "dtype": "fp4",
                "qscheme": "per_group",
                "group_size": 16,
                "is_dynamic": False,
            },
            {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
        ],
        input_tensors=[
            {
                "dtype": "fp4",
                "qscheme": "per_group",
                "group_size": 16,
                "is_dynamic": True,
            },
            {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
        ],
        weight_quant_key=kNvfp4Static,
        act_quant_key=kNvfp4Dynamic,
        dispatch_cls=QuarkNVFP4,
    ),
    QTensorConfig(
        name="w4a8_fp8_static",
        weight=[
            {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
            {
                "dtype": "int4",
                "qscheme": "per_channel",
                "is_dynamic": False,
                "symmetric": True,
                "ch_axis": 0,
            },
        ],
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_tensor",
            "is_dynamic": False,
        },
        weight_quant_key=kInt4W4A8StaticChannelSym,
        act_quant_key=kFp8StaticTensorSym,
        dispatch_cls=QuarkW4A8Fp8MoEMethod,
    ),
    QTensorConfig(
        name="w4a8_fp8_dynamic",
        weight=[
            {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
            {
                "dtype": "int4",
                "qscheme": "per_channel",
                "is_dynamic": False,
                "symmetric": True,
                "ch_axis": 0,
            },
        ],
        input_tensors={
            "dtype": "fp8_e4m3",
            "qscheme": "per_channel",
            "is_dynamic": True,
        },
        weight_quant_key=kInt4W4A8StaticChannelSym,
        act_quant_key=kFp8DynamicTokenSym,
        dispatch_cls=QuarkW4A8Fp8MoEMethod,
    ),
    QTensorConfig(
        name="w4a8_fp8_static_single_entry_input",
        weight=[
            {"dtype": "fp8_e4m3", "qscheme": "per_tensor", "is_dynamic": False},
            {
                "dtype": "int4",
                "qscheme": "per_channel",
                "is_dynamic": False,
                "symmetric": True,
                "ch_axis": 0,
            },
        ],
        input_tensors=[
            {
                "dtype": "fp8_e4m3",
                "qscheme": "per_tensor",
                "is_dynamic": False,
            }
        ],
        weight_quant_key=kInt4W4A8StaticChannelSym,
        act_quant_key=kFp8StaticTensorSym,
        dispatch_cls=QuarkW4A8Fp8MoEMethod,
    ),
]


def _make_qtensor_config(
    weight: QuarkQTensorHint,
    input_tensors: QuarkQTensorHint,
    exclude: list[str] | None = None,
) -> QuarkConfig:
    return QuarkConfig(
        {
            "global_quant_config": {
                "weight": weight,
                "input_tensors": input_tensors,
            },
            "layer_type_quant_config": {},
            "exclude": exclude or [],
        }
    )


def _make_test_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device=current_platform.device_type,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
    )


if QUARK_MXFP4_AVAILABLE:
    from quark.torch.export.nn.modules.realquantizer import StaticScaledRealQuantizer
    from quark.torch.kernel import mx as mx_kernel
    from quark.torch.quantization.config.config import FP4PerGroupSpec

try:
    hf_api().list_repo_refs(
        "amd/Llama-3.3-70B-Instruct-WMXFP4-AMXFP4-KVFP8-Scale-UINT8-SQ"
    )
    HF_HUB_AMD_ORG_ACCESS = True
except huggingface_hub.errors.RepositoryNotFoundError:
    HF_HUB_AMD_ORG_ACCESS = False


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def test_quark_config_has_no_model_specific_fused_mappings():
    config = QuarkConfig({})

    assert "gate_up_proj" not in config.packed_modules_mapping
    assert "fused_wqa_wkv" not in config.packed_modules_mapping


def test_quark_config_preserves_existing_packed_modules_mapping():
    class CustomQuarkConfig(QuarkConfig):
        packed_modules_mapping = {"custom_proj": ["a", "b"]}

    config = CustomQuarkConfig({})

    assert config.packed_modules_mapping["custom_proj"] == ["a", "b"]


def test_quant_method_dispatch_ignored(default_vllm_config):
    config = _make_qtensor_config(None, None, exclude=["linear", "experts"])

    class TestLinear(LinearBase):
        def __init__(self):
            torch.nn.Module.__init__(self)

    class TestRoutedExperts(RoutedExperts):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.moe_config = _make_test_moe_config()

    assert config.get_quant_method_target("linear", LinearBase) == (
        None,
        None,
        UnquantizedLinearMethod,
    )
    assert isinstance(
        config.get_quant_method(TestLinear(), "linear"), UnquantizedLinearMethod
    )

    assert config.get_quant_method_target("experts", RoutedExperts) == (
        None,
        None,
        UnquantizedFusedMoEMethod,
    )
    assert isinstance(
        config.get_quant_method(TestRoutedExperts(), "experts"),
        UnquantizedFusedMoEMethod,
    )

    dynamic_mxfp4_config = _make_qtensor_config(
        {
            "dtype": "fp4",
            "qscheme": "per_group",
            "group_size": 32,
            "scale_format": "e8m0",
            "is_dynamic": False,
        },
        None,
        exclude=["self_attn.q_proj", "mlp.down_proj"],
    )
    dynamic_mxfp4_config.dynamic_mxfp4_quant = True

    assert dynamic_mxfp4_config.get_quant_method_target(
        "self_attn.q_proj", LinearBase
    ) == (kMxfp4Static, None, QuarkLinearMethod)
    attention_proj = TestLinear()
    assert isinstance(
        dynamic_mxfp4_config.get_quant_method(attention_proj, "self_attn.q_proj"),
        QuarkLinearMethod,
    )
    assert isinstance(attention_proj.scheme, QuarkOCP_MX)
    assert attention_proj.scheme.dynamic_mxfp4_quant

    assert dynamic_mxfp4_config.get_quant_method_target(
        "mlp.down_proj", LinearBase
    ) == (None, None, UnquantizedLinearMethod)
    assert isinstance(
        dynamic_mxfp4_config.get_quant_method(TestLinear(), "mlp.down_proj"),
        UnquantizedLinearMethod,
    )


@pytest.mark.parametrize("case", QTENSOR_CONFIGS, ids=lambda case: case.name)
def test_quant_method_dispatch_target(case):
    config = _make_qtensor_config(case.weight, case.input_tensors)
    if case.expected_error is not None:
        error_type, error_message = case.expected_error
        with pytest.raises(error_type, match=error_message):
            config.get_quant_method_target("linear", LinearBase)
        return

    assert case.dispatch_cls is not None
    is_linear = issubclass(case.dispatch_cls, QuarkScheme)

    weight_quant_key, act_quant_key, method_cls = config.get_quant_method_target(
        "linear" if is_linear else "experts",
        LinearBase if is_linear else RoutedExperts,
    )

    assert weight_quant_key == case.weight_quant_key
    assert act_quant_key == case.act_quant_key
    assert method_cls is (QuarkLinearMethod if is_linear else case.dispatch_cls)


@pytest.mark.parametrize(
    ("weight", "input_tensors"),
    [
        pytest.param(
            {
                "dtype": "int8",
                "qscheme": "per_group",
                "is_dynamic": False,
                "symmetric": True,
            },
            {
                "dtype": "int8",
                "qscheme": "per_tensor",
                "is_dynamic": False,
                "symmetric": True,
            },
            id="single_entry",
        ),
        pytest.param(
            [
                {"dtype": "int8", "qscheme": "per_tensor"},
                {"dtype": "int8", "qscheme": "per_tensor"},
            ],
            [
                {"dtype": "int8", "qscheme": "per_tensor"},
                {"dtype": "int8", "qscheme": "per_tensor"},
            ],
            id="multi_entry",
        ),
    ],
)
def test_quant_method_dispatch_unsupported(weight, input_tensors):
    config = _make_qtensor_config(weight, input_tensors)

    class TestRoutedExperts(RoutedExperts):
        def __init__(self):
            torch.nn.Module.__init__(self)

    with pytest.raises(RuntimeError, match="^Unsupported FusedMoe scheme$"):
        config.get_quant_method_target("experts", RoutedExperts)

    with pytest.raises(RuntimeError, match="^Unsupported FusedMoe scheme$"):
        config.get_quant_method(TestRoutedExperts(), "experts")


@pytest.mark.parametrize(
    "case",
    [case for case in QTENSOR_CONFIGS if case.expected_error is None],
    ids=lambda case: case.name,
)
def test_quant_method_dispatch_instantiation(case, monkeypatch, default_vllm_config):
    config = _make_qtensor_config(case.weight, case.input_tensors)
    assert case.dispatch_cls is not None
    if issubclass(case.dispatch_cls, QuarkScheme):

        class TestLinear(LinearBase):
            def __init__(self):
                torch.nn.Module.__init__(self)

        monkeypatch.setattr(
            "vllm.model_executor.layers.quantization.quark.schemes."
            "quark_w8a8_fp8.get_current_vllm_config",
            lambda: SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
        )
        layer = TestLinear()
        method = config.get_quant_method(layer, "linear")

        assert isinstance(method, QuarkLinearMethod)
        assert isinstance(layer.scheme, case.dispatch_cls)
        if case.weight_quant_key == kFp8Static128BlockE8M0Sym:
            # TODO: Remove once E8M0 quant key is properly handled in oracle
            assert layer.scheme.weight_quant_key == kFp8Static128BlockSym
        else:
            assert layer.scheme.weight_quant_key == case.weight_quant_key
        assert layer.scheme.activation_quant_key == case.act_quant_key
    else:

        class TestRoutedExperts(RoutedExperts):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.moe_config = _make_test_moe_config()

        for target in (
            "select_fp8_moe_backend",
            "select_int8_moe_backend",
            "select_mxfp4_moe_backend",
            "backend_to_kernel_cls",
            "select_nvfp4_moe_backend",
        ):
            monkeypatch.setattr(
                f"vllm.model_executor.layers.quantization.quark.quark_moe.{target}",
                lambda *args, **kwargs: (object(), object()),
            )

        # AssertionError: W4A8 FP8 MoE requires ROCm AITER fused MoE support
        monkeypatch.setattr(
            "vllm.model_executor.layers.quantization.quark.quark_moe."
            "rocm_aiter_ops.is_fused_moe_enabled",
            lambda: True,
        )

        layer = TestRoutedExperts()
        method = config.get_quant_method(layer, "experts")

        assert isinstance(method, case.dispatch_cls)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_quark_fp8_w_per_tensor_a_per_tensor(
    kv_cache_dtype: str, monkeypatch, dist_init, workspace_init
):
    model_path = "amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test"
    checkpoint_scales = {}
    scale_names = {
        "model.layers.0.self_attn.k_proj.output_scale",
        "model.layers.0.self_attn.v_proj.output_scale",
    }
    original_load_weights = LlamaForCausalLM.load_weights

    def load_weights(self, weights):
        def capture_scales():
            for name, weight in weights:
                if name in scale_names:
                    checkpoint_scales[name] = weight.detach().cpu()
                yield name, weight

        return original_load_weights(self, capture_scales())

    monkeypatch.setattr(LlamaForCausalLM, "load_weights", load_weights)
    model, vllm_config = load_model_without_vllm_runner(
        model_path,
        model_config_kwargs={"hf_overrides": {"num_hidden_layers": 3}},
        vllm_config_kwargs={"cache_config": CacheConfig(cache_dtype=kv_cache_dtype)},
    )

    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
    assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)
    assert len(qkv_proj.input_scale.shape) == 0
    assert qkv_proj.weight.dtype is current_platform.fp8_dtype()
    assert len(qkv_proj.weight_scale.shape) == 0

    attn = model.model.layers[0].self_attn.attn
    if kv_cache_dtype == "fp8":
        assert checkpoint_scales.keys() == scale_names
        scale_multiplier = 2 if current_platform.is_fp8_fnuz() else 1
        assert attn._k_scale_float == (
            checkpoint_scales["model.layers.0.self_attn.k_proj.output_scale"].item()
            * scale_multiplier
        )
        assert attn._v_scale_float == (
            checkpoint_scales["model.layers.0.self_attn.v_proj.output_scale"].item()
            * scale_multiplier
        )
    else:
        assert attn._k_scale_float == 1.0
        assert attn._v_scale_float == 1.0

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
    positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


def test_quark_fp8_w_per_channel_a_per_token(monkeypatch, dist_init, workspace_init):
    model_path = "amd/Qwen2.5-1.5B-Instruct-ptpc-Quark-ts"
    model, vllm_config = load_model_without_vllm_runner(
        model_path,
        model_config_kwargs={"hf_overrides": {"num_hidden_layers": 3}},
    )

    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
    assert isinstance(qkv_proj.scheme, QuarkW8A8Fp8)
    assert qkv_proj.weight.dtype is current_platform.fp8_dtype()
    assert qkv_proj.weight_scale.shape[0] == qkv_proj.weight.shape[1]
    assert qkv_proj.weight_scale.shape[1] == 1

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
    positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


def test_quark_int8_w_per_tensor_a_per_tensor(monkeypatch, dist_init, workspace_init):
    model_path = "amd/Llama-3.1-8B-Instruct-w-int8-a-int8-sym-test"
    model, vllm_config = load_model_without_vllm_runner(
        model_path,
        model_config_kwargs={"hf_overrides": {"num_hidden_layers": 3}},
    )
    with set_current_vllm_config(vllm_config):
        qkv_proj = model.model.layers[0].self_attn.qkv_proj
        assert isinstance(qkv_proj.quant_method, QuarkLinearMethod)
        assert isinstance(qkv_proj.scheme, QuarkW8A8Int8)

        monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
        input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
        positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
        with set_forward_context(None, vllm_config, num_tokens=input_ids.numel()):
            hidden_states = model(input_ids, positions, None)
            logits = model.compute_logits(hidden_states)
        assert torch.isfinite(logits).all()


def test_quark_int8_w8a8_moe(monkeypatch, dist_init, workspace_init):
    """Test W8A8 INT8 MoE quantization with a tiny Qwen3 MoE model."""
    model_path = "amd/tiny-qwen3-moe-w8a8-int8"
    model, vllm_config = load_model_without_vllm_runner(
        model_path,
        model_config_kwargs={"hf_overrides": {"num_hidden_layers": 3}},
    )

    layer = model.model.layers[0]
    moe = layer.mlp.experts
    assert isinstance(moe._quant_method, QuarkW8A8Int8MoEMethod), (
        f"Expected QuarkW8A8Int8MoEMethod, got {type(moe._quant_method)}"
    )
    qkv_proj = layer.self_attn.qkv_proj
    assert isinstance(qkv_proj.scheme, QuarkW8A8Int8)

    monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
    input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
    positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(None, vllm_config, num_tokens=input_ids.numel()),
    ):
        hidden_states = model(input_ids, positions, None)
        logits = model.compute_logits(hidden_states)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    not (on_gfx950() or on_gfx942()),
    reason="Quark W4A8 (INT4-FP8) MoE requires the AITER kernel on gfx942/gfx950",
)
def test_quark_w4a8_fp8_moe(monkeypatch, dist_init, workspace_init):
    """Test W4A8 (INT4 weight + FP8 activation) MoE with a tiny Qwen3 MoE model.

    W4A8 dispatches through the AITER fused MoE kernel, so AITER must be on.
    """
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    rocm_aiter_ops.refresh_env_variables()

    model_path = "amd/tiny-qwen3-moe-w4a8"
    model, vllm_config = load_model_without_vllm_runner(
        model_path,
    )
    with set_current_vllm_config(vllm_config):
        moe = model.model.layers[0].mlp.experts
        assert isinstance(moe._quant_method, QuarkW4A8Fp8MoEMethod), (
            f"Expected QuarkW4A8Fp8MoEMethod, got {type(moe._quant_method)}"
        )

        monkeypatch.setattr(Attention, "forward", lambda _, q, k, v: q.contiguous())
        input_ids = torch.tensor([1, 2, 3, 4], device=DEVICE_TYPE)
        positions = torch.arange(input_ids.numel(), device=DEVICE_TYPE)
        with set_forward_context(None, vllm_config, num_tokens=input_ids.numel()):
            hidden_states = model(input_ids, positions, None)
            logits = model.compute_logits(hidden_states)
        assert torch.isfinite(logits).all()


def test_quark_fp8_parity(dist_init, workspace_init):
    quark_model_id = "amd-quark/llama-tiny-fp8-quark-quant-method"
    fp8_model_id = "amd-quark/llama-tiny-fp8-quant-method"

    def load_state_dict(model_id: str) -> dict[str, torch.Tensor]:
        model, _ = load_model_without_vllm_runner(model_id)
        return {k: v.cpu() for k, v in model.state_dict().items()}

    quark_state_dict = load_state_dict(quark_model_id)
    fp8_state_dict = load_state_dict(fp8_model_id)

    assert fp8_state_dict.keys() == quark_state_dict.keys()

    for key in fp8_state_dict:
        assert torch.equal(fp8_state_dict[key], quark_state_dict[key])


@dataclass
class AccuracyTestConfig:
    model_name: str
    excepted_value: float

    def get_model_args(
        self,
        tp_size: int,
        model_max_len: int | None = None,
        kwargs: dict | None = None,
    ) -> dict:
        if kwargs is None:
            kwargs = {}

        model_args = {
            "pretrained": self.model_name,
            "dtype": "auto",
            "add_bos_token": True,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": 0.7,
            **kwargs,
        }
        if model_max_len is not None:
            model_args["max_model_len"] = model_max_len

        return model_args


GSM8K_ACCURACY_CONFIGS = [
    # Private model.
    AccuracyTestConfig(
        model_name="amd/DeepSeek-R1-WMXFP4-AMXFP4-Scale-UINT8-MoE-Quant",
        excepted_value=0.96,
    ),
]


@pytest.mark.parametrize("config", GSM8K_ACCURACY_CONFIGS)
@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.",
)
def test_mxfp4_gsm8k_correctness(config: AccuracyTestConfig):
    device_count = torch.accelerator.device_count()
    if device_count < 8:
        pytest.skip(f"This test requires >=8 gpus, got only {device_count}")

    task = "gsm8k"
    rtol = 0.03

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=config.get_model_args(tp_size=8, model_max_len=38768),
        tasks=task,
        batch_size=64,
        num_fewshot=8,
    )

    EXPECTED_VALUE = config.excepted_value
    measured_value = results["results"][task]["exact_match,strict-match"]
    assert (
        measured_value - rtol < EXPECTED_VALUE
        and measured_value + rtol > EXPECTED_VALUE
    ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_fused_qdq_match_quark(float_dtype: torch.dtype, scalings: list[int]):
    torch.manual_seed(0)

    hidden_size = 64 * 32
    inp = (torch.rand(1, hidden_size, dtype=float_dtype, device=DEVICE_TYPE) - 0.5) * 2
    for i in range(hidden_size // 32):
        inp[:, i * 32 : (i + 1) * 32] = (
            inp[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]
        )

    inp_kernel = inp.clone()
    inp_kernel_clone = inp_kernel.clone()

    res_hip = mx_kernel.qdq_mxfp4_hip(inp_kernel_clone, "even")
    res_torch = qdq_mxfp4_torch(inp_kernel, "even")

    for i in range(hidden_size // 32):
        assert torch.all(torch.isfinite(res_hip[:, i * 32 : (i + 1) * 32]))
        assert torch.all(torch.isfinite(res_torch[:, i * 32 : (i + 1) * 32]))

        torch.testing.assert_close(
            res_hip[:, i * 32 : (i + 1) * 32], res_torch[:, i * 32 : (i + 1) * 32]
        )


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_dequant_kernel_match_quark(
    float_dtype: torch.dtype, scalings: list[int]
):
    qspec = FP4PerGroupSpec(
        ch_axis=-1,
        group_size=32,
        scale_format="e8m0",
        scale_calculation_mode="even",
        is_dynamic=False,
    ).to_quantization_spec()

    weight_quantizer = StaticScaledRealQuantizer(
        qspec=qspec,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=float_dtype,
        device=DEVICE_TYPE,
    )

    observer = qspec.observer_cls(qspec, device=DEVICE_TYPE)

    hidden_size = 512
    shape = (11008, hidden_size)

    w = (torch.rand(shape, device=DEVICE_TYPE, dtype=float_dtype) - 0.5) * 2

    # Make it so that different groups have different scales.
    for i in range(hidden_size // 32):
        w[:, i * 32 : (i + 1) * 32] = (
            w[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]
        )

    observer(w)
    scale, _ = observer._calculate_qparams()
    weight_quantizer.scale = scale

    w_mxfp4 = weight_quantizer.to_real_quantize_params(w).to(DEVICE_TYPE)
    weight_quantizer.maybe_convert_and_transpose_scale()

    scale = weight_quantizer.scale

    out_hip = mx_kernel.dq_mxfp4_hip(w_mxfp4, scale, float_dtype)

    out_torch = dq_mxfp4_torch(w_mxfp4, scale, float_dtype)

    assert torch.equal(out_hip, out_torch)


@pytest.mark.skipif(
    not QUARK_MXFP4_AVAILABLE,
    reason=f"amd-quark>={QUARK_MXFP4_MIN_VERSION} is not available",
)
@pytest.mark.skipif(
    not AITER_AVAILABLE,
    reason="AITER is not found or not supported on the current platform",
)
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
def test_mxfp4_dynamic_quant_match_quark(
    float_dtype: torch.dtype, scalings: list[float]
):
    """`AiterMxfp4LinearKernel` quantizes weights dynamically through AITER's
    `dynamic_mxfp4_quant`, while the emulation path quantizes/dequantizes
    through Quark's `qdq_mxfp4`. Check that both agree on the same input.
    """
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    torch.manual_seed(0)

    hidden_size = 32 * 64
    inp = (torch.rand(48, hidden_size, dtype=float_dtype, device=DEVICE_TYPE) - 0.5) * 2
    for i in range(hidden_size // 32):
        inp[:, i * 32 : (i + 1) * 32] = (
            inp[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]
        )

    x_q, x_s = dynamic_mxfp4_quant(inp)
    out_dynamic_quant = dq_mxfp4_torch(x_q, x_s, float_dtype)

    out_quark_qdq = quant_dequant_mxfp4(inp)

    assert torch.equal(out_dynamic_quant, out_quark_qdq)


# Unit tests for ``is_layer_skipped`` fused-name handling.

FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def test_fused_name_listed_directly_is_skipped():
    # Regression for Step-3.5-Flash-FP8: the checkpoint lists the fused
    # name (``qkv_proj``) directly in ``modules_to_not_convert``. When a
    # ``packed_modules_mapping`` is registered on the model, the fused
    # match must still win over per-shard expansion.
    ignored = ["model.layers.0.self_attn.qkv_proj"]
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=ignored,
        fused_mapping=FUSED_MAPPING,
    )
    assert is_layer_skipped(
        prefix="model.layers.0.mlp.gate_up_proj",
        ignored_layers=["model.layers.0.mlp.gate_up_proj"],
        fused_mapping=FUSED_MAPPING,
    )


def test_unfused_shards_listed_is_skipped():
    # Quark INT8 style: per-shard names listed; all shards present means
    # the fused layer is skipped via expansion.
    ignored = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=ignored,
        fused_mapping=FUSED_MAPPING,
    )


def test_partial_shards_raises():
    # Only some shards listed -> ambiguous, must raise. Fused name is
    # not in ignored_layers, so we fall through to per-shard expansion.
    ignored = ["model.layers.0.self_attn.q_proj"]
    with pytest.raises(ValueError):
        is_layer_skipped(
            prefix="model.layers.0.self_attn.qkv_proj",
            ignored_layers=ignored,
            fused_mapping=FUSED_MAPPING,
        )


def test_not_skipped_when_nothing_listed():
    assert not is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=["model.layers.0.mlp.gate_up_proj"],
        fused_mapping=FUSED_MAPPING,
    )


def test_non_fused_layer_unaffected():
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.o_proj",
        ignored_layers=["model.layers.0.self_attn.o_proj"],
        fused_mapping=FUSED_MAPPING,
    )
    assert not is_layer_skipped(
        prefix="model.layers.0.self_attn.o_proj",
        ignored_layers=["model.layers.1.self_attn.o_proj"],
        fused_mapping=FUSED_MAPPING,
    )


def test_substr_match_on_fused_name():
    # Substring matching: a fused-name match should also
    # short-circuit before shard expansion.
    assert is_layer_skipped(
        prefix="model.layers.0.self_attn.qkv_proj",
        ignored_layers=["self_attn.qkv_proj"],
        fused_mapping=FUSED_MAPPING,
        match_mode="substring",
    )


@pytest.mark.parametrize(
    ("prefix", "ignored_layer", "expected"),
    [
        ("model.layers.0.self_attn.b_proj", "b_proj", True),
        ("model.layers.0.self_attn.q_b_proj", "b_proj", False),
        ("model.layers.0.self_attn.kv_b_proj", "b_proj", False),
        ("model.layers.5.self_attn.g_proj", "5.self_attn.g_proj", True),
        ("model.layers.6.self_attn.g_proj", "5.self_attn.g_proj", False),
    ],
)
def test_suffix_match_at_module_boundary(prefix, ignored_layer, expected):
    assert (
        is_layer_skipped(
            prefix=prefix,
            ignored_layers=[ignored_layer],
            match_mode="suffix",
        )
        is expected
    )
