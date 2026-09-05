# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
    MEGA_MOE_BACKENDS,
    KernelConfig,
    validate_flashinfer_moe_ep_model,
)
from vllm.model_executor.layers.fused_moe import flashinfer_moe_ep as fi_ep
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType


def test_legacy_cutedsl_backend_normalizes_to_shared_backend():
    config = KernelConfig(moe_backend="flashinfer_moe_ep_mega_cutedsl")
    assert config.moe_backend == FLASHINFER_MOE_EP_CUTEDSL


def test_flashinfer_backends_use_standard_fused_moe_model_path():
    assert FLASHINFER_MOE_EP_BACKENDS.isdisjoint(MEGA_MOE_BACKENDS)


def test_only_deep_gemm_backend_is_dsv4_specific():
    validate_flashinfer_moe_ep_model(
        FLASHINFER_MOE_EP_CUTEDSL,
        ["MixtralForCausalLM"],
    )
    with pytest.raises(ValueError, match="only supported for DeepSeek-V4"):
        validate_flashinfer_moe_ep_model(
            FLASHINFER_MOE_EP_DEEP_GEMM,
            ["MixtralForCausalLM"],
        )
    validate_flashinfer_moe_ep_model(
        FLASHINFER_MOE_EP_DEEP_GEMM,
        ["DeepseekV4ForCausalLM"],
    )


@pytest.mark.parametrize(
    "moe_backend",
    (*sorted(MEGA_MOE_BACKENDS), *sorted(FLASHINFER_MOE_EP_BACKENDS)),
)
def test_internal_ep_backends_enable_dsv4_sequence_parallel(moe_backend: str):
    from vllm.models.deepseek_v4.nvidia.model import _use_sequence_parallel

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )
    assert _use_sequence_parallel(vllm_config)


def test_dsv4_requests_pre_fc2_router_weight_placement():
    dsv4 = SimpleNamespace(routing_method=RoutingMethodType.DeepseekV4)
    default = SimpleNamespace(routing_method=RoutingMethodType.Default)
    assert fi_ep.apply_topk_in_fc1(dsv4)
    assert not fi_ep.apply_topk_in_fc1(default)


def test_backend_specs_preserve_weight_format_contracts():
    cutedsl = fi_ep.flashinfer_moe_ep_backend_spec(FLASHINFER_MOE_EP_CUTEDSL)
    assert cutedsl.kernel == "cutedsl"
    assert cutedsl.weight_formats == frozenset({"nvfp4", "mxfp4"})

    deep_gemm = fi_ep.flashinfer_moe_ep_backend_spec(FLASHINFER_MOE_EP_DEEP_GEMM)
    assert deep_gemm.kernel == "deep_gemm"
    assert deep_gemm.weight_formats == frozenset({"mxfp4"})


def test_shared_backend_validation_requires_expert_parallel(monkeypatch):
    config = SimpleNamespace(
        weight_transfer_config=None,
        parallel_config=SimpleNamespace(enable_dbo=False, enable_eplb=False),
    )
    monkeypatch.setattr(fi_ep, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        fi_ep.current_platform,
        "get_device_capability",
        lambda: None,
    )
    moe = SimpleNamespace(
        moe_backend=FLASHINFER_MOE_EP_CUTEDSL,
        moe_parallel_config=SimpleNamespace(use_ep=False),
        is_lora_enabled=False,
        skip_final_all_reduce=False,
        in_dtype=torch.bfloat16,
        activation=MoEActivation.SILU,
        has_bias=False,
        swiglu_alpha=None,
        swiglu_beta=None,
        routing_method=RoutingMethodType.DeepseekV4,
    )

    with pytest.raises(ValueError, match="expert parallel disabled"):
        fi_ep.validate_flashinfer_moe_ep_config(moe, "nvfp4")

    moe.moe_parallel_config.use_ep = True
    fi_ep.validate_flashinfer_moe_ep_config(moe, "nvfp4")
