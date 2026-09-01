# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsModular,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    make_nvfp4_moe_kernel,
)

NUM_EXPERTS = 4


def _make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size=32,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        device="cpu",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
    )


def _make_layer_and_quant_config() -> tuple[torch.nn.Module, FusedMoEQuantConfig]:
    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight_scale_2",
        torch.nn.Parameter(torch.arange(2, 6, dtype=torch.float32)),
    )
    layer.register_parameter(
        "w2_weight_scale_2",
        torch.nn.Parameter(torch.arange(6, 10, dtype=torch.float32)),
    )
    layer.w13_input_scale = torch.full((NUM_EXPERTS,), 2.0)
    layer.w2_input_scale = torch.full((NUM_EXPERTS,), 4.0)
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=layer.w13_weight_scale_2,
        g2_alphas=layer.w2_weight_scale_2,
        a1_gscale=torch.full((NUM_EXPERTS,), 0.5),
        a2_gscale=torch.full((NUM_EXPERTS,), 0.25),
        w1_scale=torch.ones((NUM_EXPERTS, 1, 1)),
        w2_scale=torch.ones((NUM_EXPERTS, 1, 1)),
    )
    return layer, quant_config


def _make_experts(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dynamic_gemm2: bool,
    per_token_activation: bool = False,
) -> tuple[TrtLlmNvFp4ExpertsModular, torch.nn.Module]:
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: "cpu")
    layer, quant_config = _make_layer_and_quant_config()
    experts = TrtLlmNvFp4ExpertsModular(
        moe_config=_make_moe_config(),
        quant_config=quant_config,
        dynamic_gemm2=dynamic_gemm2,
        per_token_activation=per_token_activation,
    )
    experts.process_weights_after_loading(layer)
    return experts, layer


def test_dynamic_gemm2_preserves_static_gemm1_quantization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts, layer = _make_experts(monkeypatch, dynamic_gemm2=True)

    torch.testing.assert_close(
        layer.w13_weight_scale_2, torch.arange(2, 6, dtype=torch.float32)
    )
    torch.testing.assert_close(
        layer.w2_weight_scale_2, torch.arange(6, 10, dtype=torch.float32)
    )
    torch.testing.assert_close(experts.g1_scale_c, layer.w13_weight_scale_2)
    assert experts.expects_unquantized_inputs is False

    hidden_states = torch.zeros((3, 8), dtype=torch.uint8)
    block_scale = torch.ones((3, 1), dtype=torch.uint8)
    actual_hidden, actual_block, row_scale = experts._prepare_gemm1_input(
        hidden_states, block_scale
    )
    assert actual_hidden is hidden_states
    assert actual_block is block_scale
    assert row_scale is not None and row_scale.is_contiguous()
    torch.testing.assert_close(row_scale, torch.full((3,), 2.0))


def test_static_path_keeps_scale_folding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts, layer = _make_experts(monkeypatch, dynamic_gemm2=False)

    torch.testing.assert_close(
        layer.w13_weight_scale_2, 2 * torch.arange(2, 6, dtype=torch.float32)
    )
    torch.testing.assert_close(
        layer.w2_weight_scale_2, 4 * torch.arange(6, 10, dtype=torch.float32)
    )
    torch.testing.assert_close(experts.g1_scale_c, 0.25 * layer.w13_weight_scale_2)


def test_online_per_token_activation_still_selects_dynamic_gemm2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts, layer = _make_experts(
        monkeypatch, dynamic_gemm2=False, per_token_activation=True
    )

    assert experts.dynamic_gemm2 is True
    assert experts.expects_unquantized_inputs is True
    assert "gemm1_input_decode_scale" not in dict(layer.named_buffers())
    torch.testing.assert_close(
        layer.w13_weight_scale_2, torch.arange(2, 6, dtype=torch.float32)
    )
    torch.testing.assert_close(
        layer.w2_weight_scale_2, torch.arange(6, 10, dtype=torch.float32)
    )


def test_dynamic_gemm2_state_is_eplb_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts, layer = _make_experts(monkeypatch, dynamic_gemm2=True)

    buffers = dict(layer.named_buffers())
    assert buffers.keys() == {"gemm1_input_decode_scale"}
    assert buffers["gemm1_input_decode_scale"].shape == torch.Size([])
    assert "gemm1_input_decode_scale" not in layer.state_dict()

    params = dict(layer.named_parameters())
    assert all(param.shape[0] == NUM_EXPERTS for param in params.values())
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        for param in params.values():
            param.copy_(param[permutation])

    assert experts.g1_scale_c.data_ptr() == params["g1_scale_c"].data_ptr()
    torch.testing.assert_close(experts.gemm1_input_decode_scale, torch.tensor(2.0))


def test_dynamic_gemm2_modular_workspace_uses_quantized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts, _ = _make_experts(monkeypatch, dynamic_gemm2=True)
    workspaces = experts.workspace_shapes(
        M=3,
        N=32,
        K=8,
        topk=2,
        global_num_experts=NUM_EXPERTS,
        local_num_experts=NUM_EXPERTS,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )
    assert workspaces == ((0,), (0,), (3, 16))


class _ExpertsSpy(TrtLlmNvFp4ExpertsModular):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.moe_config = kwargs["moe_config"]


@pytest.mark.parametrize("enabled", [False, True])
def test_dynamic_gemm2_env_is_forwarded_to_trtllm_experts(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    monkeypatch.setenv(
        "VLLM_FLASHINFER_MOE_NVFP4_DYNAMIC_GEMM2", "1" if enabled else "0"
    )
    monkeypatch.setattr(
        nvfp4_oracle,
        "maybe_make_prepare_finalize",
        lambda **_: SimpleNamespace(
            activation_format=FusedMoEActivationFormat.Standard
        ),
    )
    monkeypatch.setattr(
        nvfp4_oracle.mk,
        "FusedMoEKernel",
        lambda _prepare_finalize, experts: SimpleNamespace(fused_experts=experts),
    )
    _, quant_config = _make_layer_and_quant_config()

    kernel = make_nvfp4_moe_kernel(
        moe_quant_config=quant_config,
        moe_config=_make_moe_config(),
        experts_cls=_ExpertsSpy,
        backend=NvFp4MoeBackend.FLASHINFER_TRTLLM,
    )
    assert kernel.fused_experts.kwargs["dynamic_gemm2"] is enabled
    assert kernel.fused_experts.kwargs["per_token_activation"] is False


def test_dynamic_gemm2_rejects_other_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_FLASHINFER_MOE_NVFP4_DYNAMIC_GEMM2", "1")
    _, quant_config = _make_layer_and_quant_config()
    with pytest.raises(ValueError, match="requires the FlashInfer TRTLLM"):
        make_nvfp4_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=_make_moe_config(),
            experts_cls=_ExpertsSpy,
            backend=NvFp4MoeBackend.FLASHINFER_CUTLASS,
        )
