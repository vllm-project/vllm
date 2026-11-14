# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-agnostic checks that FLASHINFER_B12X NVFP4 MoE weight preparation
and ``FlashInferB12xExperts`` agree on the intermediate size.

Only FlashInfer's ``B12xMoEWrapper`` is replaced, since its kernels need SM12x.
Weight preparation, scale swizzling and padding run for real, so this runs on
CPU or any CUDA GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
    FlashInferB12xExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
    NvFp4MoeBackend,
    convert_to_nvfp4_moe_kernel_format,
    make_nvfp4_moe_quant_config,
)
from vllm.platforms import current_platform

flashinfer_fused_moe = pytest.importorskip("flashinfer.fused_moe")
pytest.importorskip("flashinfer.cute_dsl.utils")

# Nemotron-3.5-Lightning-30B-A3B-NVFP4 non-gated MoE shapes, with few experts.
NUM_EXPERTS = 4
HIDDEN = 2688


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CPU-only shims: FusedMoEConfig rejects non-gated activations off
    # CUDA/ROCm/XPU, and swizzle_blockscale() unconditionally calls .cuda().
    import vllm.model_executor.layers.fused_moe.config as moe_config_mod

    monkeypatch.setattr(moe_config_mod.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    assert not current_platform.is_cuda()
    return torch.device("cpu")


def _random_nvfp4(
    e: int, rows: int, cols: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randint(0, 256, (e, rows, cols // 2), dtype=torch.uint8)
    scale = (torch.rand(e, rows, cols // 16) + 0.5).to(torch.float8_e4m3fn)
    return weight.to(device), scale.to(device)


@pytest.mark.parametrize(
    "activation,intermediate,expected_intermediate",
    [
        # Non-gated: swizzling pads w13 rows to a multiple of 128.
        (MoEActivation.RELU2_NO_MUL, 1856, 1920),
        (MoEActivation.RELU2_NO_MUL, 2048, 2048),
        # Gated: w13 holds [up, gate] with 2 * intermediate rows, no padding.
        (MoEActivation.SILU, 1856, 1856),
        (MoEActivation.SILU, 2048, 2048),
    ],
)
@torch.inference_mode()
def test_b12x_wrapper_matches_prepared_weights(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
    activation: MoEActivation,
    intermediate: int,
    expected_intermediate: int,
):
    torch.manual_seed(0)
    e, k = NUM_EXPERTS, HIDDEN
    w13_rows = 2 * intermediate if activation.is_gated else intermediate

    w13, w13_scale = _random_nvfp4(e, w13_rows, k, device)
    w2, w2_scale = _random_nvfp4(e, k, intermediate, device)
    ones_e = torch.ones(e, dtype=torch.float32, device=device)

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=2,
            hidden_dim=k,
            intermediate_size=intermediate,
            activation=activation,
        )
        layer = SimpleNamespace(activation=activation, moe_config=moe_config)
        (
            w13,
            w13_scale,
            w13_scale_2,
            a13_scale,
            w2,
            w2_scale,
            w2_scale_2,
            a2_scale,
        ) = convert_to_nvfp4_moe_kernel_format(
            nvfp4_backend=NvFp4MoeBackend.FLASHINFER_B12X,
            layer=layer,
            w13=w13,
            w13_scale=w13_scale,
            w13_scale_2=ones_e.clone(),
            a13_scale=ones_e.clone(),
            w2=w2,
            w2_scale=w2_scale,
            w2_scale_2=ones_e.clone(),
            a2_scale=ones_e.clone(),
            is_act_and_mul=moe_config.is_act_and_mul,
        )
        layer.w13_weight = w13
        layer.w13_weight_scale = w13_scale
        layer.w13_weight_scale_2 = w13_scale_2
        layer.w2_weight = w2
        layer.w2_weight_scale = w2_scale
        layer.w2_weight_scale_2 = w2_scale_2

        experts = FlashInferB12xExperts(
            moe_config=moe_config,
            quant_config=make_nvfp4_moe_quant_config(
                backend=NvFp4MoeBackend.FLASHINFER_B12X,
                w13_scale=w13_scale,
                w2_scale=w2_scale,
                w13_scale_2=w13_scale_2,
                w2_scale_2=w2_scale_2,
                a13_scale=a13_scale,
                a2_scale=a2_scale,
            ),
        )
        experts.process_weights_after_loading(layer)

        wrapper_kwargs: dict = {}

        class FakeB12xMoEWrapper:
            def __init__(self, **kwargs):
                wrapper_kwargs.update(kwargs)

            def run(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
                return torch.zeros_like(x)

        monkeypatch.setattr(
            flashinfer_fused_moe, "B12xMoEWrapper", FakeB12xMoEWrapper, raising=False
        )

        m = 3
        hidden_states = torch.randn(m, k, dtype=torch.bfloat16, device=device)
        output = torch.empty_like(hidden_states)
        experts.apply(
            output=output,
            hidden_states=hidden_states,
            w1=w13,
            w2=w2,
            topk_weights=torch.ones(m, 2, dtype=torch.float32, device=device),
            topk_ids=torch.tensor([[0, 1], [1, 2], [2, 3]], device=device),
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    up_mult = 2 if activation.is_gated else 1
    assert w13.shape[1] == up_mult * expected_intermediate
    assert wrapper_kwargs["intermediate_size"] == expected_intermediate
    assert wrapper_kwargs["intermediate_size"] == w13.shape[1] // up_mult
    assert wrapper_kwargs["intermediate_size"] == 2 * w2.shape[2]
