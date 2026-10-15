# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
    is_valid_flashinfer_cutlass_fused_moe,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if not has_flashinfer_cutlass_fused_moe() or not current_platform.has_device_capability(
    100
):
    pytest.skip(
        "Requires flashinfer_cutlass_fused_moe and nvfp4 support",
        allow_module_level=True,
    )

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUOAI_UNINTERLEAVE],
)
def test_flashinfer_swigluoai_params_are_forwarded(activation, monkeypatch):
    from flashinfer.fused_moe.core import ActivationType

    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=128,
        intermediate_size=128,
        num_local_experts=2,
        num_logical_experts=2,
        activation=activation,
        device="cuda",
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
    )
    quant_config = FusedMoEQuantConfig.make(
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
    )
    experts = FlashInferExperts(moe_config=moe_config, quant_config=quant_config)

    call_args = {}

    def fake_flashinfer_cutlass_fused_moe(**kwargs):
        call_args.update(kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.experts."
        "flashinfer_cutlass_moe.flashinfer_cutlass_fused_moe",
        fake_flashinfer_cutlass_fused_moe,
    )
    experts.apply(
        output=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        hidden_states=torch.empty((1, 128), device="cuda", dtype=torch.bfloat16),
        w1=torch.empty((2, 256, 128), device="cuda", dtype=torch.bfloat16),
        w2=torch.empty((2, 128, 128), device="cuda", dtype=torch.bfloat16),
        topk_weights=torch.ones((1, 1), device="cuda", dtype=torch.float32),
        topk_ids=torch.zeros((1, 1), device="cuda", dtype=torch.int64),
        activation=activation,
        global_num_experts=2,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert experts._supports_activation(activation)
    assert call_args["activation_type"] == ActivationType.Swiglu
    for name, value in (
        ("swiglu_alpha", 1.702),
        ("swiglu_beta", 1.0),
        ("swiglu_limit", 7.0),
    ):
        torch.testing.assert_close(
            call_args[name],
            torch.full((2,), value, device="cuda", dtype=torch.float32),
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16
        is_gated_act = activation.is_gated

        w1_q, w2_q, quant_config = make_test_quant_config(
            e,
            n,
            k,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
            make_gate=is_gated_act,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        flashinfer_experts = FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
        )

        flashinfer_output = flashinfer_experts.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        # Reference check:
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                quant_config.w1_scale[idx],
                (1 / quant_config.g1_alphas[idx]),
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                quant_config.w2_scale[idx],
                (1 / quant_config.g2_alphas[idx]),
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=activation
        )

        torch.testing.assert_close(
            torch_output, flashinfer_output, atol=1e-1, rtol=1e-1
        )


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)


@pytest.mark.parametrize("n,h", [(928, 128), (1856, 256), (1024, 256), (1856, 2688)])
@pytest.mark.parametrize("flat_scales", [False, True])
def test_mxfp8_non_gated_padding_preserves_payload_and_neutral_scales(
    n, h, flat_scales
):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        _pad_mxfp8_native_intermediate,
    )

    # Padding must preserve checkpoint bytes and use UE8M0 127 (scale 1).
    w13 = torch.randint(0, 120, (1, n, h), dtype=torch.uint8).view(torch.float8_e4m3fn)
    w2 = torch.randint(0, 120, (1, h, n), dtype=torch.uint8).view(torch.float8_e4m3fn)
    s13 = torch.randint(110, 135, (1, n, h // 32), dtype=torch.uint8)
    s2 = torch.randint(110, 135, (1, h, n // 32), dtype=torch.uint8)
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            is_act_and_mul=False, intermediate_size_per_partition=n, hidden_dim=h
        )
    )
    a, b, c, d = _pad_mxfp8_native_intermediate(
        layer,
        w13,
        w2,
        s13.flatten(1) if flat_scales else s13,
        s2.flatten(1) if flat_scales else s2,
    )
    pn, ph = ((n + 127) // 128) * 128, ((h + 255) // 256) * 256
    assert a.shape == (1, pn, ph) and b.shape == (1, ph, pn)
    assert layer.moe_config.intermediate_size_per_partition == pn
    assert layer.moe_config.hidden_dim == ph
    c, d = c.reshape(1, pn, ph // 32), d.reshape(1, ph, pn // 32)
    assert torch.equal(a[:, :n, :h].view(torch.uint8), w13.view(torch.uint8))
    assert torch.equal(b[:, :h, :n].view(torch.uint8), w2.view(torch.uint8))
    assert torch.equal(c[:, :n, : h // 32], s13)
    assert torch.equal(d[:, :h, : n // 32], s2)
    assert not torch.count_nonzero(a[:, n:, :].view(torch.uint8))
    assert not torch.count_nonzero(a[:, :, h:].view(torch.uint8))
    assert not torch.count_nonzero(b[:, h:, :].view(torch.uint8))
    assert not torch.count_nonzero(b[:, :, n:].view(torch.uint8))
    assert torch.all(c[:, n:, :] == 127) and torch.all(c[:, :, h // 32 :] == 127)
    assert torch.all(d[:, h:, :] == 127) and torch.all(d[:, :, n // 32 :] == 127)


def test_mxfp8_non_gated_padding_preserves_relu2_output():
    # The runner's padded input and cropped output preserve non-gated RELU2.
    torch.manual_seed(17)
    x = torch.randn(3, 128)
    up = torch.randn(928, 128)
    down = torch.randn(128, 928)
    reference = torch.relu(x @ up.T).square() @ down.T
    xp = torch.nn.functional.pad(x, (0, 128))
    upp = torch.nn.functional.pad(up, (0, 128, 0, 96))
    downp = torch.nn.functional.pad(down, (0, 96, 0, 128))
    padded = (torch.relu(xp @ upp.T).square() @ downp.T)[:, :128]
    torch.testing.assert_close(reference, padded, rtol=1e-5, atol=0.01)
