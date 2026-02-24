# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    break_fp4_bytes,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_experts, torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts,
    is_valid_flashinfer_cutlass_fused_moe,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
    flashinfer_trtllm_fp4_moe,
    prepare_nvfp4_moe_layer_for_fi_or_cutlass,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe
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
            intermediate_size_per_partition=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=activation,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            is_act_and_mul=is_gated_act,
            routing_method=RoutingMethodType.TopK,
        )

        flashinfer_experts = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(moe_config=moe_config, quant_config=quant_config),
            inplace=False,
        )

        flashinfer_output = flashinfer_experts(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
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


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.RELU2_NO_MUL])
@torch.inference_mode()
def test_flashinfer_trtllm_fp4_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    activation: MoEActivation,
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

        w13_scale_2 = quant_config.g1_alphas.contiguous()
        w2_scale_2 = quant_config.g2_alphas.contiguous()
        w13_scale_linear = torch.empty_like(quant_config.w1_scale)
        w2_scale_linear = torch.empty_like(quant_config.w2_scale)
        w13_rows = (2 if is_gated_act else 1) * n
        # Convert to linear because flashinfer TRTLLM expects linear scales
        for idx in range(e):
            w13_scale_linear[idx] = convert_swizzled_to_linear(
                quant_config.w1_scale[idx], w13_rows, k, quant_blocksize
            )
            w2_scale_linear[idx] = convert_swizzled_to_linear(
                quant_config.w2_scale[idx], k, n, quant_blocksize
            )
        a13_scale = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_scale = torch.ones((e,), device="cuda", dtype=torch.float32)

        layer = torch.nn.Module()
        layer.activation = activation
        layer.intermediate_size_per_partition = n
        layer.ep_rank = 0
        layer.local_num_experts = e
        layer.routing_method_type = RoutingMethodType.Renormalize

        (
            w13_weight,
            w13_weight_scale,
            w13_scale_2_prepared,
            a13_scale_prepared,
            w2_weight,
            w2_weight_scale,
            w2_scale_2_prepared,
            a2_scale_prepared,
        ) = prepare_nvfp4_moe_layer_for_fi_or_cutlass(
            backend=NvFp4MoeBackend.FLASHINFER_TRTLLM,
            layer=layer,
            w13=w1_q,
            w13_scale=w13_scale_linear,
            w13_scale_2=w13_scale_2,
            a13_scale=a13_scale,
            w2=w2_q,
            w2_scale=w2_scale_linear,
            w2_scale_2=w2_scale_2,
            a2_scale=a2_scale,
            is_act_and_mul=is_gated_act,
        )

        layer.w13_weight = w13_weight
        layer.w13_weight_scale = w13_weight_scale
        layer.w13_weight_scale_2 = w13_scale_2_prepared
        layer.w13_input_scale = a13_scale_prepared
        layer.w2_weight = w2_weight
        layer.w2_weight_scale = w2_weight_scale
        layer.w2_weight_scale_2 = w2_scale_2_prepared
        layer.w2_input_scale = a2_scale_prepared

        a_global_scale = layer.a1_gscale.max().to(torch.float32)
        a_fp4_kernel, a_scale_interleaved_kernel = ops.scaled_fp4_quant(
            a, a_global_scale, is_sf_swizzled_layout=False
        )

        flashinfer_output = flashinfer_trtllm_fp4_moe(
            layer=layer,
            x=(a_fp4_kernel, a_scale_interleaved_kernel),
            router_logits=score,
            top_k=topk,
            activation=activation,
            global_num_experts=e,
            num_expert_group=None,
            topk_group=None,
            custom_routing_function=None,
            e_score_correction_bias=None,
        )

        # Reference check:
        a_scale_linear = a_scale_interleaved_kernel.view(torch.float8_e4m3fn).reshape(
            m, k // quant_blocksize
        )
        a_in_dtype = _dequant_nvfp4_to_shape(
            a_fp4_kernel,
            a_scale_linear,
            a_global_scale,
            dtype=dtype,
            block_size=quant_blocksize,
            target_shape=(m, k),
        )

        w1_d = torch.empty(
            (e, (2 if is_gated_act else 1) * n, k), device="cuda", dtype=dtype
        )
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = _dequant_nvfp4_to_shape(
                w1_q[idx],
                w13_scale_linear[idx],
                w13_scale_2_prepared[idx],
                dtype=dtype,
                block_size=quant_blocksize,
                target_shape=(w13_rows, k),
            )

            w2_d[idx] = _dequant_nvfp4_to_shape(
                w2_q[idx],
                w2_scale_linear[idx],
                w2_scale_2_prepared[idx],
                dtype=dtype,
                block_size=quant_blocksize,
                target_shape=(k, n),
            )

        topk_weights, topk_ids, _ = fused_topk(
            a_in_dtype, score, topk, renormalize=True
        )
        torch_output = torch_experts(
            a_in_dtype,
            w1_d,
            w2_d,
            topk_weights,
            topk_ids,
            activation=activation,
        )

        torch.testing.assert_close(
            torch_output, flashinfer_output, atol=1e-1, rtol=1e-1
        )


def _dequant_nvfp4_to_shape(
    quantized_tensor: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    target_shape: torch.Size,
) -> torch.Tensor:
    return (
        (
            break_fp4_bytes(quantized_tensor, torch.float32).reshape(
                target_shape[-2], target_shape[-1] // block_size, block_size
            )
            * (scale.to(torch.float32) * global_scale).unsqueeze(-1)
        )
        .reshape(target_shape[-2], target_shape[-1])
        .to(dtype)
    )


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
