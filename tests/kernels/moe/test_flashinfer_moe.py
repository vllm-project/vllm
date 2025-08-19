# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest
import torch

from tests.kernels.moe.utils import make_test_weights
from tests.kernels.quantization.nvfp4_utils import (FLOAT4_E2M1_MAX,
                                                    FLOAT8_E4M3_MAX,
                                                    dequantize_nvfp4_to_dtype)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.flashinfer_cutlass_moe import (
    FlashInferExperts, is_valid_flashinfer_cutlass_fused_moe)
from vllm.model_executor.layers.fused_moe.fused_moe import (fused_experts,
                                                            fused_topk)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    apply_flashinfer_per_tensor_scale_fp8, flashinfer_cutlass_moe_fp8,
    register_moe_scaling_factors, rotate_flashinfer_fp8_moe_weights,
    swap_w13_to_w31)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    input_to_float8)
from vllm.model_executor.models.llama4 import Llama4MoE
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_cutlass_fused_moe

if not has_flashinfer_cutlass_fused_moe(
) or not current_platform.has_device_capability(100):
    pytest.skip("Requires flashinfer_cutlass_fused_moe and nvfp4 support",
                allow_module_level=True)

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 1024, 1536),
    (2, 3072, 1024),
    (2, 3072, 1536),
    (64, 1024, 1024),
    (64, 1024, 1536),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
#@pytest.mark.parametrize("e", [128, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_flashinfer_fp4_moe_no_graph(m: int, n: int, k: int, e: int, topk: int,
                                     dtype: torch.dtype):
    current_platform.seed_everything(7)
    with set_current_vllm_config(
            VllmConfig(parallel_config=ParallelConfig(
                pipeline_parallel_size=1))):

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        quant_blocksize = 16

        (_, w1_q, w1_blockscale,
         w1_gs), (_, w2_q, w2_blockscale, w2_gs) = make_test_weights(
             e,
             n,
             k,
             in_dtype=dtype,
             quant_dtype="nvfp4",
             block_shape=None,  # use quant_blocksize?
             per_act_token_quant=False,
         )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a,
                                               score,
                                               topk,
                                               renormalize=False)

        a1_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e, ), device="cuda", dtype=torch.float32)

        assert is_valid_flashinfer_cutlass_fused_moe(a, w1_q, w2_q)

        assert w1_gs is not None
        assert w2_gs is not None
        assert w1_blockscale is not None
        assert w2_blockscale is not None

        flashinfer_experts = FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            FlashInferExperts(
                a1_gscale=a1_gs,
                g1_alphas=(1 / w1_gs),
                a2_gscale=a2_gs,
                g2_alphas=(1 / w2_gs),
                out_dtype=dtype,
                quant_dtype="nvfp4",
            ))

        flashinfer_output = flashinfer_experts(
            hidden_states=a,
            w1=w1_q,
            w1_scale=w1_blockscale,
            w2=w2_q,
            w2_scale=w2_blockscale,
            a1_scale=a1_gs,
            a2_scale=a2_gs,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )

        # Reference check:
        a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) /
                          torch.amax(a.flatten(), dim=-1)).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(a_fp4,
                                               a_scale_interleaved,
                                               a_global_scale,
                                               dtype=a.dtype,
                                               device=a.device,
                                               block_size=quant_blocksize)

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(w1_q[idx],
                                                  w1_blockscale[idx],
                                                  w1_gs[idx],
                                                  dtype=dtype,
                                                  device=w1_q.device,
                                                  block_size=quant_blocksize)
            w2_d[idx] = dequantize_nvfp4_to_dtype(w2_q[idx],
                                                  w2_blockscale[idx],
                                                  w2_gs[idx],
                                                  dtype=dtype,
                                                  device=w2_q.device,
                                                  block_size=quant_blocksize)

        torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk)

        torch.testing.assert_close(torch_output,
                                   flashinfer_output,
                                   atol=1e-1,
                                   rtol=1e-1)


NUM_EXPERTS = [16]
TOP_KS = [1]

MNK_FACTORS_FP8 = [
    (256, 8192, 5120),
    (256, 4096, 5120),
    (127, 8192, 5120),
    (127, 4096, 5120),
    (10, 8192, 5120),
    (10, 4096, 5120),
    (1, 8192, 5120),
    (1, 4096, 5120),
]

vllm_config = VllmConfig(parallel_config=ParallelConfig(
    pipeline_parallel_size=1))
vllm_config.scheduler_config.max_num_seqs = 128
vllm_config.scheduler_config.max_model_len = 8192


def quant_fp8_per_tensor_batches(a):
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        a_fp8, a_global_sf = input_to_float8(a[i])
        a_global_sf = 1.0 / a_global_sf
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


@dataclass
class TestData:
    hidden_states: torch.Tensor
    w13_quantized: torch.Tensor
    w2_quantized: torch.Tensor
    a1_scale: torch.Tensor
    a2_scale: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    layer: torch.nn.Module

    @staticmethod
    def make_moe_tensors_8bit(m: int, k: int, n: int, e: int,
                              reorder: bool) -> "TestData":
        hidden_states = torch.randn(
            (m, k), device="cuda", dtype=torch.bfloat16) / 10
        w13 = torch.randn((e, 2 * n, k), device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn((e, k, n), device="cuda", dtype=torch.bfloat16)

        # Scale to fp8
        _, a1_scale = input_to_float8(hidden_states)
        a1_scale = 1.0 / a1_scale
        a2_scale = torch.scalar_tensor(1.0).to(device="cuda").to(
            dtype=torch.float32)
        w13_quantized, w13_weight_scale = quant_fp8_per_tensor_batches(w13)
        w2_quantized, w2_weight_scale = quant_fp8_per_tensor_batches(w2)

        layer = torch.nn.Module()
        layer.w13_weight = w13_quantized.clone()
        layer.w2_weight = w2_quantized.clone()
        layer.w13_input_scale = a1_scale
        layer.w2_input_scale = a2_scale
        layer.w13_weight_scale = w13_weight_scale
        layer.w2_weight_scale = w2_weight_scale

        register_moe_scaling_factors(layer)

        # flashinfer expects swapped rows for w13
        layer.w13_weight.data = swap_w13_to_w31(layer.w13_weight.data)
        if reorder:
            rotate_flashinfer_fp8_moe_weights(layer.w13_weight,
                                              layer.w2_weight)
        layer.custom_routing_function = Llama4MoE.custom_routing_function
        layer.intermediate_size_per_partition = n
        layer.ep_rank = 0
        layer.local_num_experts = e

        return TestData(
            hidden_states=hidden_states,
            w13_quantized=w13_quantized,
            w2_quantized=w2_quantized,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            layer=layer,
        )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS_FP8)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
def test_flashinfer_per_tensor_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    monkeypatch,
):
    current_platform.seed_everything(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(m, k, n, e, reorder=True)

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=td.hidden_states,
            router_logits=score,
            use_grouped_topk=False,
            top_k=topk,
            renormalize=False,
            custom_routing_function=Llama4MoE.custom_routing_function,
            scoring_func="softmax")

        output = fused_experts(
            td.hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation="silu",
            use_fp8_w8a8=True,
            per_channel_quant=False,
            global_num_experts=e,
            expert_map=None,
            w1_scale=td.w13_weight_scale,
            w2_scale=td.w2_weight_scale,
            a1_scale=td.a1_scale,
            a2_scale=td.a2_scale,
            apply_router_weight_on_input=True,
        )

        flashinfer_output = apply_flashinfer_per_tensor_scale_fp8(
            layer=td.layer,
            hidden_states=td.hidden_states,
            router_logits=score,
            routing_bias=None,
            global_num_experts=e,
            top_k=topk,
            num_expert_group=None,
            topk_group=None,
            apply_router_weight_on_input=True)

        torch.testing.assert_close(output,
                                   flashinfer_output,
                                   atol=5.5e-2,
                                   rtol=1e-2)


@pytest.mark.skip(
    "Requires flashinfer version that contains https://github.com/flashinfer-ai/flashinfer/pull/1472"
)
@pytest.mark.parametrize("m,n,k", MNK_FACTORS_FP8)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
def test_flashinfer_cutlass_moe_fp8_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    monkeypatch,
):
    current_platform.seed_everything(7)
    monkeypatch.setenv("VLLM_FUSED_MOE_CHUNK_SIZE", "8192")
    with set_current_vllm_config(vllm_config):
        td = TestData.make_moe_tensors_8bit(m, k, n, e, reorder=False)

        score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=td.hidden_states,
            router_logits=score,
            use_grouped_topk=False,
            top_k=topk,
            renormalize=False,
            custom_routing_function=Llama4MoE.custom_routing_function,
            scoring_func="softmax")

        output = fused_experts(
            td.hidden_states,
            td.w13_quantized,
            td.w2_quantized,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation="silu",
            use_fp8_w8a8=True,
            per_channel_quant=False,
            global_num_experts=e,
            expert_map=None,
            w1_scale=td.w13_weight_scale,
            w2_scale=td.w2_weight_scale,
            a1_scale=td.a1_scale,
            a2_scale=td.a2_scale,
            apply_router_weight_on_input=True,
        )

        td.layer.dp_size = 1

        flashinfer_cutlass_output = flashinfer_cutlass_moe_fp8(
            td.hidden_states,
            td.layer,
            topk_weights,
            topk_ids,
            activation="silu",
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=True,
        )

        torch.testing.assert_close(output,
                                   flashinfer_cutlass_output,
                                   atol=5.5e-2,
                                   rtol=1e-2)


if __name__ == "__main__":
    test_flashinfer_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
