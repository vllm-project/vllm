# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the FlashInfer TRTLLM BF16 MoE backend
(`TrtLlmBf16ExpertsModular`).

This mirrors the TRTLLM NvFP4 modular test shape: construct the modular
expert wrapper directly, pass production-format BlockMajorK weights, and
compare against a torch MoE reference using the original BF16 weights.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.utils import torch_moe
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsModular,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    convert_moe_weights_to_flashinfer_trtllm_block_layout,
    swap_w13_to_w31,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if pytest and (
    not has_flashinfer_trtllm_fused_moe()
    or not current_platform.has_device_capability(100)
):
    pytest.skip(
        "Requires flashinfer TRTLLM fused MoE BF16 backend (SM100)",
        allow_module_level=True,
    )

MNK_FACTORS = [
    (2, 1024, 1024),
    (32, 1024, 1024),
]


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_trtllm_bf16_moe_modular_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size_per_partition=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=MoEActivation.SILU,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            is_act_and_mul=True,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        trtllm_w1, trtllm_w2 = convert_moe_weights_to_flashinfer_trtllm_block_layout(
            {},
            swap_w13_to_w31(w1),
            w2,
        )

        trtllm_experts = mk.FusedMoEKernel(
            maybe_make_prepare_finalize(
                moe=moe_config,
                quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            TrtLlmBf16ExpertsModular(
                moe_config=moe_config,
                quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
            ),
        )

        trtllm_output = trtllm_experts.apply(
            hidden_states=a,
            w1=trtllm_w1,
            w2=trtllm_w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        torch_output = torch_moe(
            a,
            w1,
            w2,
            score,
            topk,
            activation=MoEActivation.SILU,
        )

        close = torch.isclose(trtllm_output, torch_output, atol=1e-1, rtol=0.85)
        assert close.float().mean() > 0.925
