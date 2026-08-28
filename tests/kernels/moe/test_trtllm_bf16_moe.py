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

# (m, n, k) = (tokens, intermediate_size_per_partition, hidden_dim).
# Covers larger NvFP4-like shapes while keeping BF16's FlashInfer TRTLLM
# intermediate-size multiple-of-128 requirement.
MNK_FACTORS = [
    (2, 1024, 1024),
    (64, 2048, 1536),
    (64, 1024, 4096),
]


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("topk", [8])
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
        scores = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(scores, topk)
        topk_weights = topk_weights.contiguous()
        topk_ids = topk_ids.to(torch.int32).contiguous()

        moe_config = FusedMoEConfig(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            num_local_experts=e,
            num_logical_experts=e,
            activation=MoEActivation.SILU,
            device="cuda",
            moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
            in_dtype=dtype,
            routing_method=RoutingMethodType.TopK,
            max_num_tokens=next_power_of_2(m),
        )

        trtllm_w1, trtllm_w2 = convert_moe_weights_to_flashinfer_trtllm_block_layout(
            {},
            w1,
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

        torch.testing.assert_close(
            torch_output,
            trtllm_output,
            atol=1e-1,
            rtol=2e-1,
        )
