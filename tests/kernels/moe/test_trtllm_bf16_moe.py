# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FlashInfer TRTLLM BF16 MoE backend
(`TrtLlmBf16ExpertsModular`).

This mirrors the TRTLLM NvFP4 modular test shape: construct the modular
expert wrapper directly, pass production-format BlockMajorK weights, and
compare against a torch MoE reference using the original BF16 weights.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
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
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmBf16LoRAExperts,
)
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed

if pytest and (
    not has_flashinfer_trtllm_fused_moe()
    or not current_platform.is_device_capability_family(100)
):
    pytest.skip(
        "Requires flashinfer TRTLLM fused MoE BF16 backend (SM100)",
        allow_module_level=True,
    )

# (m, n, k) = (tokens, intermediate_size_per_partition, hidden_dim).
MNK_FACTORS = [
    (2, 160, 2560),
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
        # The FlashInfer conversion may rewrite unpadded input storage in place.
        # Preserve the original layout for the independent torch reference.
        reference_w1 = w1.clone()
        reference_w2 = w2.clone()
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

        trtllm_w1, trtllm_w2 = convert_to_unquantized_kernel_format(
            UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            moe_config,
            w1,
            w2,
        )
        expected_n = (n + 127) // 128 * 128
        assert moe_config.intermediate_size_per_partition == expected_n
        assert trtllm_w2.numel() == e * k * expected_n

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
            reference_w1,
            reference_w2,
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


@pytest.mark.parametrize("has_lora_delta", [False, True])
@torch.inference_mode()
def test_trtllm_bf16_lora_accepts_checkpoint_shaped_weights(has_lora_delta):
    """LoRA dispatch accepts both 3D parameters and legacy 4D packed views."""
    config = make_dummy_moe_config(
        num_experts=128, hidden_dim=256, intermediate_size=128
    )
    experts = TrtLlmBf16LoRAExperts(config, FUSED_MOE_UNQUANTIZED_CONFIG)
    x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16) / 10
    w1, w2 = convert_to_unquantized_kernel_format(
        UnquantizedMoeBackend.FLASHINFER_TRTLLM,
        config,
        torch.randn(128, 256, 256, device="cuda", dtype=torch.bfloat16) / 10,
        torch.randn(128, 256, 128, device="cuda", dtype=torch.bfloat16) / 10,
    )
    assert w1.ndim == w2.ndim == 3
    topk = (
        torch.tensor([[0], [1]], device="cuda", dtype=torch.int32),
        torch.ones(2, 1, device="cuda", dtype=torch.float32),
    )
    delta = torch.randn(2, 1, 256, device="cuda", dtype=torch.bfloat16)

    def invoke(w1, w2):
        result = experts.invoke_routed_moe(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_ids_and_weights=topk,
            gemm1_lora_delta=delta if has_lora_delta else None,
            global_num_experts=128,
            a1q_scale=None,
            output=torch.empty_like(x),
        )
        if has_lora_delta:
            # Only compare rows belonging to real tokens, excluding padding.
            indices = result[2].flatten().long()
            return result[0][indices], result[3][indices]
        return result

    actual = invoke(w1, w2)
    expected = invoke(w1.view(128, 4, 256, 64), w2.view(128, 2, 256, 64))
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)
