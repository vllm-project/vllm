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


def _run_bf16_padded(experts, a, trtllm_w1, trtllm_w2, topk_weights, topk_ids, e):
    """Drive TrtLlmBf16ExpertsModular.apply directly on a padded layout.

    ``a`` / ``topk_ids`` already carry the padding tail (rows whose topk_ids
    are -1). Workspaces are FlashInfer-managed, so we pass the (0,)-shaped
    stand-ins that workspace_shapes advertises.
    """
    output = torch.empty_like(a)
    empty_ws = torch.empty((0,), device=a.device, dtype=a.dtype)
    experts.apply(
        output=output,
        hidden_states=a,
        w1=trtllm_w1,
        w2=trtllm_w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=e,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=empty_ws,
        workspace2=empty_ws,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return output


@pytest.mark.parametrize("m,n,k", [(64, 1024, 1024)])
@pytest.mark.parametrize("e", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_trtllm_bf16_moe_padding_robust(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    """Padding rows (topk_ids == -1) must be skipped, never fed into results.

    DeepEP v2 hands these experts a PaddedStandard buffer whose tail rows are
    inactive, marked by topk_ids == -1. This constructs that layout directly,
    poisons the padding rows with NaN, and checks the valid outputs are finite
    and bit-identical to a zero-padding run -- garbage in the skipped rows never
    reaches a valid row.

    Note: this proves robustness/correctness (padding excluded from results),
    not physical compute-skipping. MoE rows are independent in the token
    dimension and activation quant is per-row, so skipping and
    compute-then-discard yield identical valid outputs; the compute-savings of
    tile-granularity skipping belongs in a kernel microbenchmark.
    """
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        pad = 64
        a_valid = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        scores = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weights_valid, topk_ids_valid = torch.topk(scores, topk)

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
            max_num_tokens=next_power_of_2(m + pad),
        )
        trtllm_w1, trtllm_w2 = convert_to_unquantized_kernel_format(
            UnquantizedMoeBackend.FLASHINFER_TRTLLM,
            moe_config,
            w1,
            w2,
        )
        experts = TrtLlmBf16ExpertsModular(
            moe_config=moe_config,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        )

        # Padded layout: valid rows followed by an inactive tail (topk_ids=-1).
        topk_ids = torch.full((m + pad, topk), -1, device="cuda", dtype=torch.int32)
        topk_ids[:m] = topk_ids_valid.to(torch.int32)
        topk_weights = torch.zeros(
            (m + pad, topk), device="cuda", dtype=topk_weights_valid.dtype
        )
        topk_weights[:m] = topk_weights_valid

        a_clean = torch.zeros((m + pad, k), device="cuda", dtype=dtype)
        a_clean[:m] = a_valid
        a_poison = a_clean.clone()
        a_poison[m:] = float("nan")

        out_clean = _run_bf16_padded(
            experts, a_clean, trtllm_w1, trtllm_w2, topk_weights, topk_ids, e
        )
        out_poison = _run_bf16_padded(
            experts, a_poison, trtllm_w1, trtllm_w2, topk_weights, topk_ids, e
        )

        assert torch.isfinite(out_poison[:m]).all(), (
            "padding garbage leaked into valid outputs"
        )
        torch.testing.assert_close(out_poison[:m], out_clean[:m], rtol=0, atol=0)

        torch_output = torch_moe(
            a_valid,
            w1,
            w2,
            score,
            topk,
            activation=MoEActivation.SILU,
        )
        torch.testing.assert_close(
            torch_output,
            out_clean[:m],
            atol=1e-1,
            rtol=2e-1,
        )
