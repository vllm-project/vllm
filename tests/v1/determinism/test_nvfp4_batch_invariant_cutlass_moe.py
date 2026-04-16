# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_quant_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        "Nvfp4 Requires compute capability of 10 or above.", allow_module_level=True
    )


_DTYPES = (torch.bfloat16,)
_NUM_EXPERTS = (40, 64)
_TOPKS = (1, 4)

_BATCH_INVARIANT_CASES = (
    {
        "m": 12,
        "n": 1024,
        "k": 1024,
        "subset_indices": ((0, 3, 7), (1, 4, 8, 10), (11, 5, 2, 9, 6)),
    },
    {
        "m": 129,
        "n": 1024,
        "k": 1024,
        "subset_indices": (
            (0, 64, 128),
            (1, 63, 64, 127),
            (32, 33, 95, 96, 128),
        ),
    },
    {
        "m": 73,
        "n": 1472,
        "k": 1536,
        "subset_indices": ((0, 17, 35), (8, 36, 72), (1, 9, 33, 48, 64)),
    },
)


def _make_cutlass_fp4_moe_batch_invariant_case(
    case_config: dict[str, Any],
    activation: MoEActivation,
    e: int,
    topk: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    set_random_seed(7)
    assert topk <= e

    hidden_states = (
        torch.randn((case_config["m"], case_config["k"]), device="cuda", dtype=dtype)
        / 10
    )

    w1_q, w2_q, quant_config = make_test_quant_config(
        e,
        case_config["n"],
        case_config["k"],
        in_dtype=dtype,
        quant_dtype="nvfp4",
        block_shape=None,
        per_act_token_quant=False,
        make_gate=activation.is_gated,
    )

    score = torch.randn((case_config["m"], e), device="cuda", dtype=dtype)

    moe_config = make_dummy_moe_config(
        num_experts=e,
        experts_per_token=topk,
        hidden_dim=case_config["k"],
        intermediate_size_per_partition=case_config["n"],
        in_dtype=dtype,
    )
    kernel = mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        CutlassExpertsFp4(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
        inplace=False,
    )

    return {
        "config": case_config,
        "hidden_states": hidden_states,
        "score": score,
        "kernel": kernel,
        "w1_q": w1_q,
        "w2_q": w2_q,
        "activation": activation,
        "e": e,
        "topk": topk,
        "dtype": dtype,
    }


def _run_cutlass_fp4_moe(
    case: dict[str, Any],
    hidden_states: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, case["topk"], renormalize=False
    )
    return case["kernel"].apply(
        hidden_states=hidden_states,
        w1=case["w1_q"],
        w2=case["w2_q"],
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=case["e"],
        activation=case["activation"],
        apply_router_weight_on_input=False,
        expert_map=None,
    )


@pytest.mark.parametrize("case_config", _BATCH_INVARIANT_CASES)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.SWIGLUSTEP])
@pytest.mark.parametrize("e", _NUM_EXPERTS)
@pytest.mark.parametrize("topk", _TOPKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch.inference_mode()
def test_cutlass_fp4_moe_batch_invariant(
    case_config: dict[str, Any],
    activation: MoEActivation,
    e: int,
    topk: int,
    dtype: torch.dtype,
    default_vllm_config,
    workspace_init,
):
    case = _make_cutlass_fp4_moe_batch_invariant_case(
        case_config, activation, e, topk, dtype
    )
    case_id = f"{case['config']}, e={case['e']}, topk={case['topk']}"
    # Establish the baseline output for the full mixed batch once, then
    # compare every smaller replay against the corresponding slice here.
    batch_output = _run_cutlass_fp4_moe(case, case["hidden_states"], case["score"])

    assert CutlassExpertsFp4._supports_batch_invariance()

    # Re-run the whole batch with a different batch-row permutation to make
    # sure the grouped GEMM result is invariant to how tokens are packed into
    # expert work for scheduling.
    indices = torch.arange(
        case["hidden_states"].size(0), device=case["hidden_states"].device
    )
    for perm_name, perm in (
        ("reversed", torch.flip(indices, dims=(0,))),
        ("evens_then_odds", torch.cat((indices[::2], indices[1::2]))),
    ):
        permuted_output = _run_cutlass_fp4_moe(
            case,
            case["hidden_states"][perm],
            case["score"][perm],
        )
        torch.testing.assert_close(
            batch_output[perm],
            permuted_output,
            atol=0,
            rtol=0,
            msg=(
                f"{case_id}: permutation '{perm_name}' changed outputs "
                "relative to the baseline batch order."
            ),
        )

    # Re-run every batch-row as a batch-size-1 input and compare against the
    # matching row from the full batch.
    for idx in range(case["hidden_states"].size(0)):
        single_idx = torch.tensor([idx], device=case["hidden_states"].device)
        single_output = _run_cutlass_fp4_moe(
            case,
            case["hidden_states"][single_idx],
            case["score"][single_idx],
        )
        torch.testing.assert_close(
            batch_output[single_idx],
            single_output,
            atol=0,
            rtol=0,
            msg=(
                f"{case_id}: batch-row {idx} changed between full-batch "
                "and batch-size-1 execution."
            ),
        )

    # Re-run a few nontrivial sub-batches to catch interactions that only
    # appear when multiple tokens are grouped together.
    for subset_ids in case["config"]["subset_indices"]:
        subset = torch.tensor(subset_ids, device=case["hidden_states"].device)
        subset_output = _run_cutlass_fp4_moe(
            case,
            case["hidden_states"][subset],
            case["score"][subset],
        )
        torch.testing.assert_close(
            batch_output[subset],
            subset_output,
            atol=0,
            rtol=0,
            msg=(
                f"{case_id}: sub-batch {list(subset_ids)} changed "
                "relative to full-batch execution."
            ),
        )
