# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUTLASS batch-invariance tests.

The NVFP4 CUTLASS tests in this file must run with ``VLLM_BATCH_INVARIANT=1``
before the first relevant native kernel call. Do not run them in the same pytest
process after tests that intentionally exercise non-batch-invariant NVFP4
scaled-mm kernels.
"""

from typing import Any

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import get_nvfp4_global_scale
from tests.utils import TestFP8Layer, requires_fp8
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsFp4,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytest.importorskip("torch.cuda")

_NVFP4_REQUIRES_SM100 = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="Nvfp4 Requires compute capability of 10 or above.",
)

_NVFP4_LINEAR_DTYPES = [torch.float16, torch.bfloat16]
_NVFP4_LINEAR_CONSISTENCY_SHAPES = [
    (256, 128, 4096),
    (512, 256, 4096),
    (256, 256, 2048),
    (241, 160, 2048),
    (401, 352, 1984),
    (333, 320, 1008),
    (287, 96, 4096),
]

_NVFP4_MOE_DTYPES = (torch.bfloat16,)
_NVFP4_MOE_NUM_EXPERTS = (40, 64)
_NVFP4_MOE_TOPKS = (1, 4)

_NVFP4_MOE_BATCH_INVARIANT_CASES = (
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


@pytest.fixture(autouse=True)
def setup_cuda():
    if not current_platform.is_cuda():
        pytest.skip("CUTLASS FP8 kernels require CUDA.")
    torch.set_default_device("cuda")


@requires_fp8
@pytest.mark.parametrize("weight_shape", [(1024, 2048), (4608, 4096)])
@pytest.mark.parametrize("batch_size", [1, 16, 17, 32, 64, 65, 256, 257])
@torch.inference_mode()
def test_cutlass_fp8_batch_invariant_fixed_config(
    weight_shape: tuple[int, int],
    batch_size: int,
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    torch.manual_seed(0)
    layer = TestFP8Layer(
        weight_shape=weight_shape,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_quant_key=kFp8StaticTensorSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device=torch.device("cuda"),
        force_kernel=CutlassFP8ScaledMMLinearKernel,
    )
    assert isinstance(layer.kernel, CutlassFP8ScaledMMLinearKernel)

    in_features = weight_shape[1]
    needle = torch.randn((1, in_features), device="cuda", dtype=torch.bfloat16)
    baseline = layer(needle)[0]

    filler = torch.randn(
        (max(batch_size - 1, 0), in_features), device="cuda", dtype=torch.bfloat16
    )

    front_batch = torch.cat([needle, filler], dim=0)
    back_batch = torch.cat([filler, needle], dim=0)

    front_output = layer(front_batch)[0]
    back_output = layer(back_batch)[-1]

    torch.testing.assert_close(front_output, baseline, rtol=0, atol=0)
    torch.testing.assert_close(back_output, baseline, rtol=0, atol=0)


@_NVFP4_REQUIRES_SM100
@pytest.mark.parametrize("dtype", _NVFP4_LINEAR_DTYPES)
@pytest.mark.parametrize("shape", _NVFP4_LINEAR_CONSISTENCY_SHAPES)
@torch.inference_mode()
def test_cutlass_nvfp4_scaled_mm_batch_invariant(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
) -> None:
    """Each row of a full-M GEMM must match its M=1 counterpart."""
    set_random_seed(12345)
    m, n, packed_k = shape
    k = packed_k * 2  # real K (FP4 elements)

    a_dtype = torch.randn((m, k), dtype=dtype, device="cuda")
    b_dtype = torch.randn((n, k), dtype=dtype, device="cuda")

    a_global_scale = get_nvfp4_global_scale(a_dtype)
    b_global_scale = get_nvfp4_global_scale(b_dtype)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(b_dtype, b_global_scale)

    a_fp4_full, a_sf_full = ops.scaled_fp4_quant(a_dtype, a_global_scale)
    out_full = ops.cutlass_scaled_fp4_mm(
        a_fp4_full,
        b_fp4,
        a_sf_full,
        b_scale_interleaved,
        alpha,
        dtype,
    )

    for i in range(m):
        a_row = a_dtype[i : i + 1]
        a_fp4_row, a_sf_row = ops.scaled_fp4_quant(a_row, a_global_scale)
        out_row = ops.cutlass_scaled_fp4_mm(
            a_fp4_row,
            b_fp4,
            a_sf_row,
            b_scale_interleaved,
            alpha,
            dtype,
        )

        assert torch.equal(out_full[i], out_row[0]), (
            f"VLLM_BATCH_INVARIANT: row {i} differs between M={m} and M=1: "
            f"max_abs_diff={(out_full[i] - out_row[0]).abs().max().item()}"
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


@_NVFP4_REQUIRES_SM100
@pytest.mark.parametrize("case_config", _NVFP4_MOE_BATCH_INVARIANT_CASES)
@pytest.mark.parametrize("activation", [MoEActivation.SILU, MoEActivation.SWIGLUSTEP])
@pytest.mark.parametrize("e", _NVFP4_MOE_NUM_EXPERTS)
@pytest.mark.parametrize("topk", _NVFP4_MOE_TOPKS)
@pytest.mark.parametrize("dtype", _NVFP4_MOE_DTYPES)
@torch.inference_mode()
def test_cutlass_nvfp4_moe_batch_invariant(
    case_config: dict[str, Any],
    activation: MoEActivation,
    e: int,
    topk: int,
    dtype: torch.dtype,
    default_vllm_config,
    workspace_init,
) -> None:
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
