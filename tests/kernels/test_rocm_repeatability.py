# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run-to-run repeatability of the ROCm kernels vLLM dispatches.

Calling a kernel again with the same inputs must return the same bits, with
AITER on and off. Otherwise near-tie greedy tokens flip between identical runs
and the deterministic-scheduling promise in docs/usage/reproducibility.md does
not hold.

AITER kernels that fail this in the pinned AITER are strict xfails, so the
AITER bump that fixes one fails here until its marker is removed.
"""

import math
from functools import partial

import pytest
import torch

import vllm.kernels  # noqa: F401 - registers the RMSNorm implementations
import vllm.model_executor.layers.utils  # noqa: F401 - registers the GEMM op
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.utils import _assert_deterministic
from tests.utils import TestFP8Layer
from vllm import ir
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CompilationConfig, CompilationMode, VllmConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)

N_CALLS = 8

# Tracking for the AITER kernels that are not repeatable, as measured on gfx950.
AITER_A16W16_SPLIT_K = (
    "https://github.com/ROCm/aiter/pull/7053, https://github.com/ROCm/aiter/issues/8349"
)
AITER_A8W8_BLOCKSCALE_SPLIT_K = (
    "https://github.com/ROCm/aiter/pull/8541, https://github.com/ROCm/aiter/pull/8609"
)
AITER_FUSED_MOE = "AITER_MOE_TRACKING_URL"


@pytest.fixture(autouse=True)
def seeded_cuda_device():
    torch.set_default_device("cuda")
    set_random_seed(0)


def _expect_unrepeatable(request: pytest.FixtureRequest, tracking: str) -> None:
    request.applymarker(
        pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=f"AITER kernel is not repeatable on gfx950: {tracking}",
        )
    )


def _route(
    num_tokens: int, num_experts: int, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(torch.randn(num_tokens, num_experts), dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    return topk_weights, topk_ids.to(torch.int32)


@pytest.mark.parametrize(
    ("out_features", "rows", "aiter_split_k"),
    [
        (4096, 1, False),
        (4096, 16, False),
        (4096, 32, True),
        (4096, 64, False),
        (1024, 8, True),
        (1024, 16, False),
        (1024, 64, True),
        (1024, 512, False),
    ],
)
def test_unquantized_gemm_is_repeatable(
    out_features, rows, aiter_split_k, use_aiter, request
):
    """BF16 linear through the ROCm GEMM dispatch: hipBLASLt, vLLM's skinny
    GEMMs, or AITER's tuned GEMM, whose split-K rows are the known exception."""
    if aiter_split_k and rocm_aiter_ops.is_tgemm_enabled():
        _expect_unrepeatable(request, AITER_A16W16_SPLIT_K)

    in_features = 4096
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02
    x = torch.randn(rows, in_features, dtype=torch.bfloat16)

    _assert_deterministic(
        torch.ops.vllm.rocm_unquantized_gemm, x, weight, None, n_runs=N_CALLS
    )


@pytest.mark.parametrize(
    ("rows", "aiter_split_k"), [(16, True), (64, True), (128, False), (256, False)]
)
def test_fp8_block_linear_is_repeatable(
    rows, aiter_split_k, use_aiter, default_vllm_config, request
):
    """FP8 128x128 block-scaled linear through the kernel vLLM selects: AITER's
    gemm_a8w8_blockscale, whose split-K rows are the known exception, or Triton."""
    if not current_platform.supports_fp8():
        pytest.skip("FP8 is not supported on this hardware")
    if aiter_split_k and rocm_aiter_ops.is_linear_enabled() and on_gfx950():
        _expect_unrepeatable(request, AITER_A8W8_BLOCKSCALE_SPLIT_K)

    layer = TestFP8Layer(
        (3072, 3072),
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    x = torch.randn(rows, 3072, dtype=torch.bfloat16)

    _assert_deterministic(layer, x, n_runs=N_CALLS)


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_fused_moe_is_repeatable(topk, use_aiter, request):
    """BF16 fused MoE: AITER's kernel, which is the known exception from three
    experts per token, or vLLM's Triton kernel."""
    num_experts, hidden_dim, intermediate_dim, num_tokens = 32, 1024, 512, 16
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.bfloat16
    ) / math.sqrt(hidden_dim)
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim, dtype=torch.bfloat16
    ) / math.sqrt(intermediate_dim)
    topk_weights, topk_ids = _route(num_tokens, num_experts, topk)

    if rocm_aiter_ops.is_fused_moe_enabled():
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            AiterQuantType,
            rocm_aiter_fused_experts,
        )

        if topk >= 3 and on_gfx950():
            _expect_unrepeatable(request, AITER_FUSED_MOE)
        w1, w2 = rocm_aiter_ops.shuffle_weights(w1, w2)
        run_moe = partial(
            rocm_aiter_fused_experts,
            x,
            w1,
            w2,
            topk_weights,
            topk_ids,
            moe_config=make_dummy_moe_config(
                num_experts=num_experts,
                experts_per_token=topk,
                hidden_dim=hidden_dim,
                intermediate_size=intermediate_dim,
            ),
            quant_type=AiterQuantType.NO,
        )
    else:
        run_moe = partial(fused_experts, x, w1, w2, topk_weights, topk_ids)

    _assert_deterministic(run_moe, n_runs=N_CALLS)


@pytest.mark.parametrize("topk", [1, 2, 4])
def test_aiter_mxfp4_fused_moe_is_repeatable(topk, request):
    """AITER's MXFP4-weight (W4A16) fused MoE, an AITER-only kernel, which is
    the known exception from three experts per token."""
    from vllm._aiter_ops import is_aiter_found_and_supported
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterMxfp4Experts,
        AiterQuantType,
        mxfp4_w4a16_extra_kwargs,
        rocm_aiter_fused_experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_quant_config,
    )

    if not (
        is_aiter_found_and_supported()
        and AiterMxfp4Experts._supports_quant_scheme(kMxfp4Static, None)
    ):
        pytest.skip("AITER MXFP4 MoE is not supported on this hardware")
    if topk >= 3:
        _expect_unrepeatable(request, AITER_FUSED_MOE)

    num_experts, hidden_dim, intermediate_dim, num_tokens = 32, 1024, 512, 16
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.bfloat16)
    topk_weights, topk_ids = _route(num_tokens, num_experts, topk)

    # Any byte is a valid pair of FP4 values; small E8M0 scales keep the
    # outputs finite.
    def packed_fp4(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8)

    def e8m0_scale(*shape):
        return torch.randint(118, 122, shape, dtype=torch.uint8)

    w1, w2, w1_scale, w2_scale, _, _ = (
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_BF16,
            layer=torch.nn.Module(),
            w13_weight=packed_fp4(num_experts, 2 * intermediate_dim, hidden_dim // 2),
            w2_weight=packed_fp4(num_experts, hidden_dim, intermediate_dim // 2),
            w13_weight_scale=e8m0_scale(
                num_experts, 2 * intermediate_dim, hidden_dim // 32
            ),
            w2_weight_scale=e8m0_scale(num_experts, hidden_dim, intermediate_dim // 32),
        )
    )
    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_dim,
    )
    quant_config = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.AITER_MXFP4_BF16, w1_scale=w1_scale, w2_scale=w2_scale
    )
    run_moe = partial(
        rocm_aiter_fused_experts,
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        moe_config=moe_config,
        quant_type=AiterQuantType.BLOCK_1X32,
        activation=MoEActivation.SWIGLUOAI,
        quant_config=quant_config,
        **mxfp4_w4a16_extra_kwargs(quant_config, moe_config),
    )

    _assert_deterministic(run_moe, n_runs=N_CALLS)


@pytest.mark.parametrize("op_name", ["rms_norm", "fused_add_rms_norm"])
def test_rms_norm_is_repeatable(op_name, use_aiter):
    """RMSNorm through the platform's op priority without Inductor: AITER's
    kernel when it is preferred, else vLLM's C kernel."""
    config = VllmConfig(compilation_config=CompilationConfig(mode=CompilationMode.NONE))
    op = getattr(ir.ops, op_name)
    hidden_dim = 4096
    x = torch.randn(64, hidden_dim, dtype=torch.bfloat16)
    weight = torch.randn(hidden_dim, dtype=torch.bfloat16)
    args = (x, weight, 1e-6) if op_name == "rms_norm" else (x, x.flip(0), weight, 1e-6)

    with op.set_priority(getattr(config.kernel_config.ir_op_priority, op_name)):
        _assert_deterministic(op, *args, n_runs=N_CALLS)
