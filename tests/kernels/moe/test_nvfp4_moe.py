# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    break_fp4_bytes,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
    kE2M1ToFloat,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsFp4,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    ref_nvfp4_quant,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        "Nvfp4 Requires compute capability of 10 or above.", allow_module_level=True
    )

MNK_FACTORS = [
    (2, 1024, 1024),
    (2, 1024, 1536),
    (2, 3072, 1024),
    (64, 1024, 1024),
    (64, 3072, 1024),
    (64, 2048, 1536),
    (224, 1024, 1024),
    (224, 1024, 1536),
]


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [40, 64, 256])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_cutlass_fp4_moe_no_graph(
    m: int, n: int, k: int, e: int, topk: int, dtype: torch.dtype, workspace_init
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        quant_blocksize = 16

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
            make_test_weights(
                e,
                n,
                k,
                in_dtype=dtype,
                quant_dtype="nvfp4",
                block_shape=None,  # use quant_blocksize?
                per_out_ch_quant=False,
            )
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)

        assert w1_gs is not None
        assert w2_gs is not None
        assert w1_blockscale is not None
        assert w2_blockscale is not None

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs),
            g2_alphas=(1 / w2_gs),
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )
        moe_config = make_dummy_moe_config()

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
        )

        cutlass_output = kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=mk.MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

        # Reference check:
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)

        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                w1_blockscale[idx],
                w1_gs[idx],
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                w2_blockscale[idx],
                w2_gs[idx],
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(a_in_dtype, w1_d, w2_d, score, topk)

        torch.testing.assert_close(torch_output, cutlass_output, atol=1e-1, rtol=1e-1)


# step3.5-flash uses swiglustep activation (clipped SwiGLU with limit=7.0)
# for MoE layers 43-44. This tests the non-fused activation fallback path
# in run_cutlass_moe_fp4 (apply_moe_activation + separate fp4 quantization).
# Model dims: e=288, topk=8, n=1280 (moe_intermediate_size), k=4096 (hidden)
SWIGLUSTEP_MNK_FACTORS = [
    (2, 1280, 4096),
    (64, 1280, 4096),
    (224, 1280, 4096),
]


@pytest.mark.parametrize("m,n,k", SWIGLUSTEP_MNK_FACTORS)
@pytest.mark.parametrize("e", [64, 288])
@pytest.mark.parametrize("topk", [1, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_cutlass_fp4_moe_swiglustep(
    m: int, n: int, k: int, e: int, topk: int, dtype: torch.dtype, workspace_init
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        quant_blocksize = 16

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
            make_test_weights(
                e,
                n,
                k,
                in_dtype=dtype,
                quant_dtype="nvfp4",
                block_shape=None,
                per_out_ch_quant=False,
            )
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)

        assert w1_gs is not None
        assert w2_gs is not None
        assert w1_blockscale is not None
        assert w2_blockscale is not None

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs),
            g2_alphas=(1 / w2_gs),
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )

        kernel = mk.FusedMoEKernel(
            make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
            CutlassExpertsFp4(
                moe_config=make_dummy_moe_config(),
                quant_config=quant_config,
            ),
        )

        cutlass_output = kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SWIGLUSTEP,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        # Reference: dequantize everything and run torch_moe with swiglustep
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)

        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                w1_blockscale[idx],
                w1_gs[idx],
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                w2_blockscale[idx],
                w2_gs[idx],
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        torch_output = torch_moe(
            a_in_dtype,
            w1_d,
            w2_d,
            score,
            topk,
            activation=MoEActivation.SWIGLUSTEP,
        )

        torch.testing.assert_close(torch_output, cutlass_output, atol=1e-1, rtol=1e-1)


def _assert_unbiased(step_diff: torch.Tensor) -> None:
    """Assert signed one-step differences lean neither way: unbiased rounding
    mismatches are +-1 coin flips, so their sum stays within ~4 sigma, while a
    one-sided drift would shift every downstream GEMM sum."""
    assert step_diff.sum().abs() <= 4 * step_diff.abs().sum().clamp(min=1).sqrt()


def _assert_fp4_close(
    actual: torch.Tensor, expected: torch.Tensor, max_mismatch: float
) -> None:
    """Assert packed FP4 `actual` is at most one E2M1 grid step from
    `expected` (packed, or already decoded to float32), that fewer than
    `max_mismatch` of the values differ at all, and that the differences do
    not systematically grow or shrink magnitudes."""
    grid = kE2M1ToFloat.to(actual.device)

    def grid_index(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = break_fp4_bytes(x, torch.float32)
        return torch.searchsorted(grid, x.abs()) * x.sign()

    actual_idx, expected_idx = grid_index(actual), grid_index(expected)
    step_diff = (actual_idx - expected_idx).abs()
    assert step_diff.max() <= 1
    assert (step_diff != 0).float().mean() < max_mismatch
    _assert_unbiased(actual_idx.abs() - expected_idx.abs())


def _route_experts(
    m: int, e: int, k: int, topk: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a random routing and return (a_map, expert_offsets,
    blockscale_offsets), mirroring what CutlassExpertsFp4 does before
    calling into the per-expert quant kernels."""
    gating = torch.randn(m, e, device=device)
    topk_ids = torch.topk(gating, topk, dim=-1).indices.to(torch.int32)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        k,
        k,
        blockscale_offsets,
        is_gated=True,
    )
    return a_map, expert_offsets, blockscale_offsets


@pytest.mark.parametrize("m,k", [(2, 1024), (64, 1024), (224, 1536)])
@pytest.mark.parametrize("e", [8, 40])
@pytest.mark.parametrize("topk", [1, 6])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_scaled_fp4_experts_quant(
    m: int, k: int, e: int, topk: int, dtype: torch.dtype
) -> None:
    """scaled_fp4_experts_quant must match the pure-PyTorch ref_nvfp4_quant
    on each expert's rows, using that expert's global scale. Each expert's
    scale region is padded to 128 rows, i.e. a dense 128x4 swizzled layout."""
    set_random_seed(1)
    device = "cuda"

    hidden_states = torch.randn(m, k, device=device, dtype=dtype)
    a_map, expert_offsets, blockscale_offsets = _route_experts(m, e, k, topk, device)
    a = ops.shuffle_rows(hidden_states, a_map)

    amax = hidden_states.abs().max().float()
    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax) * (
        torch.rand(e, device=device, dtype=torch.float32) * 0.5 + 0.5
    )
    out, out_scale = ops.scaled_fp4_experts_quant(
        a, a_global_scale, expert_offsets, blockscale_offsets, topk
    )

    for i in range(e):
        row_lo, row_hi = int(expert_offsets[i]), int(expert_offsets[i + 1])
        sf_lo, sf_hi = int(blockscale_offsets[i]), int(blockscale_offsets[i + 1])
        count = row_hi - row_lo
        if count == 0:
            continue

        ref_out, ref_scale = ref_nvfp4_quant(
            a[row_lo:row_hi], a_global_scale[i], block_size=16
        )

        moe_scale = convert_swizzled_to_linear(
            out_scale[sf_lo:sf_hi], count, k, block_size=16
        ).float()

        torch.testing.assert_close(moe_scale, ref_scale, atol=0, rtol=0)
        # The kernel uses rcp.approx, so values within an ulp of an E2M1
        # rounding boundary may land one grid step away from the reference.
        _assert_fp4_close(out[row_lo:row_hi], ref_out, max_mismatch=1e-3)


@pytest.mark.parametrize("m,k", [(2, 1024), (64, 1024), (224, 1536)])
@pytest.mark.parametrize("e", [8, 40])
@pytest.mark.parametrize("topk", [1, 6])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@torch.inference_mode()
def test_silu_and_mul_scaled_fp4_experts_quant(
    m: int, k: int, e: int, topk: int, dtype: torch.dtype
) -> None:
    """silu_and_mul_scaled_fp4_experts_quant must match silu(gate) * up computed
    in fp64 and rounded once to dtype, followed by scaled_fp4_experts_quant.
    The kernel rounds once from fp32, so values and block scales may differ by
    one step near rounding boundaries. SiluAndMul.forward_native is not used:
    it rounds twice in dtype (after silu and after the multiply), which biases
    bf16 magnitudes low."""
    set_random_seed(2)
    device = "cuda"

    gate_up = torch.randn(m, 2 * k, device=device, dtype=dtype)
    a_map, expert_offsets, blockscale_offsets = _route_experts(m, e, k, topk, device)
    a = ops.shuffle_rows(gate_up, a_map)

    silu_mul = torch.nn.functional.silu(gate_up[:, :k]) * gate_up[:, k:]
    silu_mul_amax = silu_mul.abs().max().float()
    a_global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / silu_mul_amax) * (
        torch.rand(e, device=device, dtype=torch.float32) * 0.5 + 0.5
    )
    out, out_scale = ops.silu_and_mul_scaled_fp4_experts_quant(
        a, a_global_scale, expert_offsets, blockscale_offsets, topk
    )
    a64 = a.double()
    silu_mul_ref = (torch.nn.functional.silu(a64[:, :k]) * a64[:, k:]).to(dtype)
    ref_out, ref_scale = ops.scaled_fp4_experts_quant(
        silu_mul_ref,
        a_global_scale,
        expert_offsets,
        blockscale_offsets,
        topk,
    )

    _assert_fp4_close(out, ref_out, max_mismatch=2e-2)

    scale_diffs = []
    for i in range(e):
        count = int(expert_offsets[i + 1]) - int(expert_offsets[i])
        if count == 0:
            continue
        sf_lo, sf_hi = int(blockscale_offsets[i]), int(blockscale_offsets[i + 1])
        moe_code, ref_code = (
            convert_swizzled_to_linear(s[sf_lo:sf_hi], count, k, block_size=16)
            .view(torch.uint8)
            .int()
            .flatten()
            for s in (out_scale, ref_scale)
        )
        scale_diffs.append(moe_code - ref_code)
    scale_diff = torch.cat(scale_diffs)
    assert scale_diff.abs().max() <= 1
    assert (scale_diff != 0).float().mean() < 1e-1
    _assert_unbiased(scale_diff)


if __name__ == "__main__":
    test_cutlass_fp4_moe_no_graph((2, 1024, 1024), 40, 1, torch.half)
