# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.batch_invariant_fp4_moe import (
    NvFP4MoEWorkspace,
    _grouped_matmul_nvfp4_packed,
    _map_extra_rows,
    fused_moe_batch_invariant_nvfp4,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.triton_utils.allocation import set_triton_allocator
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Batch-invariant FP4 MoE requires Blackwell or newer (sm100+).",
        allow_module_level=True,
    )

if not hasattr(tl, "dot_scaled"):
    pytest.skip(
        reason="Installed Triton build does not expose tl.dot_scaled.",
        allow_module_level=True,
    )

DTYPE = torch.bfloat16
DEVICE = "cuda:0"
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FLOAT4_E2M1_MAX = 6.0
BATCH_INVARIANT_CASES = [
    (1, False),
    (2, False),
    (4, False),
    (1, True),
]


@pytest.fixture(autouse=True)
def _triton_allocator_for_tma_kernels():
    """tl.make_tensor_descriptor in grouped NVFP4 GEMM needs triton.set_allocator."""
    set_triton_allocator(torch.device(DEVICE))
    yield


def _batch_invariant_fp4_workspaces(
    m: int,
    topk: int,
    w1_row_dim: int,
    hidden_dim: int,
    activation: MoEActivation,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scratch buffers matching ``BatchInvariantNvfp4Experts.workspace_shapes``."""
    act_out_dim = mk.FusedMoEExpertsModular.adjust_N_for_activation(
        w1_row_dim, activation
    )
    m_total = m * topk
    cols13 = max(w1_row_dim, hidden_dim)
    extra = _map_extra_rows(m_total, cols13)
    workspace13 = torch.empty((m_total + extra, cols13), device=device, dtype=dtype)
    workspace2 = torch.empty(
        (m_total, max(act_out_dim, hidden_dim)), device=device, dtype=dtype
    )
    return workspace13, workspace2


def _make_nvfp4_workspace(
    num_experts: int,
    device: torch.device | str = DEVICE,
) -> NvFP4MoEWorkspace:
    """Create an ``NvFP4MoEWorkspace`` for test helpers."""
    return NvFP4MoEWorkspace(
        expert_offsets=torch.empty(num_experts + 1, dtype=torch.int32, device=device),
        blockscale_offsets=torch.empty(
            num_experts + 1, dtype=torch.int32, device=device
        ),
        problem_sizes1=torch.empty(num_experts, 3, dtype=torch.int32, device=device),
        problem_sizes2=torch.empty(num_experts, 3, dtype=torch.int32, device=device),
        tiles_per_expert=torch.empty(num_experts, dtype=torch.int32, device=device),
        expert_tile_start=torch.zeros(
            num_experts + 1, dtype=torch.int32, device=device
        ),
        dummy_bias=torch.empty(0, device=device, dtype=torch.float32),
    )


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.abs().max()).to(torch.float32)


def _make_nvfp4_weights_and_scales(
    *,
    e: int,
    n: int,
    k: int,
    scale_source: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
        make_test_weights(
            e,
            n,
            k,
            in_dtype=DTYPE,
            quant_dtype="nvfp4",
            block_shape=None,
            per_out_ch_quant=False,
        )
    )
    assert w1_blockscale is not None and w2_blockscale is not None
    assert w1_gs is not None and w2_gs is not None

    a1_gscale = _global_scale(scale_source).expand(e).contiguous()
    a2_gscale = _global_scale(scale_source).expand(e).contiguous()
    g1_alphas = ((1.0 / w1_gs) / a1_gscale).to(torch.float32)
    g2_alphas = ((1.0 / w2_gs) / a2_gscale).to(torch.float32)
    return (
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    )


def _make_nvfp4_moe_tensors(
    *,
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    hidden_states = torch.randn((m, k), device=DEVICE, dtype=DTYPE) / 10
    (
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_weights_and_scales(
        e=e,
        n=n,
        k=k,
        scale_source=hidden_states,
    )

    score = torch.randn((m, e), device=DEVICE, dtype=DTYPE)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, score, topk, renormalize=False
    )

    return (
        hidden_states,
        topk_weights,
        topk_ids,
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    )


def _run_batch_invariant_nvfp4(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_q: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_q: torch.Tensor,
    w2_blockscale: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    g1_alphas: torch.Tensor,
    g2_alphas: torch.Tensor,
    activation: MoEActivation = MoEActivation.SILU,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    workspace13, workspace2 = _batch_invariant_fp4_workspaces(
        m=hidden_states.shape[0],
        topk=topk_ids.shape[1],
        w1_row_dim=w1_q.shape[1],
        hidden_dim=hidden_states.shape[1],
        activation=activation,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    output = torch.empty_like(hidden_states)
    fused_moe_batch_invariant_nvfp4(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_weight=w1_q,
        w13_weight_scale=w1_blockscale,
        w2_weight=w2_q,
        w2_weight_scale=w2_blockscale,
        a1_gscale=a1_gscale,
        g1_alphas=g1_alphas,
        a2_gscale=a2_gscale,
        g2_alphas=g2_alphas,
        activation=activation,
        workspace13=workspace13,
        workspace2=workspace2,
        output=output,
        workspace=_make_nvfp4_workspace(w1_q.shape[0]),
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=expert_map,
    )
    return output


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    BATCH_INVARIANT_CASES,
)
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_matches_cutlass(
    workspace_init,
    topk: int,
    apply_router_weight_on_input: bool,
) -> None:
    set_random_seed(11)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        (
            hidden_states,
            topk_weights,
            topk_ids,
            w1_q,
            w1_blockscale,
            w2_q,
            w2_blockscale,
            a1_gscale,
            a2_gscale,
            g1_alphas,
            g2_alphas,
        ) = _make_nvfp4_moe_tensors(m=32, n=128, k=128, e=8, topk=topk)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=g1_alphas,
            g2_alphas=g2_alphas,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )
        kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            CutlassExpertsFp4(
                moe_config=make_dummy_moe_config(),
                quant_config=quant_config,
            ),
            inplace=False,
        )

        cutlass_out = kernel.apply(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=8,
            expert_map=None,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        fallback_out = _run_batch_invariant_nvfp4(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1_q=w1_q,
            w1_blockscale=w1_blockscale,
            w2_q=w2_q,
            w2_blockscale=w2_blockscale,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
            g1_alphas=g1_alphas,
            g2_alphas=g2_alphas,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        torch.testing.assert_close(fallback_out, cutlass_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "topk,apply_router_weight_on_input",
    BATCH_INVARIANT_CASES,
)
@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_batch_size_invariance(
    topk: int,
    apply_router_weight_on_input: bool,
) -> None:
    set_random_seed(13)
    e, n, k = 8, 128, 128
    x_single = torch.randn((1, k), device=DEVICE, dtype=DTYPE) / 10
    x_batch = torch.cat(
        [x_single, torch.randn((7, k), device=DEVICE, dtype=DTYPE) / 10], dim=0
    )

    (
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_weights_and_scales(
        e=e,
        n=n,
        k=k,
        scale_source=x_batch,
    )

    topk_ids_single = torch.randint(0, e, (1, topk), device=DEVICE, dtype=torch.int32)
    topk_ids_batch = torch.cat(
        [
            topk_ids_single,
            torch.randint(0, e, (7, topk), device=DEVICE, dtype=torch.int32),
        ],
        dim=0,
    )

    topk_weights_single = torch.rand((1, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_single /= topk_weights_single.sum(dim=-1, keepdim=True)
    topk_weights_batch = torch.rand((8, topk), device=DEVICE, dtype=torch.float32)
    topk_weights_batch /= topk_weights_batch.sum(dim=-1, keepdim=True)
    topk_weights_batch[0] = topk_weights_single[0]

    out_single = _run_batch_invariant_nvfp4(
        hidden_states=x_single,
        topk_weights=topk_weights_single,
        topk_ids=topk_ids_single,
        w1_q=w1_q,
        w1_blockscale=w1_blockscale,
        w2_q=w2_q,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    out_batch = _run_batch_invariant_nvfp4(
        hidden_states=x_batch,
        topk_weights=topk_weights_batch,
        topk_ids=topk_ids_batch,
        w1_q=w1_q,
        w1_blockscale=w1_blockscale,
        w2_q=w2_q,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    assert torch.equal(out_single[0], out_batch[0])


@torch.inference_mode()
def test_batch_invariant_nvfp4_moe_all_invalid_routes_return_zero() -> None:
    set_random_seed(31)
    m, e, n, k = 16, 8, 128, 128
    (
        hidden_states,
        _,
        _,
        w1_q,
        w1_blockscale,
        w2_q,
        w2_blockscale,
        a1_gscale,
        a2_gscale,
        g1_alphas,
        g2_alphas,
    ) = _make_nvfp4_moe_tensors(m=m, n=n, k=k, e=e, topk=1)

    topk_ids = torch.full((m, 2), -1, dtype=torch.int32, device=DEVICE)
    topk_weights = torch.rand((m, 2), dtype=torch.float32, device=DEVICE)
    out = _run_batch_invariant_nvfp4(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_q=w1_q,
        w1_blockscale=w1_blockscale,
        w2_q=w2_q,
        w2_blockscale=w2_blockscale,
        a1_gscale=a1_gscale,
        a2_gscale=a2_gscale,
        g1_alphas=g1_alphas,
        g2_alphas=g2_alphas,
    )

    torch.testing.assert_close(out, torch.zeros_like(out), atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_grouped_matmul_nvfp4_packed_matches_cutlass_reference() -> None:
    set_random_seed(17)

    # Non-uniform M across experts exercises offset/problem-size dispatch.
    per_expert_rows = [17, 131, 64, 5]
    E = len(per_expert_rows)
    N = 128
    K = 256

    xs = [
        torch.randn((rows, K), dtype=DTYPE, device=DEVICE) for rows in per_expert_rows
    ]
    ws = [torch.randn((N, K), dtype=DTYPE, device=DEVICE) for _ in range(E)]

    input_gs = [_global_scale(x) for x in xs]
    weight_gs = [_global_scale(w) for w in ws]
    alphas = [
        (1.0 / (ig * wg)).to(torch.float32) for ig, wg in zip(input_gs, weight_gs)
    ]

    packed_a_fp4 = []
    packed_a_scale = []
    packed_b_fp4 = []
    packed_b_scale = []
    ref_outputs = []
    expert_offsets = []
    a_scale_offsets = []

    row_offset = 0
    scale_row_offset = 0
    for expert_id, rows in enumerate(per_expert_rows):
        a_fp4, a_scale = scaled_fp4_quant(
            xs[expert_id], input_gs[expert_id], is_sf_swizzled_layout=True
        )
        b_fp4, b_scale_raw = scaled_fp4_quant(ws[expert_id], weight_gs[expert_id])
        b_scale = swizzle_blockscale(b_scale_raw)
        b_fp4, weights_padding_cols = pad_nvfp4_weight_for_cutlass(b_fp4)
        a_fp4 = pad_nvfp4_activation_for_cutlass(a_fp4, weights_padding_cols)

        # Reference: one CUTLASS launch per expert.
        ref = cutlass_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            a_scale,
            b_scale,
            alphas[expert_id],
            DTYPE,
        )
        ref_outputs.append(slice_nvfp4_output(ref, N))

        packed_a_fp4.append(a_fp4)
        packed_a_scale.append(a_scale)
        packed_b_fp4.append(b_fp4)
        packed_b_scale.append(b_scale)
        expert_offsets.append(row_offset)
        a_scale_offsets.append(scale_row_offset)
        row_offset += rows
        scale_row_offset += a_scale.shape[0]

    expert_offsets.append(row_offset)
    a_scale_offsets.append(scale_row_offset)

    packed_a_fp4_t = torch.cat(packed_a_fp4, dim=0)
    packed_a_scale_t = torch.cat(packed_a_scale, dim=0)
    packed_b_fp4_t = torch.stack(packed_b_fp4, dim=0)
    packed_b_scale_t = torch.stack(packed_b_scale, dim=0)
    alpha_t = torch.stack(alphas)
    expert_offsets_t = torch.tensor(expert_offsets, dtype=torch.int32, device=DEVICE)
    a_scale_offsets_t = torch.tensor(a_scale_offsets, dtype=torch.int32, device=DEVICE)
    problem_sizes_t = torch.tensor(
        [[rows, N, K] for rows in per_expert_rows],
        dtype=torch.int32,
        device=DEVICE,
    )

    packed_out = torch.empty((row_offset, N), device=DEVICE, dtype=DTYPE)
    _grouped_matmul_nvfp4_packed(
        a_fp4=packed_a_fp4_t,
        b_fp4=packed_b_fp4_t,
        a_scale=packed_a_scale_t,
        b_scale=packed_b_scale_t,
        alpha=alpha_t,
        expert_offsets=expert_offsets_t,
        a_scale_offsets=a_scale_offsets_t,
        problem_sizes=problem_sizes_t,
        output=packed_out,
        tiles_per_expert=torch.empty(E, dtype=torch.int32, device=DEVICE),
        expert_tile_start=torch.zeros(E + 1, dtype=torch.int32, device=DEVICE),
        dummy_bias=torch.empty(0, device=DEVICE, dtype=torch.float32),
    )
    if packed_out.shape[1] != N:
        packed_out = packed_out[:, :N].contiguous()

    ref_cat = torch.cat(ref_outputs, dim=0)
    torch.testing.assert_close(packed_out, ref_cat, atol=1e-1, rtol=1e-1)
