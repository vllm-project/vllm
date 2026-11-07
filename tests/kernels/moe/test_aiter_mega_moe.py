# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-rank AITER MegaMoEV2 numerics against a dequantized PyTorch reference."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    parallel_launch_with_config,
)
from tests.kernels.moe.utils import make_dummy_moe_config
from tests.utils import multi_gpu_test
from vllm.config import KernelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import (
    PassThroughPrepareAndFinalize,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_mori
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

WORLD_SIZE = 2
NUM_EXPERTS = 32
ROUTED_PER_RANK = NUM_EXPERTS // WORLD_SIZE
MAX_NUM_TOKENS = 256
# Up to 128 tokens take the fixed-slot runtime, more take the compact one.
NUM_TOKENS = (1, 128, 256)
REL_L2_BUDGET = 0.02


def _quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    q, s = dynamic_mxfp4_quant(w.flatten(0, 1))
    return q.view(*w.shape[:2], -1), s.view(*w.shape[:2], -1)


def _dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    from aiter.utility import fp4_utils

    values = fp4_utils.mxfp4_to_f32(q).view(q.shape[0], -1)
    return values * fp4_utils.e8m0_to_f32(s).repeat_interleave(32, dim=-1)


def _mxfp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    return dequant_mxfp8_to_bf16(*_mxfp8_e4m3_quantize_torch(x)).float()


def _reference(x, weights, ids, w13, w2, swiglu_limit):
    """MXFP4 weights with MXFP8 activations, as MegaMoEV2 computes them.

    Every expert in w13/w2 is dequantized; ids index into them directly.
    """
    (w13_q, w13_s), (w2_q, w2_s) = w13, w2
    x = _mxfp8_round_trip(x)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in torch.unique(ids).tolist():
        rows, slots = torch.nonzero(ids == expert, as_tuple=True)
        gate, up = (x[rows] @ _dequantize(w13_q[expert], w13_s[expert]).T).chunk(
            2, dim=-1
        )
        if swiglu_limit > 0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        hidden = _mxfp8_round_trip(F.silu(gate) * up)
        y = hidden @ _dequantize(w2_q[expert], w2_s[expert]).T
        out.index_add_(0, rows, y * weights[rows, slots, None])
    return out


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected).norm() / expected.norm()).item()


def _mega_moe_worker(pgi, vllm_config, cpu_group, k, n, topk, swiglu_limit, fse):
    from vllm.model_executor.layers.fused_moe.experts.aiter_mega_moe import (
        AiterMegaMoEExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_quant_config,
    )

    init_workspace_manager(pgi.device)
    backend = Mxfp4MoeBackend.AITER_MEGA_MOE
    # Every rank builds the same global experts, plus one shared expert per
    # rank that differs between ranks so a remote shared route shows up.
    set_random_seed(7)
    total = NUM_EXPERTS + WORLD_SIZE
    w13 = _quantize(torch.randn(total, 2 * n, k, dtype=torch.bfloat16) * k**-0.25)
    w2 = _quantize(torch.randn(total, k, n, dtype=torch.bfloat16) * n**-0.25)

    start = pgi.rank * ROUTED_PER_RANK
    local = list(range(start, start + ROUTED_PER_RANK))
    if fse:
        local.append(NUM_EXPERTS + pgi.rank)
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        dp_size=WORLD_SIZE,
        dp_rank=pgi.rank,
        ep_size=WORLD_SIZE,
        ep_rank=pgi.rank,
        use_ep=True,
        all2all_backend="passthrough",
    )
    moe = replace(
        make_dummy_moe_config(
            num_experts=NUM_EXPERTS,
            num_local_experts=len(local),
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            max_num_tokens=MAX_NUM_TOKENS,
        ),
        moe_parallel_config=parallel,
    )
    w13_q, w2_q, w13_s, w2_s, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        backend,
        None,
        w13[0][local].contiguous(),
        w2[0][local].contiguous(),
        w13[1][local].contiguous(),
        w2[1][local].contiguous(),
    )
    quant = make_mxfp4_moe_quant_config(
        backend, w13_s, w2_s, swiglu_limit=swiglu_limit or None
    )
    assert quant is not None
    prepare = maybe_make_prepare_finalize(moe, quant, allow_new_interface=True)
    assert isinstance(prepare, PassThroughPrepareAndFinalize)
    experts = AiterMegaMoEExperts(moe, quant)
    assert experts.num_fused_shared_experts == int(fse)
    layer = torch.nn.Module()
    layer.w13_weight, layer.w2_weight = w13_q, w2_q
    layer.w13_weight_scale, layer.w2_weight_scale = w13_s, w2_s
    layer.apply_router_weight_on_input = False
    experts.process_weights_after_loading(layer)
    kernel = FusedMoEKernel(prepare, experts)

    def run(x, weights, ids):
        with set_forward_context(None, vllm_config):
            return kernel.apply(
                x,
                w13_q,
                w2_q,
                weights,
                ids,
                activation=MoEActivation.SILU,
                global_num_experts=NUM_EXPERTS,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

    for m in NUM_TOKENS:
        set_random_seed(101 + pgi.rank + m)
        x = torch.randn(m, k, dtype=torch.bfloat16)
        scores = torch.randn(m, NUM_EXPERTS)
        # Guarantee a route to both ranks, including the one-token case.
        scores[:, 0] = scores[:, ROUTED_PER_RANK] = float("inf")
        ids = scores.topk(topk, dim=-1).indices.to(torch.int32)
        weights = torch.randn(m, topk).softmax(dim=-1)

        actual = run(x, weights, ids)
        if fse:
            shared = torch.full((m, 1), NUM_EXPERTS + pgi.rank, dtype=ids.dtype)
            ids = torch.cat([ids, shared], dim=-1)
            weights = torch.cat([weights, torch.ones(m, 1)], dim=-1)
        rel_l2 = _rel_l2(actual, _reference(x, weights, ids, w13, w2, swiglu_limit))
        assert rel_l2 < REL_L2_BUDGET, f"m={m} relL2={rel_l2:.4f}"

    # Decode runs under CUDA graphs; replay must track new inputs.
    x = torch.randn(1, k, dtype=torch.bfloat16)
    weights, ids = torch.randn(1, NUM_EXPERTS).topk(topk, dim=-1)
    weights, ids = weights.softmax(dim=-1), ids.to(torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(x, weights, ids)
    x.mul_(-1.5)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(captured, run(x, weights, ids), atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_rocm() or not has_mori(), reason="Requires ROCm MoRI"
)
@pytest.mark.parametrize(
    "k,n,topk,swiglu_limit,fse",
    [
        pytest.param(4096, 2048, 6, 10.0, True, id="dsv4_fused_shared"),
        pytest.param(3584, 512, 8, 0.0, False, id="routed_no_clamp"),
    ],
)
@multi_gpu_test(num_gpus=WORLD_SIZE)
def test_aiter_mega_moe(k, n, topk, swiglu_limit, fse):
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("AITER MegaMoE requires gfx950")
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=WORLD_SIZE,
            enable_expert_parallel=True,
            all2all_backend="passthrough",
        ),
        kernel_config=KernelConfig(moe_backend="aiter_mega_moe"),
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=MAX_NUM_TOKENS
        ),
    )
    env = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MOE": "1"}
    parallel_launch_with_config(
        WORLD_SIZE, _mega_moe_worker, config, env, k, n, topk, swiglu_limit, fse
    )
