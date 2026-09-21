# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real multi-node MoRI dispatch/AITER GEMM/MoRI combine regression test for
the buffer-trim fix in rocm_aiter_moe.py / prepare_finalize/mori.py.

Drives the three real production primitives directly (MoriPrepareAndFinalize
.prepare()/.finalize() and AiterExperts.apply()) instead of going through
FusedMoEFactory/the full FusedMoE layer -- this is intentional, see
test_moe_layer_multinode.py's docstring for why that heavier construction
path doesn't work here (older pinned vLLM install used for multi-node CI
lacks FusedMoEFactory).

Reproduces on real 4-node/EP32 InterNodeV1 hardware: without the trim,
repeated dispatch/combine rounds on the same (reused) mori_op buffers
produce all-zero combined output for genuinely valid, real tokens (MoRI's
own reported valid-row count balloons toward the oversized buffer's full
capacity instead of the true tiny per-round count, and AITER's output for
those rows is discarded/corrupted downstream in combine()). A single round
right after construction is NOT enough to show this -- it takes at least
one prior large round to have populated/reused the buffer. With the trim
applied, results stay correct and nonzero across repeated rounds.

Launch (mirrors test_moe_layer_multinode.py -- one torchrun per host, all
pointed at the same rendezvous, one host acting as rank 0):

    MASTER_ADDR=<node0-ip> MASTER_PORT=29525 \\
    torchrun --nnodes 4 --nproc-per-node 8 --node-rank <0..3> \\
      --rdzv_backend=c10d --rdzv_endpoint=<node0-ip>:29525 \\
      -m pytest -v -s kernels/moe/test_mori_aiter_ep_trim_regression.py
"""

import importlib.util
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mori") is None, reason="mori not installed"
)

GPU_PER_NODE = 8
HIDDEN = 4096
INTER = 1024
TOPK = 8
EXPERTS_PER_RANK = 8
MAX_TOK = 128
NUM_BIG_SMALL_ROUNDS = 3


def _make_topk_ids(num_tokens, num_experts, topk):
    logits = torch.randn(num_tokens, num_experts, device="cuda")
    _, ids = torch.topk(torch.softmax(logits, dim=-1), k=topk, dim=-1)
    return ids.to(torch.int32)


def test_mori_dispatch_aiter_combine_trim_regression():
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size == 0:
        pytest.skip(
            "must be launched under torchrun (WORLD_SIZE unset) -- see "
            "this file's module docstring"
        )
    if world_size // GPU_PER_NODE < 2:
        pytest.skip(
            f"world_size={world_size} implies <2 nodes -- this test needs "
            "genuine multi-node MoRI InterNodeV1 traffic (validated at "
            "EP32/4 nodes) to reproduce the bug"
        )

    import mori
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
    torch._C._distributed_c10d._register_process_group("mori", cpu_group)
    mori.shmem.shmem_torch_process_group_init("mori")

    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        AiterExperts,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.mori import (
        MoriPrepareAndFinalize,
    )

    from .test_rocm_aiter_moe import _assert_aiter_supported, _shuffle_moe_weights
    from .utils import make_dummy_moe_config

    _assert_aiter_supported()

    num_experts = EXPERTS_PER_RANK * world_size
    mori_config = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=HIDDEN,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=MAX_TOK,
        num_experts_per_rank=EXPERTS_PER_RANK,
        num_experts_per_token=TOPK,
        warp_num_per_block=16,
        block_num=32,
        rdma_block_num=16,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1,
        gpu_per_node=GPU_PER_NODE,
    )
    op = mori.ops.EpDispatchCombineOp(mori_config)
    pf = MoriPrepareAndFinalize(
        op, max_tokens_per_rank=MAX_TOK, num_dispatchers=world_size
    )

    torch.manual_seed(rank + 1)
    w1 = torch.randn(
        EXPERTS_PER_RANK, INTER * 2, HIDDEN, dtype=torch.bfloat16, device="cuda"
    ) / (HIDDEN**0.5)
    w2 = torch.randn(
        EXPERTS_PER_RANK, HIDDEN, INTER, dtype=torch.bfloat16, device="cuda"
    ) / (INTER**0.5)
    w1s, w2s = _shuffle_moe_weights(w1, w2)

    moe_config = make_dummy_moe_config(
        num_experts=num_experts,
        num_local_experts=EXPERTS_PER_RANK,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTER,
        in_dtype=torch.bfloat16,
        max_num_tokens=MAX_TOK,
    )
    experts = AiterExperts(
        moe_config=moe_config, quant_config=FUSED_MOE_UNQUANTIZED_CONFIG
    )
    vllm_config = VllmConfig()

    def round_trip(n_real, marker):
        a1 = torch.zeros(MAX_TOK, HIDDEN, dtype=torch.bfloat16, device="cuda")
        ids = torch.zeros(MAX_TOK, TOPK, dtype=torch.int32, device="cuda")
        weights = torch.zeros(MAX_TOK, TOPK, dtype=torch.float32, device="cuda")
        if n_real > 0:
            a1[:n_real].fill_(marker)
            ids[:n_real] = _make_topk_ids(n_real, num_experts, TOPK)
            weights[:n_real] = 1.0 / TOPK

        (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        ) = pf.prepare(
            a1, weights, ids, num_experts, None, False, FUSED_MOE_UNQUANTIZED_CONFIG
        )

        m = dispatch_a1.shape[0]
        ws13, ws2, out_shape = experts.workspace_shapes(
            M=m,
            N=INTER,
            K=HIDDEN,
            topk=TOPK,
            global_num_experts=num_experts,
            local_num_experts=EXPERTS_PER_RANK,
            expert_tokens_meta=expert_tokens_meta,
            activation=MoEActivation.SILU,
        )
        workspace13 = torch.empty(ws13, dtype=torch.bfloat16, device="cuda")
        workspace2 = torch.empty(ws2, dtype=torch.bfloat16, device="cuda")
        aiter_out = torch.full(
            out_shape, float("nan"), dtype=torch.bfloat16, device="cuda"
        )

        with set_forward_context(None, vllm_config, num_tokens=m):
            experts.apply(
                output=aiter_out,
                hidden_states=dispatch_a1,
                w1=w1s,
                w2=w2s,
                topk_weights=dispatch_weights,
                topk_ids=dispatch_ids,
                activation=MoEActivation.SILU,
                global_num_experts=num_experts,
                expert_map=None,
                a1q_scale=dispatch_scale,
                a2_scale=None,
                workspace13=workspace13,
                workspace2=workspace2,
                expert_tokens_meta=expert_tokens_meta,
                apply_router_weight_on_input=False,
            )

        final_out = torch.zeros(MAX_TOK, HIDDEN, dtype=torch.bfloat16, device="cuda")
        pf.finalize(final_out, aiter_out, weights, ids, False, None)
        torch.accelerator.synchronize()
        dist.barrier()
        return final_out[:n_real].float() if n_real > 0 else final_out[:0].float()

    # warmup, then repeated (big, small) cycles reusing the same mori_op /
    # dispatch buffers -- validated on real EP32/4-node hardware to require
    # >=1 prior big round before a small round's real tokens get corrupted
    # (a single round right after construction is not enough).
    round_trip(MAX_TOK, 5.0)
    for _ in range(NUM_BIG_SMALL_ROUNDS):
        round_trip(MAX_TOK, 99.0)
        small_out = round_trip(4, 1.0)
        assert torch.isfinite(small_out).all()
        assert small_out.abs().sum() > 0, (
            "MoRI dispatch/combine returned all-zero output for real "
            "tokens after buffer reuse -- the dispatch-output trim in "
            "MoriPrepareAndFinalize.prepare() appears to be missing"
        )

    dist.barrier()
    dist.destroy_process_group()
