# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random
from dataclasses import replace
from datetime import timedelta

import pytest
import torch

from tests.utils import ensure_current_vllm_config, multi_gpu_test
from vllm import _custom_ops as ops
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.lora.ops.triton_ops import fused_moe_lora
from vllm.lora.ops.triton_ops.utils import FUSED_MOE_LORA_TWO_KERNEL_MAX_TOKENS
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed

# The tensor-parallel cases build their distributed environment inside processes
# spawned by torch.multiprocessing.spawn, so this process never has one to tear down.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("num_tokens", [1, 5])
@pytest.mark.parametrize("experts", [4, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "capture", [False, True] if current_platform.is_cuda() else [False]
)
@ensure_current_vllm_config()
def test_routed_expert_projections_match_composed_weights(
    num_tokens, experts, dtype, capture
):
    _check_routed_expert_projections(num_tokens, experts, dtype, capture)


def _check_routed_expert_projections(
    num_tokens, experts, dtype, capture, tp_rank=0, tp_size=1, fully_sharded=False
):
    """Compose expert weights before activation, including an initially idle route."""
    from vllm.config import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper import get_punica_wrapper
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    set_random_seed(42)
    device = current_platform.device_type
    hidden, intermediate, rank, topk = 32, 32, 8, 2

    def weight(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.05

    wrapper = get_punica_wrapper(
        num_tokens,
        num_tokens,
        device,
        lora_config=LoRAConfig(max_loras=2, max_lora_rank=8, max_loras_per_request=2),
    )
    wrapper.update_metadata(
        LoRAMapping((0,) * num_tokens, (0,) * num_tokens, is_prefill=False),
        [1, 2],
        3,
        0,
    )
    config = FusedMoEConfig(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=experts,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Unspecified,
        moe_parallel_config=replace(
            FusedMoEParallelConfig.make_no_parallel(), tp_size=tp_size, tp_rank=tp_rank
        ),
        in_dtype=dtype,
    )
    layer = TritonExperts(config, FusedMoEQuantConfig.make())
    slots = torch.zeros(num_tokens, 2, dtype=torch.int32, device=device)
    slots[:, 1] = -1
    coefficients = torch.full((num_tokens, 2), 0.5, device=device)
    ctx = MoELoRAContext(
        w13_lora_a_stacked=tuple(weight(2, experts, rank, hidden) for _ in range(2)),
        w13_lora_b_stacked=tuple(
            weight(2, experts, intermediate, rank) for _ in range(2)
        ),
        w2_lora_a_stacked=(weight(2, experts, rank, intermediate),),
        w2_lora_b_stacked=(weight(2, experts, hidden, rank),),
        adapter_enabled=torch.ones(3, dtype=torch.int32, device=device),
        max_loras=2,
        top_k=topk,
        w13_num_slices=2,
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=experts,
        punica_wrapper=wrapper,
        use_tuned_config=False,
        local_lora_routes=(slots, coefficients),
    )
    x = weight(num_tokens, hidden)
    ids = (
        torch.arange(num_tokens * topk, device=device).view(num_tokens, topk) % experts
    )
    expert_weights = torch.full((num_tokens, topk), 0.5, device=device)
    w1, w2 = (
        weight(experts, 2 * intermediate, hidden),
        weight(experts, hidden, intermediate),
    )
    reference_ctx, reference_w1, reference_w2 = ctx, w1, w2
    local_intermediate = intermediate // tp_size
    partition = slice(tp_rank * local_intermediate, (tp_rank + 1) * local_intermediate)
    rank_part = slice(tp_rank * rank // tp_size, (tp_rank + 1) * rank // tp_size)
    hidden_part = slice(tp_rank * hidden // tp_size, (tp_rank + 1) * hidden // tp_size)
    if tp_size > 1:
        ctx = replace(
            ctx,
            tp_rank=tp_rank,
            tp_size=tp_size,
            fully_sharded=fully_sharded,
            w13_lora_a_stacked=tuple(
                a[:, :, rank_part, :].contiguous() if fully_sharded else a
                for a in ctx.w13_lora_a_stacked
            ),
            w13_lora_b_stacked=tuple(
                b[:, :, partition, :].contiguous() for b in ctx.w13_lora_b_stacked
            ),
            w2_lora_a_stacked=tuple(
                a[:, :, :, partition].contiguous() for a in ctx.w2_lora_a_stacked
            ),
            w2_lora_b_stacked=tuple(
                b[:, :, hidden_part, :].contiguous() if fully_sharded else b
                for b in ctx.w2_lora_b_stacked
            ),
        )
        w1 = torch.cat([w[:, partition, :] for w in w1.chunk(2, dim=1)], dim=1)
        w2 = w2[:, :, partition].contiguous()
    base_up = torch.einsum("th,tkoh->tko", x, w1[ids])
    up = torch.empty_like(base_up)
    down = torch.empty(num_tokens, topk, hidden, device=device, dtype=dtype)

    def run():
        up.copy_(base_up)
        state = layer.apply_w13_lora(
            ctx,
            y=up,
            x=x,
            topk_ids=ids,
            topk_weights=expert_weights,
            expert_map=None,
            w1=w1,
            w2=w2,
            num_tokens=num_tokens,
            top_k_num=topk,
        )
        gate, value = up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * value
        down.copy_(torch.einsum("tki,tkhi->tkh", activated, w2[ids]))
        down.mul_(expert_weights[..., None])
        layer.apply_w2_lora(
            ctx,
            y=down,
            x=activated.reshape(-1, local_intermediate),
            topk_weights=expert_weights,
            lora_metadata=state,
            num_tokens=num_tokens,
            w1=w1,
            w2=w2,
            top_k_num=topk,
        )
        if tp_size > 1:
            down.copy_(tensor_model_parallel_all_reduce(down))

    run()
    graph = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    slots[:, 1] = 1
    coefficients[:, 0] = 0.7
    coefficients[:, 1] = 0.3
    slots[1::3] = -1
    slots[2::3, 1] = -1
    coefficients[2::3, 0] = 1.0
    if graph is not None:
        graph.replay()
    else:
        run()

    # Keep the oracle independent of device-specific GEMM implementations.
    ref_ids = ids.cpu()
    merged_up = reference_w1.float().cpu()[ref_ids]
    merged_down = reference_w2.float().cpu()[ref_ids]
    for route in range(2):
        adapter = slots[:, route].cpu()
        coefficient = coefficients[:, route].cpu() * (adapter >= 0)
        adapter = adapter.clamp(min=0)
        for part in range(2):
            merged_up[:, :, part * intermediate : (part + 1) * intermediate].add_(
                coefficient[:, None, None, None]
                * (
                    reference_ctx.w13_lora_b_stacked[part]
                    .float()
                    .cpu()[adapter[:, None], ref_ids]
                    @ reference_ctx.w13_lora_a_stacked[part]
                    .float()
                    .cpu()[adapter[:, None], ref_ids]
                )
            )
        merged_down.add_(
            coefficient[:, None, None, None]
            * (
                reference_ctx.w2_lora_b_stacked[0]
                .float()
                .cpu()[adapter[:, None], ref_ids]
                @ reference_ctx.w2_lora_a_stacked[0]
                .float()
                .cpu()[adapter[:, None], ref_ids]
            )
        )
    expected_up = torch.einsum("th,tkoh->tko", x.float().cpu(), merged_up)
    gate, value = expected_up.chunk(2, dim=-1)
    expected = (
        torch.einsum(
            "tki,tkhi->tkh", torch.nn.functional.silu(gate) * value, merged_down
        )
        * expert_weights.cpu()[..., None]
    )
    actual_up = up
    if tp_size > 1:
        actual_up = torch.cat(
            [tensor_model_parallel_all_gather(p.contiguous()) for p in up.chunk(2, -1)],
            dim=-1,
        )
    torch.testing.assert_close(
        actual_up.float().cpu(), expected_up, atol=1e-4, rtol=1e-2
    )
    torch.testing.assert_close(down.float().cpu(), expected, atol=1e-5, rtol=1e-2)


def _run_routed_expert_tp_rank(local_rank, port, dtype, fully_sharded):
    torch.accelerator.set_device_index(local_rank)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=2,
                rank=local_rank,
                local_rank=local_rank,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend=current_platform.dist_backend,
                timeout=timedelta(seconds=180),
            )
            initialize_model_parallel(2, 1)
            _check_routed_expert_projections(
                5, 4, dtype, False, local_rank, 2, fully_sharded
            )
    finally:
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fully_sharded", [False, True])
def test_routed_expert_projections_tp2(dtype, fully_sharded):
    """Real projection collectives preserve the composed unsharded result."""
    torch.multiprocessing.spawn(
        _run_routed_expert_tp_rank,
        args=(get_open_port(), dtype, fully_sharded),
        nprocs=2,
    )


def _run_routed_expert_ep_rank(pgi, vllm_config, cpu_group, different_slots, use_graph):
    from vllm.config import LoRAConfig
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.lora.layers import LoRAMapping, LoRARouteMapping
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.lora_weights import LoRALayerWeights
    from vllm.lora.model_manager import LoRAModelManager, SupportsLoRAModel
    from vllm.lora.request import LoRARequest
    from vllm.lora.routing_utils import DistributedLoRAPlan
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )
    from vllm.model_executor.layers.linear import ReplicatedLinear

    set_random_seed(42)
    device, dtype = pgi.device, torch.bfloat16
    hidden, intermediate, rank, experts, topk = 32, 32, 8, 4, 2
    counts = (3, 5)
    local = slice(sum(counts[: pgi.rank]), sum(counts[: pgi.rank + 1]))
    expert_slice = slice(pgi.rank * 2, (pgi.rank + 1) * 2)

    def weight(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.05

    # Use the allocator's real activation order, not a hand-written slot table.
    model = SupportsLoRAModel()
    model.config = vllm_config.model_config.hf_config
    model.add_module("proj", ReplicatedLinear(hidden, hidden, bias=False))
    manager = LoRAModelManager(
        model,
        8,
        8,
        32,
        LoRAConfig(
            max_loras=2, max_lora_rank=8, max_loras_per_request=2, lora_dtype=dtype
        ),
        device,
        vllm_config,
    )
    order = [2, 1] if different_slots and pgi.rank else [1, 2]
    for adapter_id in order:
        manager.add_adapter(
            LoRAModel(
                adapter_id,
                rank,
                {
                    "proj": LoRALayerWeights(
                        "proj",
                        rank,
                        rank,
                        torch.zeros(rank, hidden, device=device),
                        torch.zeros(hidden, rank, device=device),
                    )
                },
            )
        )
        manager.activate_adapter(adapter_id)
    wrapper = manager.punica_wrapper_mapping["language_model"]
    resident_ids = manager.lora_index_to_id
    assert sorted(i for i in resident_ids if i is not None) == [1, 2]
    slot_order = torch.tensor(
        [i - 1 for i in resident_ids if i is not None], device=device
    )
    full_a13 = tuple(weight(2, experts, rank, hidden) for _ in range(2))
    full_b13 = tuple(weight(2, experts, intermediate, rank) for _ in range(2))
    full_a2, full_b2 = (
        weight(2, experts, rank, intermediate),
        weight(2, experts, hidden, rank),
    )
    ctx = MoELoRAContext(
        w13_lora_a_stacked=tuple(
            a[slot_order, expert_slice].contiguous() for a in full_a13
        ),
        w13_lora_b_stacked=tuple(
            b[slot_order, expert_slice].contiguous() for b in full_b13
        ),
        w2_lora_a_stacked=(full_a2[slot_order, expert_slice].contiguous(),),
        w2_lora_b_stacked=(full_b2[slot_order, expert_slice].contiguous(),),
        adapter_enabled=torch.ones(3, dtype=torch.int32, device=device),
        max_loras=2,
        top_k=topk,
        w13_num_slices=2,
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=2,
        punica_wrapper=wrapper,
        use_tuned_config=False,
    )
    config = FusedMoEConfig(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=2,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Unspecified,
        moe_parallel_config=replace(
            FusedMoEParallelConfig.make_no_parallel(),
            use_ep=True,
            ep_size=2,
            ep_rank=pgi.rank,
            dp_size=2,
        ),
        in_dtype=dtype,
    )
    layer = TritonExperts(config, FusedMoEQuantConfig.make())
    prepare = MoEPrepareAndFinalizeNaiveDPEPModular()
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(device)
    layer.set_lora_context(ctx)
    prepare.set_lora_context(ctx)
    kernel = FusedMoEKernel(prepare, layer)
    expert_map = torch.full((experts,), -1, dtype=torch.int32, device=device)
    expert_map[expert_slice] = torch.arange(2, device=device, dtype=torch.int32)
    x, w1, w2 = (
        weight(8, hidden),
        weight(experts, 2 * intermediate, hidden),
        weight(experts, hidden, intermediate),
    )
    ids = torch.tensor([[0, 2], [3, 1]] * 4, dtype=torch.int32, device=device)
    expert_weights = torch.tensor([[0.3, 0.7]] * 8, device=device)
    graph = None

    def run():
        return kernel.apply(
            x[local].clone(),
            w1[expert_slice].contiguous(),
            w2[expert_slice].contiguous(),
            expert_weights[local],
            ids[local],
            MoEActivation.SILU,
            experts,
            expert_map,
            False,
        )

    for step in range(3):
        adapter_ids = [(1, 2), (0, 0), (2, 0), (2, 1), (1, 2), (0, 0), (1, 0), (2, 1)]
        coefficients = [
            (0.7, 0.3),
            (0.0, 0.0),
            (1.0, 0.0),
            (0.2, 0.8),
            (0.0, 1.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (0.4, 0.6),
        ]
        if step == 1:
            adapter_ids = [(b, a) for a, b in adapter_ids]
        if step == 2:
            # Rank 1 switches to an entirely scalar/base batch after routing.
            adapter_ids[3:] = [(1, 0), (0, 0), (2, 0), (0, 0), (1, 0)]
            coefficients[3:] = [(1.0, 0.0)] * 5
        mapping = LoRARouteMapping(
            tuple(adapter_ids[local]), tuple(coefficients[local]), (), False
        )
        if step == 2 and pgi.rank == 1:
            mapping = LoRAMapping(tuple(a for a, _ in adapter_ids[local]), (), False)
        manager.set_adapter_mapping(mapping)
        plan = DistributedLoRAPlan.from_rank_requests(
            [
                (
                    2,
                    (
                        LoRARequest("a", 1, "/a"),
                        LoRARequest("b", 2, "/b"),
                    ),
                )
            ]
        )
        assert wrapper.distributed_lora_tensors is not None
        wrapper.distributed_lora_tensors.update(plan, mapping, resident_ids)
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=counts[pgi.rank],
            num_tokens_across_dp=torch.tensor(counts, dtype=torch.int32, device="cpu"),
        ):
            dp_metadata = get_forward_context().dp_metadata
            assert dp_metadata is not None
            with dp_metadata.sp_local_sizes(1):
                if use_graph:
                    if graph is None:
                        for _ in range(3):
                            run()
                        torch.accelerator.synchronize()
                        torch.distributed.barrier(group=cpu_group)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            actual = run()
                    graph.replay()
                else:
                    actual = run()
        expected = []
        for token in range(local.start, local.stop):
            output = torch.zeros(hidden, device=device, dtype=torch.float32)
            for choice, expert in enumerate(ids[token].tolist()):
                up, down = w1[expert].float().clone(), w2[expert].float().clone()
                for adapter_id, coefficient in zip(
                    adapter_ids[token], coefficients[token]
                ):
                    if adapter_id == 0:
                        continue
                    a = adapter_id - 1
                    for part in range(2):
                        up[part * intermediate : (part + 1) * intermediate].add_(
                            coefficient
                            * (
                                full_b13[part][a, expert].float()
                                @ full_a13[part][a, expert].float()
                            )
                        )
                    down.add_(
                        coefficient
                        * (full_b2[a, expert].float() @ full_a2[a, expert].float())
                    )
                gate, value = (up @ x[token].float()).chunk(2)
                output.add_(
                    expert_weights[token, choice]
                    * (down @ (torch.nn.functional.silu(gate) * value))
                )
            expected.append(output)
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(
            actual.float(), torch.stack(expected), atol=1e-5, rtol=1e-2
        )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("different_slots", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Distributed composition requires CUDA"
)
def test_routed_expert_dispatch_dp2_ep2(different_slots, use_graph, tmp_path):
    """Dispatch preserves adapter identity across uneven, changing DP batches."""
    from transformers import OlmoeConfig

    from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
        parallel_launch_with_config,
    )
    from vllm.config import ModelConfig, VllmConfig

    OlmoeConfig(
        architectures=["OlmoeForCausalLM"],
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            enforce_eager=True,
        )
    )
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.is_moe_model = True
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    parallel_launch_with_config(
        2, _run_routed_expert_ep_rank, config, None, different_slots, use_graph
    )


def _run_routed_expert_ep_rejection(pgi, vllm_config, cpu_group):
    from tests.lora.utils import make_moe_lora_context
    from vllm.lora.layers import LoRAMapping, LoRARouteMapping
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )

    tokens = 3 if pgi.rank == 0 else 5
    ctx, _ = make_moe_lora_context(tokens, 32, 32, 4, 2, routed=True)
    prepare = MoEPrepareAndFinalizeNaiveDPEPModular()
    prepare.set_lora_context(ctx)
    x = torch.ones(tokens, 32, device=pgi.device, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 2]] * tokens, device=pgi.device, dtype=torch.int32)
    weights = torch.full((tokens, 2), 0.5, device=pgi.device)
    for routed_rank in (0, 1, -1):
        mapping = (
            LoRARouteMapping(((1, 2),) * tokens, ((0.7, 0.3),) * tokens, (), False)
            if pgi.rank == routed_rank
            else LoRAMapping((0,) * tokens, (), False)
        )
        ctx.punica_wrapper.update_metadata(mapping, [1, 2], 3, 0)
        # Both ranks must reject, including the one with no routed requests.
        with pytest.raises(NotImplementedError, match="residency and slot identities"):
            prepare.prepare(x, weights, ids, 4, None, False, FusedMoEQuantConfig.make())
        torch.distributed.barrier(group=cpu_group)


def _run_distributed_residency(pgi, config, cpu_group, adapter_path, missing_path):
    from vllm.config import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.model_manager import SupportsLoRAModel
    from vllm.lora.request import LoRARequest
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.model_executor.layers.linear import ReplicatedLinear

    config.lora_config = LoRAConfig(
        max_loras=2,
        max_cpu_loras=2,
        max_lora_rank=8,
        max_loras_per_request=2,
        lora_dtype=torch.bfloat16,
    )
    config.scheduler_config.max_num_seqs = 8
    config.scheduler_config.max_num_batched_tokens = 8
    model = SupportsLoRAModel()
    model.config = config.model_config.hf_config
    model.add_module("proj", ReplicatedLinear(32, 32, bias=False))
    manager = LRUCacheWorkerLoRAManager(config, pgi.device, {})
    manager.create_lora_manager(model, config)
    a, b, c = (LoRARequest(str(i), i, adapter_path) for i in (1, 2, 3))
    manager.add_adapter(a if pgi.rank == 0 else b)
    plan, _, lookup = manager.prepare_distributed_adapters(
        (a,) if pgi.rank == 0 else (b,), cpu_group
    )
    assert [r.lora_int_id for r in plan.requests] == [1, 2]
    assert lookup == ([-1, 0, 1] if pgi.rank == 0 else [-1, 1, 0])
    assert manager.list_adapters() == {1, 2}
    # An empty local batch still needs the peer's complete union.
    plan, _, _ = manager.prepare_distributed_adapters(
        (b, c) if pgi.rank == 0 else (), cpu_group
    )
    assert [r.lora_int_id for r in plan.requests] == [2, 3]
    assert manager.list_adapters() == {2, 3}
    with pytest.raises(ValueError, match="exceeds"):
        manager.prepare_distributed_adapters(
            (a, b) if pgi.rank == 0 else (c,), cpu_group
        )
    conflicting = LoRARequest(b.lora_name, b.lora_int_id, missing_path)
    with pytest.raises(ValueError, match="Conflicting LoRA descriptors"):
        manager.prepare_distributed_adapters(
            (b,) if pgi.rank == 0 else (conflicting,), cpu_group
        )
    # Staging errors must also reach every rank before any model collective.
    mapping = LoRAMapping((2 if pgi.rank == 0 else 999,), (), False)
    with pytest.raises(RuntimeError, match="rank 1"):
        manager.prepare_distributed_adapters((b,), cpu_group, mapping)
    # Rank 0 retains ID 4 in cache; rank 1 must actually read its missing file.
    missing = LoRARequest("missing", 4, missing_path)
    if pgi.rank == 0:
        manager.add_dummy_lora(missing, rank=8)
    with pytest.raises(RuntimeError, match="rank 1"):
        manager.prepare_distributed_adapters((missing,), cpu_group)
    torch.distributed.barrier(group=cpu_group)


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Distributed composition requires CUDA"
)
def test_distributed_lora_residency_and_collective_errors(tmp_path):
    import json

    from safetensors.torch import save_file
    from transformers import OlmoeConfig

    from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
        parallel_launch_with_config,
    )
    from vllm.config import ModelConfig, VllmConfig

    OlmoeConfig(
        architectures=["OlmoeForCausalLM"],
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
    ).save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, enforce_eager=True
        )
    )
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    metadata = dict(peft_type="LORA", r=8, lora_alpha=8, target_modules=["proj"])
    (adapter / "adapter_config.json").write_text(json.dumps(metadata))
    save_file(
        {
            "base_model.model.proj.lora_A.weight": torch.ones(8, 32),
            "base_model.model.proj.lora_B.weight": torch.ones(32, 8),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "adapter_config.json").write_text(json.dumps(metadata))
    parallel_launch_with_config(
        2, _run_distributed_residency, config, None, str(adapter), str(missing)
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("enable_ep", [False, True])
def test_routed_expert_dispatch_rejected_across_ranks(enable_ep):
    """Unsupported routing fails on every rank before dispatch can deadlock."""
    from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
        parallel_launch_with_config,
    )
    from vllm.config import VllmConfig

    config = VllmConfig()
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.enable_expert_parallel = enable_ep
    config.parallel_config.is_moe_model = True
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    parallel_launch_with_config(2, _run_routed_expert_ep_rejection, config, None)


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Marlin requires CUDA")
@pytest.mark.parametrize("routed", [False, True])
@ensure_current_vllm_config()
@torch.inference_mode()
def test_marlin_expert_composition(routed, workspace_init):
    """Quantized bases retain both adapter updates without requantizing them."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from tests.lora.utils import make_moe_lora_context, moe_lora_reference_weights
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import gptq_marlin_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types

    set_random_seed(0)
    tokens, hidden, intermediate, experts, topk = 5, 512, 256, 4, 2
    dtype = torch.bfloat16

    def quantize(rows, columns):
        results = [
            marlin_quantize(
                torch.randn(columns, rows, dtype=dtype, device="cuda") * 0.05,
                scalar_types.uint4b8,
                128,
            )
            for _ in range(experts)
        ]
        ref, packed, scales = zip(*results)
        return (
            torch.stack(ref).transpose(1, 2),
            torch.stack(packed),
            torch.stack(scales),
        )

    w13, packed13, scale13 = quantize(2 * intermediate, hidden)
    w2, packed2, scale2 = quantize(hidden, intermediate)
    config = make_dummy_moe_config(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
    )
    layer = MarlinExperts(config, gptq_marlin_moe_quant_config(scale13, scale2, 4, 128))
    ctx, coefficients = make_moe_lora_context(
        tokens, hidden, intermediate, experts, topk, routed=routed
    )
    layer.set_lora_context(ctx)
    kernel = FusedMoEKernel(MoEPrepareAndFinalizeNoDPEPModular(), layer)
    x = torch.randn(tokens, hidden, dtype=dtype, device="cuda") * 0.05
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(tokens, topk)
        % experts
    )
    weights = torch.tensor([[0.3, 0.7]] * tokens, device="cuda")
    output = kernel.apply(
        x, packed13, packed2, weights, ids, MoEActivation.SILU, experts, None, False
    )
    merged13, merged2 = moe_lora_reference_weights(w13, w2, ctx, coefficients)

    def reference(a, b):
        gate, up = torch.einsum("th,tkih->tki", x.float(), a.float()[ids.long()]).chunk(
            2, -1
        )
        act = torch.nn.functional.silu(gate) * up
        return (
            torch.einsum("tki,tkhi->tkh", act, b.float()[ids.long()])
            * weights[..., None]
        ).sum(1)

    expected, base = reference(merged13, merged2), reference(w13, w2)
    assert (expected - base).norm() / expected.norm() > 0.02
    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output.float(), expected, atol=1e-4, rtol=1e-2)


@pytest.mark.parametrize("no_adapter", [False, True])
@pytest.mark.parametrize("add_inputs", [False, True])
@ensure_current_vllm_config()
def test_single_route_matches_scalar_expert_projections(no_adapter, add_inputs):
    """One route is ordinary LoRA; base-only must also preserve output semantics."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from tests.lora.utils import make_moe_lora_context
    from vllm.lora.layers import LoRAMapping, LoRARouteMapping
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    set_random_seed(0)
    tokens, hidden, intermediate, experts, topk = 5, 32, 32, 4, 2
    device = current_platform.device_type
    ctx, _ = make_moe_lora_context(
        tokens, hidden, intermediate, experts, topk, routed=False
    )
    layer = TritonExperts(
        make_dummy_moe_config(
            num_experts=experts,
            experts_per_token=topk,
            hidden_dim=hidden,
            intermediate_size=intermediate,
        ),
        FusedMoEQuantConfig.make(),
    )
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    activated = torch.randn(tokens * topk, intermediate, device=device, dtype=x.dtype)
    ids = (
        torch.arange(tokens * topk, device=device, dtype=torch.int32).view(tokens, topk)
        % experts
    )
    weights = torch.full((tokens, topk), 0.5, device=device)
    w1 = torch.empty(experts, 2 * intermediate, hidden, device="meta", dtype=x.dtype)
    w2 = torch.empty(experts, hidden, intermediate, device="meta", dtype=x.dtype)
    initial_up = torch.randn(
        tokens, topk, 2 * intermediate, device=device, dtype=x.dtype
    )
    initial_down = torch.randn(tokens, topk, hidden, device=device, dtype=x.dtype)
    results = []
    for routed in (False, True):
        adapter_id = 0 if no_adapter else 1
        mapping = (
            LoRARouteMapping(((adapter_id,),) * tokens, ((1.0,),) * tokens, (), False)
            if routed
            else LoRAMapping((adapter_id,) * tokens, (), False)
        )
        ctx.punica_wrapper.update_metadata(mapping, [1, 2], 3, 0)
        # Delta-only callers initialize their output to zero even on no-LoRA.
        up = initial_up.clone() if add_inputs else torch.zeros_like(initial_up)
        down = initial_down.clone() if add_inputs else torch.zeros_like(initial_down)
        state = layer.apply_w13_lora(
            ctx,
            y=up,
            x=x,
            topk_ids=ids,
            topk_weights=weights,
            expert_map=None,
            w1=w1,
            w2=w2,
            num_tokens=tokens,
            top_k_num=topk,
            add_inputs=add_inputs,
        )
        layer.apply_w2_lora(
            ctx,
            y=down,
            x=activated,
            topk_weights=weights,
            lora_metadata=state,
            num_tokens=tokens,
            w1=w1,
            w2=w2,
            top_k_num=topk,
            add_inputs=add_inputs,
        )
        results.append((up, down))
    for scalar, routed in zip(*results):
        # Routed accumulation rounds the BF16 delta before adding the base.
        torch.testing.assert_close(scalar, routed, atol=1e-3, rtol=1.6e-2)
    if no_adapter:
        for actual, initial in zip(results[1], (initial_up, initial_down)):
            torch.testing.assert_close(
                actual,
                initial if add_inputs else torch.zeros_like(initial),
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("dispatched", [False, True])
@ensure_current_vllm_config()
def test_trtllm_no_lora_fast_path_respects_routes(dispatched):
    """Scalar base-only metadata cannot suppress routed expert contributions."""
    from tests.lora.utils import make_moe_lora_context
    from vllm.lora.layers import LoRAMapping
    from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
        TrtLlmMxfp4LoRAExperts,
    )

    ctx, _ = make_moe_lora_context(2, 32, 32, 4, 2, routed=True)
    if dispatched:
        ctx.local_lora_routes = ctx.punica_wrapper.lora_route_mapping
        ctx.punica_wrapper.update_metadata(LoRAMapping((0, 0), (), False), [1, 2], 3, 0)
    assert not TrtLlmMxfp4LoRAExperts._batch_has_no_lora(ctx)


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def assign_loras_to_tokens(num_tokens: int, num_sequences: int, max_loras: int):
    """Split `num_tokens` into `num_sequences` sequences.
    Each sequence randomly selects 1 LoRA index from [0, max_loras),
    and all tokens in that sequence are assigned this LoRA index.

    Args:
        num_tokens (int): Total number of tokens.
        num_sequences (int): Number of sequences to split the tokens into.
        max_loras (int): Total number of available LoRA modules.

    Returns:
        torch.Tensor: 1D tensor of shape [num_tokens], where each value
                      is the LoRA index assigned to that token.

    """
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences, "num_tokens must be >= num_sequences"

    # Compute token distribution per sequence (distribute remainder evenly)
    tokens_per_seq = num_tokens // num_sequences
    remainder = num_tokens % num_sequences

    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)

    start = 0
    for seq_idx in range(num_sequences):
        # Determine the token range for this sequence
        end = start + tokens_per_seq + (1 if seq_idx < remainder else 0)

        # Randomly select one LoRA ID for this sequence
        lora_id = random.randint(0, max_loras - 1)

        # Assign the same LoRA ID to all tokens in this sequence
        token_lora_mapping[start:end] = lora_id

        start = end

    return token_lora_mapping


def assign_experts_to_tokens(num_tokens: int, num_experts: int, top_k_num: int):
    """For each token, randomly select `top_k_num` distinct experts out of
    `num_experts`, and assign normalized random weights that sum to 1.

    Args:
        num_tokens (int): Total number of tokens.
        num_experts (int): Total number of available experts.
        top_k_num (int): Number of experts to select per token.

    Returns:
        expert_indices (torch.Tensor): shape [num_tokens, top_k_num],
                                       expert index for each token.
        expert_weights (torch.Tensor): shape [num_tokens, top_k_num],
                                       normalized weights (sum = 1 per row).

    """
    assert top_k_num <= num_experts, "top_k_num must be <= num_experts"

    # Randomly select top_k_num distinct experts for each token
    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        # Randomly choose unique expert indices
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected

    # Generate random weights and normalize along dim=1
    expert_weights = torch.rand((num_tokens, top_k_num), dtype=torch.float32)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    return expert_indices, expert_weights


def sample_data(
    num_tokens: int,
    num_sequences: int,
    max_loras: int,
    num_experts: int,
    top_k_num: int,
):
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num
    )
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    active_lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)
    lora_ids = torch.unique(token_lora_mapping, sorted=True)
    active_lora_ids[: lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
    return topk_ids, topk_weights, token_lora_mapping, active_lora_ids


def use_fused_moe_lora_kernel(
    topk_ids,
    topk_weights,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_ids,
    lora_a_stacked,
    lora_b_stacked,
    hidden_states,
    output,
    max_loras,
    num_experts,
    block_size,
    fully_sharded=False,
    offset=0,
    add_inputs=True,
    shrink_split_k=1,
    expert_map=None,
):
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # init output tensors
    sorted_token_ids = torch.empty(
        (max_loras * max_num_tokens_padded,),
        dtype=torch.int32,
    )
    expert_ids = torch.empty((max_loras * max_num_m_blocks,), dtype=torch.int32)
    num_tokens_post_padded = torch.empty((max_loras,), dtype=torch.int32)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)

    # call kernel
    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        adapter_enabled,
        lora_ids,
        expert_map,
    )

    config = {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_WARPS": 4,
        "NUM_STAGES": 3,
        "SPLIT_K": 1,
    }

    mul_routed_weight = False
    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1)

    # num_active_loras is the number of active LoRAs
    # (max_loras + 1 to include no-lora case)
    # Stored as CPU tensor to match the kernel API (torch.compile compatibility)
    num_active_loras = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")

    fused_moe_lora(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        num_active_loras,
        adapter_enabled,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        shrink_split_k,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
        mul_routed_weight,
        fully_sharded=fully_sharded,
        offset=offset,
        add_inputs=add_inputs,
    )


def use_torch(
    hidden_states,
    token_lora_mapping,
    topk_ids,
    lora_a_stacked,
    lora_b_stacked,
    top_k_num,
    num_slices=1,
):
    outputs = []
    for i in range(hidden_states.shape[0]):
        slice_tensors = []
        for slice_id in range(num_slices):
            lora_idx = token_lora_mapping[i]
            expert_ids = topk_ids[i]
            lora_a = lora_a_stacked[slice_id][lora_idx][expert_ids]
            lora_b = lora_b_stacked[slice_id][lora_idx][expert_ids]
            tensors = [
                hidden_states[i] @ lora_a[x].T @ lora_b[x].T for x in range(top_k_num)
            ]
            slice_tensors.append(torch.stack(tensors, dim=0))

        outputs.append(torch.concat(slice_tensors, dim=-1))
    return torch.stack(outputs, dim=0)


DEVICE_TYPE = current_platform.device_type
DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"{DEVICE_TYPE}:{0}"]
SEED = [42]


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6, 12])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 6, 16])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_kernel(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    num_slices,
    dtype,
    device,
    seed,
):
    torch.set_default_device(device)
    set_random_seed(seed)
    # the number of randomly generated sentences.
    num_sequences = 10
    # generate data
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    # init lora weights
    lora_a_stacked = [
        torch.rand(
            (
                max_loras,
                num_experts,
                max_lora_rank,
                K,
            ),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (
                max_loras,
                num_experts,
                N // num_slices,
                max_lora_rank,
            ),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand(
        (
            num_tokens,
            K,
        ),
        dtype=dtype,
    )

    # fused_moe_lora_kernel output
    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output,
        max_loras,
        num_experts,
        block_size,
    )
    # pytorch output
    output2 = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        num_slices,
    )

    torch.testing.assert_close(output, output2, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("max_lora_rank", [128, 256])
@pytest.mark.parametrize("add_inputs", [True, False])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_large_rank_fallback(
    max_lora_rank,
    add_inputs,
    num_slices,
    dtype,
    device,
):
    """Ranks above the one-shot limit fall back to shrink/expand kernels."""
    torch.set_default_device(device)
    set_random_seed(42)

    num_tokens = 8
    num_sequences = 4
    top_k_num = 2
    num_experts = 4
    max_loras = 2
    N = 256
    K = 256
    block_size = 16
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand(
            (max_loras, num_experts, max_lora_rank, K),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (max_loras, num_experts, N // num_slices, max_lora_rank),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    residual = torch.randn((num_tokens, top_k_num, N), dtype=dtype)
    output = residual.clone()
    use_fused_moe_lora_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output,
        max_loras,
        num_experts,
        block_size,
        add_inputs=add_inputs,
    )

    strict_fp16 = dtype == torch.float16 and max_lora_rank == 256
    if strict_fp16:
        # An fp32 oracle is needed to catch an unintended fp16 -> bf16 cast.
        reference_hidden_states = hidden_states.float()
        reference_lora_a = [lora_a.float() for lora_a in lora_a_stacked]
        reference_lora_b = [lora_b.float() for lora_b in lora_b_stacked]
    else:
        reference_hidden_states = hidden_states
        reference_lora_a = lora_a_stacked
        reference_lora_b = lora_b_stacked

    expected = use_torch(
        reference_hidden_states,
        token_lora_mapping,
        topk_ids,
        reference_lora_a,
        reference_lora_b,
        top_k_num,
        num_slices,
    )
    if add_inputs:
        expected += residual.to(expected.dtype)

    if strict_fp16:
        # This separates the fp16 dot path from the old bf16-cast path while
        # allowing normal fp16 accumulation and output-rounding error.
        torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=8e-4)
    else:
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


def use_fused_moe_lora_kernel_naive(
    topk_ids,
    topk_weights,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_ids,
    lora_a_stacked,
    lora_b_stacked,
    hidden_states,
    output,
    max_loras,
    block_size,
    fully_sharded=False,
    offset=0,
    add_inputs=True,
    shrink_split_k=1,
):
    """Test helper for naive_block_assignment path.
    Skips moe_lora_align_block_size and uses flattened topk_ids as expert_ids.
    """
    config = {
        "BLOCK_SIZE_M": block_size,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_WARPS": 4,
        "NUM_STAGES": 3,
        "SPLIT_K": 1,
    }

    mul_routed_weight = False

    # In naive mode:
    # - expert_ids = topk_ids.view(-1), shape: (num_tokens * top_k,)
    # - sorted_token_ids = None
    # - num_tokens_post_padded = None
    expert_ids = topk_ids.reshape(-1)
    sorted_token_ids = None
    num_tokens_post_padded = None

    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)

    # num_active_loras is the number of active LoRAs
    # (max_loras + 1 to include no-lora case)
    # Stored as CPU tensor to match the kernel API (torch.compile compatibility)
    num_active_loras = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")

    fused_moe_lora(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        num_active_loras,
        adapter_enabled,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        shrink_split_k,
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
        mul_routed_weight=mul_routed_weight,
        fully_sharded=fully_sharded,
        offset=offset,
        add_inputs=add_inputs,
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("max_loras", [4, 8])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_kernel_naive_block_assignment(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    num_slices,
    dtype,
    device,
    seed,
):
    """Test the naive_block_assignment path of the fused_moe_lora kernel.
    This path is triggered when batch_size * top_k is much smaller than
    num_experts * max_loras, and skips the moe_lora_align_block_size kernel.
    """
    torch.set_default_device(device)
    set_random_seed(seed)

    # Verify this configuration would trigger naive_block_assignment
    # (num_tokens * top_k * SPARSITY_FACTOR <= num_experts * max_loras)
    SPARSITY_FACTOR = 8
    assert num_tokens * top_k_num * SPARSITY_FACTOR <= num_experts * max_loras, (
        f"Test configuration doesn't meet naive_block_assignment condition: "
        f"{num_tokens} * {top_k_num} * {SPARSITY_FACTOR} > {num_experts} * {max_loras}"
    )

    # the number of randomly generated sentences.
    num_sequences = min(num_tokens, 4)
    # generate data
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    # init lora weights
    lora_a_stacked = [
        torch.rand(
            (
                max_loras,
                num_experts,
                max_lora_rank,
                K,
            ),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (
                max_loras,
                num_experts,
                N // num_slices,
                max_lora_rank,
            ),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand(
        (
            num_tokens,
            K,
        ),
        dtype=dtype,
    )

    # fused_moe_lora_kernel output (naive path)
    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_kernel_naive(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output,
        max_loras,
        block_size,
    )

    # pytorch reference output
    output_ref = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        num_slices,
    )

    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("column_parallel", [True, False])
def test_fused_moe_lora_kernel_fully_sharded(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    seed,
    column_parallel,
):
    set_random_seed(seed)
    # the number of randomly generated sentences.
    num_sequences = 10
    # generate data
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                nprocs,
                f"tcp://{os.getenv('LOCALHOST', 'localhost')}:{get_open_port()}",
                dtype,
                seed,
                N,
                K,
                num_tokens,
                topk_ids,
                topk_weights,
                token_lora_mapping,
                max_lora_rank,
                top_k_num,
                lora_ids,
                max_loras,
                num_experts,
                block_size,
                column_parallel,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(use_fused_moe_lora_kernel_tensor_parallel, nprocs=2)


def use_fused_moe_lora_kernel_tensor_parallel(
    local_rank,
    world_size,
    init_method,
    dtype,
    seed,
    N,
    K,
    num_tokens,
    topk_ids,
    topk_weights,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_ids,
    max_loras,
    num_experts,
    block_size,
    column_parallel,
):
    def _get_shard_slice(shard_size):
        return slice(local_rank * shard_size, (local_rank + 1) * shard_size)

    set_random_seed(seed)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method=init_method,
        backend=current_platform.dist_backend,
    )
    with ensure_current_vllm_config():
        initialize_model_parallel(world_size, 1)
    tp_size = get_tensor_model_parallel_world_size()

    input_dim = K if column_parallel else N
    output_dim = N if column_parallel else K

    # init lora weights
    lora_a = torch.rand(
        (
            max_loras,
            num_experts,
            max_lora_rank,
            input_dim,
        ),
        dtype=dtype,
    )
    lora_b = torch.rand(
        (
            max_loras,
            num_experts,
            output_dim,
            max_lora_rank,
        ),
        dtype=dtype,
    )

    hidden_states = torch.rand(
        (
            num_tokens,
            input_dim,
        ),
        dtype=dtype,
    )

    output = torch.zeros((num_tokens, top_k_num, output_dim), dtype=dtype)
    topk_ids = topk_ids.to(device)
    topk_weights = topk_weights.to(device)
    token_lora_mapping = token_lora_mapping.to(device)
    lora_ids = lora_ids.to(device)

    ref_output = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        [lora_a],
        [lora_b],
        top_k_num,
    )

    if column_parallel:
        # Column parallel (e.g. gate_up_proj): LoRA A is sliced along the rank dim,
        # and Lora B is sliced along the output dim
        lora_a_shard_size = max_lora_rank // tp_size
        lora_a = lora_a[:, :, _get_shard_slice(lora_a_shard_size), :]
        max_lora_rank = lora_a_shard_size
        offset = 0

        lora_b_shard_size = output_dim // tp_size
        lora_b = lora_b[:, :, _get_shard_slice(lora_b_shard_size), :]
        output = output[:, :, _get_shard_slice(lora_b_shard_size)].contiguous()
    else:
        # Row parallel (e.g. down proj): LoRA A is sliced along the input dim,
        # and LoRA B is sliced along the output dim
        lora_a_shard_size = input_dim // tp_size
        lora_a = lora_a[:, :, :, _get_shard_slice(lora_a_shard_size)]
        hidden_states = hidden_states[:, _get_shard_slice(lora_a_shard_size)]

        lora_b_shard_size = output_dim // tp_size
        lora_b = lora_b[:, :, _get_shard_slice(lora_b_shard_size), :]
        offset = lora_b_shard_size * local_rank

    use_fused_moe_lora_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        [lora_a],
        [lora_b],
        hidden_states,
        output,
        max_loras,
        num_experts,
        block_size,
        fully_sharded=True,
        offset=offset,
    )

    if column_parallel:
        output = tensor_model_parallel_all_gather(output)
    else:
        output = tensor_model_parallel_all_reduce(output)

    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


# -- one-shot fast-path coverage --------------------------------------------
# The fused shrink+expand one-shot kernel pads `BLOCK_R` to next_pow2(rank),
# with a floor of 16 (tensor-core minimum). Small ranks (4, 8) exercise the
# rank-dim masking and are not covered by the original tests, which start at
# rank=16. The legacy two-kernel path additionally fails on rank=4 in TMA
# mode because the rank-dim stride (rank * elem_size) is not 16-byte
# aligned; the one-shot fast path takes precedence whenever fully_sharded
# is False so this regression is hidden in normal use, but the test still
# ensures the one-shot logic is correct against the pytorch reference.


@pytest.mark.parametrize("num_tokens", [16, 100])
@pytest.mark.parametrize("top_k_num", [2])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [4, 8])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_kernel_small_rank(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    num_slices,
    dtype,
    device,
    seed,
):
    """One-shot fast path covering rank<16 (padded to BLOCK_R=16 inside kernel)."""
    torch.set_default_device(device)
    set_random_seed(seed)
    num_sequences = max(1, min(num_tokens, 8))
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_stacked = [
        torch.rand(
            (max_loras, num_experts, max_lora_rank, K),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (max_loras, num_experts, N // num_slices, max_lora_rank),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output,
        max_loras,
        num_experts,
        block_size,
    )
    output_ref = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        num_slices,
    )
    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_tokens", [288, 320])
@pytest.mark.parametrize("top_k_num", [2])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("max_loras", [4])
@pytest.mark.parametrize("N", [2048])
@pytest.mark.parametrize("K", [4096])
@pytest.mark.parametrize("max_lora_rank", [8, 16, 32, 64])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_slices", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_kernel_npid_path(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    num_slices,
    dtype,
    device,
    seed,
):
    """Exercise the NPID > 1 branch of the one-shot fast path.

    With these sizes the one-shot wrapper computes NPID_FACTOR > 1 (base CTA count
    < SM count), so each program covers only an outer chunk of N. The
    cross-outer-block write mask is the correctness-critical bit.
    """
    torch.set_default_device(device)
    set_random_seed(seed)
    num_sequences = max(1, min(num_tokens, 4))
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    lora_a_stacked = [
        torch.rand(
            (max_loras, num_experts, max_lora_rank, K),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (max_loras, num_experts, N // num_slices, max_lora_rank),
            dtype=dtype,
        )
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    use_fused_moe_lora_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output,
        max_loras,
        num_experts,
        block_size,
    )
    output_ref = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        num_slices,
    )
    torch.testing.assert_close(output, output_ref, atol=2e-2, rtol=2e-2)


def _record_fast_paths(monkeypatch):
    from vllm.lora.ops.triton_ops import fused_moe_lora_op

    calls = []

    def record(name):
        original = getattr(fused_moe_lora_op, name)

        def wrapper(*args, **kwargs):
            calls.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(fused_moe_lora_op, name, wrapper)

    record("_run_fused_moe_lora_one_shot")
    record("_run_fused_moe_lora_small_batch")
    return calls


BF16, FP16 = torch.bfloat16, torch.float16


@pytest.mark.parametrize(
    "num_tokens,top_k_num,num_experts,max_loras,max_lora_rank,K,N,num_slices,dtype,"
    "shrink_split_k",
    [
        (1, 2, 64, 4, 32, 2048, 1024, 1, BF16, 1),
        (4, 2, 64, 4, 32, 2048, 1024, 2, BF16, 1),
        (16, 2, 8, 4, 32, 2048, 1024, 1, BF16, 1),
        (16, 1, 64, 4, 4, 2048, 1024, 1, BF16, 1),
        (32, 2, 8, 4, 32, 2048, 1024, 2, BF16, 1),
        (64, 2, 8, 4, 32, 2048, 1024, 2, BF16, 1),
        (65, 2, 8, 4, 32, 2048, 1024, 2, BF16, 1),
        (96, 2, 128, 16, 32, 2048, 1024, 1, BF16, 1),
        (96, 2, 8, 4, 32, 2048, 1024, 2, BF16, 1),
        (256, 2, 8, 4, 32, 2048, 1024, 2, BF16, 1),
        (257, 2, 8, 4, 32, 2048, 1024, 2, BF16, 1),
        (300, 1, 256, 16, 4, 2048, 1024, 1, BF16, 1),
        (300, 1, 256, 16, 8, 2048, 1024, 2, FP16, 1),
        (8, 6, 64, 4, 16, 2048, 1024, 2, FP16, 1),
        (4, 8, 128, 4, 64, 2048, 1024, 1, BF16, 1),
        (32, 8, 64, 4, 128, 1024, 1024, 2, FP16, 1),
        (1, 6, 64, 4, 128, 2048, 1024, 1, BF16, 1),
        (16, 8, 128, 4, 32, 6144, 1536, 2, BF16, 1),
        (2, 8, 128, 4, 16, 6144, 1536, 2, FP16, 1),
        (64, 8, 128, 4, 64, 768, 6144, 1, FP16, 1),
        (4, 8, 128, 4, 128, 768, 6144, 1, BF16, 1),
        (4, 2, 64, 4, 32, 2048, 1024, 2, BF16, 4),
        (32, 2, 8, 4, 32, 2048, 1024, 1, BF16, 4),
    ],
    ids=[
        "m1-naive",
        "m4-naive",
        "m16-sorted",
        "m16-topk1-r4-naive",
        "m32-sorted",
        "m64-sorted",
        "m65-sorted",
        "m96-naive",
        "m96-sorted",
        "m256-sorted",
        "m257-sorted-one-shot",
        "m300-topk1-r4-naive-one-shot",
        "m300-topk1-r8-naive-one-shot-fp16",
        "m8-topk6-r16-fp16",
        "m4-topk8-r64-naive",
        "m32-topk8-r128-fp16",
        "m1-topk6-r128-naive",
        "w13-m16-r32",
        "w13-m2-r16-fp16-naive",
        "w2-m64-r64-fp16",
        "w2-m4-r128-naive",
        "splitk4-m4-naive",
        "splitk4-m32-sorted",
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_token_threshold_dispatch(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    max_lora_rank,
    K,
    N,
    num_slices,
    dtype,
    shrink_split_k,
    device,
    monkeypatch,
):
    fast_calls = _record_fast_paths(monkeypatch)
    torch.set_default_device(device)
    set_random_seed(0)
    block_size = 16
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, min(num_tokens, 4), max_loras, num_experts, top_k_num
    )
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (max_loras, num_experts, N // num_slices, max_lora_rank), dtype=dtype
        )
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)
    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype)
    args = (
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output,
        max_loras,
    )
    if num_tokens * top_k_num * 8 <= num_experts * max_loras:
        use_fused_moe_lora_kernel_naive(
            *args, block_size, shrink_split_k=shrink_split_k
        )
    else:
        use_fused_moe_lora_kernel(
            *args, num_experts, block_size, shrink_split_k=shrink_split_k
        )
    assert bool(fast_calls) == (num_tokens > FUSED_MOE_LORA_TWO_KERNEL_MAX_TOKENS), (
        fast_calls
    )
    output_ref = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        num_slices,
    )
    torch.testing.assert_close(output, output_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [4, 64, 300], ids=["m4", "m64", "m300"])
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_expert_parallel(num_tokens, device, monkeypatch):
    fast_calls = _record_fast_paths(monkeypatch)
    torch.set_default_device(device)
    set_random_seed(0)
    top_k_num, num_experts, max_loras, max_lora_rank = 2, 16, 4, 32
    K, N, num_slices, block_size, dtype = 2048, 1024, 2, 16, torch.bfloat16
    topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
        num_tokens, min(num_tokens, 4), max_loras, num_experts, top_k_num
    )
    topk_ids[0] = torch.tensor([1, 3], dtype=torch.int32)
    local_experts = torch.arange(0, num_experts, 2)
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
    expert_map[local_experts] = torch.arange(local_experts.numel(), dtype=torch.int32)
    is_local = expert_map[topk_ids.long()] >= 0
    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (max_loras, num_experts, N // num_slices, max_lora_rank), dtype=dtype
        )
        for _ in range(num_slices)
    ]
    for lora_a, lora_b in zip(lora_a_stacked, lora_b_stacked):
        lora_a[:, expert_map < 0] = 0
        lora_b[:, expert_map < 0] = 0
    hidden_states = torch.rand((num_tokens, K), dtype=dtype)
    residual = torch.randn((num_tokens, top_k_num, N), dtype=dtype)
    output = residual.clone()
    use_fused_moe_lora_kernel(
        topk_ids,
        topk_weights,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        [lora_a[:, local_experts].contiguous() for lora_a in lora_a_stacked],
        [lora_b[:, local_experts].contiguous() for lora_b in lora_b_stacked],
        hidden_states,
        output,
        max_loras,
        num_experts,
        block_size,
        expert_map=expert_map,
    )
    assert bool(fast_calls) == (num_tokens > FUSED_MOE_LORA_TWO_KERNEL_MAX_TOKENS), (
        fast_calls
    )
    expected = residual + use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        num_slices,
    )
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(output[~is_local], residual[~is_local], atol=0, rtol=0)


# -- one-shot corner-case coverage ------------------------------------------
# Each of the following exercises a path where the kernel is launched but
# every program early-exits, leaving the output unchanged. The contract is
# additive (`output += contribution`), so an empty contribution must leave
# the input residual untouched.


def _build_one_shot_inputs(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    max_lora_rank,
    K,
    N,
    num_slices,
    block_size,
    dtype,
):
    """Common scaffolding for the corner-case tests below."""
    num_sequences = max(1, min(num_tokens, 4)) if num_tokens > 0 else 1
    if num_tokens > 0:
        topk_ids, topk_weights, token_lora_mapping, lora_ids = sample_data(
            num_tokens, num_sequences, max_loras, num_experts, top_k_num
        )
    else:
        # M=0 path: caller may still hand us empty tensors with the right shape.
        topk_ids = torch.empty((0, top_k_num), dtype=torch.int32)
        topk_weights = torch.empty((0, top_k_num), dtype=torch.float32)
        token_lora_mapping = torch.empty((0,), dtype=torch.int32)
        lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)

    lora_a_stacked = [
        torch.rand((max_loras, num_experts, max_lora_rank, K), dtype=dtype)
        for _ in range(num_slices)
    ]
    lora_b_stacked = [
        torch.rand(
            (max_loras, num_experts, N // num_slices, max_lora_rank), dtype=dtype
        )
        for _ in range(num_slices)
    ]
    hidden_states = torch.rand((max(num_tokens, 0), K), dtype=dtype)
    return (
        topk_ids,
        topk_weights.to(dtype),
        token_lora_mapping,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
    )


def _call_one_shot(
    output,
    hidden_states,
    lora_a_stacked,
    lora_b_stacked,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_ids,
    num_active_loras,
    adapter_enabled,
    block_size,
    add_inputs=True,
):
    """Direct call into fused_moe_lora with one-shot-routed defaults."""
    from vllm.lora.ops.triton_ops import fused_moe_lora as _op

    _op(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        num_active_loras,
        adapter_enabled,
        block_size,
        32,
        64,
        1,
        4,
        3,
        1,
        block_size,
        32,
        64,
        1,
        4,
        3,
        1,
        False,
        False,
        0,
        add_inputs,
    )


@pytest.mark.parametrize(
    "trigger",
    ["sorted_lora_ids_neg", "naive_mapping_neg", "naive_all_disabled"],
)
@pytest.mark.parametrize("one_shot", [False, True], ids=["two_kernel", "one_shot"])
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_one_shot_early_exit(
    trigger, one_shot, device, monkeypatch
):
    """Both the one-shot and two-kernel paths must leave the residual
    byte-identical when every program must early-exit. Three trigger
    conditions are covered:

      - "sorted_lora_ids_neg":  sorted path, lora_ids all -1 (lora_id<0 check)
      - "naive_mapping_neg":    naive path, token_lora_mapping all -1
      - "naive_all_disabled":   naive path, adapter_enabled all 0
    """
    fast_calls = _record_fast_paths(monkeypatch)
    torch.set_default_device(device)
    set_random_seed(0)

    # Per-trigger shapes: naive_mapping_neg needs the naive dispatch gate
    # `num_tokens*top_k*8 <= num_experts*max_loras` to hold, hence the
    # larger E/max_loras and smaller num_tokens.
    if trigger == "naive_mapping_neg":
        num_tokens, top_k, E, max_loras, R = 8, 2, 64, 8, 16
        if one_shot:
            top_k, E, max_loras = 1, 256, 16
    elif trigger == "naive_all_disabled":
        num_tokens, top_k, E, max_loras, R = 32, 2, 8, 4, 32
    else:  # sorted_lora_ids_neg
        num_tokens, top_k, E, max_loras, R = 32, 2, 8, 4, 16
    if one_shot:
        num_tokens = 300
    K, N = 1024, 1024
    block_size, num_slices, dtype = 16, 2, torch.bfloat16

    (
        topk_ids,
        topk_weights,
        token_lora_mapping,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
    ) = _build_one_shot_inputs(
        num_tokens, top_k, E, max_loras, R, K, N, num_slices, block_size, dtype
    )

    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    num_active_loras = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")

    if trigger == "sorted_lora_ids_neg":
        lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)
        max_pad = topk_ids.numel() + E * (block_size - 1)
        max_pad = round_up(max_pad, block_size)
        max_blocks = CEILDIV(max_pad, block_size)
        sorted_token_ids = torch.zeros((max_loras, max_pad), dtype=torch.int32)
        expert_ids = torch.full((max_loras, max_blocks), -1, dtype=torch.int32)
        num_post = torch.zeros((max_loras,), dtype=torch.int32)
    else:
        sorted_token_ids = None
        expert_ids = topk_ids.reshape(-1).contiguous()
        num_post = None
        if trigger == "naive_mapping_neg":
            token_lora_mapping = torch.full((num_tokens,), -1, dtype=torch.int32)
            lora_ids = torch.full((max_loras + 1,), -1, dtype=torch.int32)
        else:  # naive_all_disabled
            adapter_enabled = torch.zeros(max_loras + 1, dtype=torch.int32)

    residual = torch.randn((num_tokens, top_k, N), dtype=dtype) * 0.1
    output = residual.clone()

    _call_one_shot(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_post,
        token_lora_mapping,
        R,
        top_k,
        lora_ids,
        num_active_loras,
        adapter_enabled,
        block_size,
    )
    assert bool(fast_calls) == one_shot, fast_calls
    torch.testing.assert_close(output, residual, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [8, 300], ids=["two_kernel", "one_shot"])
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_zero_grid_no_crash(num_tokens, device, monkeypatch):
    """num_active_loras=0 (or num_slices=0) would otherwise launch a grid
    with a zero dimension. Both paths must short-circuit before launch."""
    fast_calls = _record_fast_paths(monkeypatch)
    torch.set_default_device(device)
    set_random_seed(0)
    top_k, E, max_loras, R, K, N = 2, 8, 4, 16, 1024, 1024
    block_size, num_slices, dtype = 16, 2, torch.bfloat16

    (
        topk_ids,
        topk_weights,
        token_lora_mapping,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
    ) = _build_one_shot_inputs(
        num_tokens,
        top_k,
        E,
        max_loras,
        R,
        K,
        N,
        num_slices,
        block_size,
        dtype,
    )
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    num_active_loras = torch.tensor([0], dtype=torch.int32, device="cpu")
    residual = torch.randn((num_tokens, top_k, N), dtype=dtype) * 0.1
    output = residual.clone()

    # sorted path is the one that uses num_active_loras for grid axis 2
    max_pad = topk_ids.numel() + E * (block_size - 1)
    max_pad = round_up(max_pad, block_size)
    max_blocks = CEILDIV(max_pad, block_size)
    sorted_token_ids = torch.zeros((max_loras, max_pad), dtype=torch.int32)
    expert_ids = torch.full((max_loras, max_blocks), -1, dtype=torch.int32)
    num_post = torch.zeros((max_loras,), dtype=torch.int32)
    _call_one_shot(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_post,
        token_lora_mapping,
        R,
        top_k,
        lora_ids,
        num_active_loras,
        adapter_enabled,
        block_size,
    )
    assert bool(fast_calls) == (num_tokens > FUSED_MOE_LORA_TWO_KERNEL_MAX_TOKENS), (
        fast_calls
    )
    torch.testing.assert_close(output, residual, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_rejects_bad_block_size_m(device):
    """one-shot must surface a clear assertion when shrink_block_size_m is not
    a power of 2 / less than 16, instead of the cryptic Triton compile
    failure (`arange's range must be a power of 2`)."""
    torch.set_default_device(device)
    set_random_seed(0)
    num_tokens, top_k, E, max_loras, R, K, N = 300, 2, 8, 4, 16, 1024, 1024
    num_slices, dtype = 2, torch.bfloat16
    block_size = 24  # NOT a power of 2

    (
        topk_ids,
        topk_weights,
        token_lora_mapping,
        lora_ids,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
    ) = _build_one_shot_inputs(
        num_tokens,
        top_k,
        E,
        max_loras,
        R,
        K,
        N,
        num_slices,
        16,
        dtype,
    )
    # Build sorted-mode metadata at block_size=16 so shapes are sane,
    # but pass block_size=24 to the op (the buggy combination).
    max_pad = topk_ids.numel() + E * (16 - 1)
    max_pad = round_up(max_pad, 16)
    max_blocks = CEILDIV(max_pad, 16)
    sorted_token_ids = torch.zeros((max_loras, max_pad), dtype=torch.int32)
    expert_ids = torch.full((max_loras, max_blocks), -1, dtype=torch.int32)
    num_post = torch.zeros((max_loras,), dtype=torch.int32)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32)
    num_active_loras = torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu")
    output = torch.zeros((num_tokens, top_k, N), dtype=dtype)

    with pytest.raises(AssertionError, match="shrink_block_size_m"):
        _call_one_shot(
            output,
            hidden_states,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_post,
            token_lora_mapping,
            R,
            top_k,
            lora_ids,
            num_active_loras,
            adapter_enabled,
            block_size,
        )
