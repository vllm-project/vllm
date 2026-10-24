# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random

import pytest
import torch

from tests.utils import ensure_current_vllm_config, multi_gpu_test
from vllm import _custom_ops as ops
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from vllm.lora.ops.triton_ops import fused_moe_lora
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed

# The tensor-parallel cases build their distributed environment inside processes
# spawned by torch.multiprocessing.spawn, so this process never has one to tear down.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


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
        config["SPLIT_K"],
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


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU only")
@pytest.mark.parametrize("ep_size", [1, 2])
@pytest.mark.parametrize("num_slices", [1, 2])
@pytest.mark.parametrize("lora_mode", ["base", "disabled", "mixed"])
@pytest.mark.parametrize("execution", ["eager", "graph"])
@torch.inference_mode()
def test_xpu_mxfp4_moe_lora_matches_reference(
    ep_size, num_slices, lora_mode, execution
):
    """LoRA must affect both GEMMs, with one routing weight and EP masking."""
    from dataclasses import replace

    from vllm.config import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper.punica_xpu import PunicaWrapperXPU
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
        mxfp4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.lora_context import (
        MoELoRAContext,
    )
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsMxFp4,
    )

    set_random_seed(42)
    M, K, N, E, topk, rank = 5, 256, 128, 4, 2, 8
    local_e = E // ep_size
    device, dtype = "xpu", torch.bfloat16
    x = torch.randn(M, K, device=device, dtype=dtype) / 4
    w1 = torch.randint(256, (local_e, 2 * N, K // 2), device=device, dtype=torch.uint8)
    w2 = torch.randint(256, (local_e, K, N // 2), device=device, dtype=torch.uint8)
    s1 = torch.full((local_e, 2 * N, K // 32), 124, device=device, dtype=torch.uint8)
    s2 = torch.full((local_e, K, N // 32), 124, device=device, dtype=torch.uint8)
    b1 = torch.randn(local_e, 2 * N, device=device, dtype=dtype) / 4
    b2 = torch.randn(local_e, K, device=device, dtype=dtype) / 4
    ids = torch.tensor(
        [[3, 0], [1, 2], [2, 0], [3, 2], [0, 1]], device=device, dtype=torch.int32
    )
    weights = torch.rand(M, topk, device=device).softmax(-1)
    expert_map = None
    if ep_size > 1:
        expert_map = torch.tensor([0, -1, 1, -1], device=device, dtype=torch.int32)
    config = FusedMoEConfig(
        num_experts=E,
        experts_per_token=topk,
        hidden_dim=K,
        intermediate_size=N,
        num_local_experts=local_e,
        num_logical_experts=E,
        activation=MoEActivation.SWIGLUOAI,
        device=device,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=replace(
            FusedMoEParallelConfig.make_no_parallel(),
            ep_size=ep_size,
            use_ep=ep_size > 1,
        ),
        in_dtype=dtype,
        is_lora_enabled=True,
    )
    experts = XPUExpertsMxFp4(config, mxfp4_w4a16_moe_quant_config(s1, s2, b1, b2))
    punica = PunicaWrapperXPU(
        M, M, device, lora_config=LoRAConfig(max_loras=2, max_lora_rank=rank)
    )
    adapters = [-1, 1, 2, 1, -1] if lora_mode == "mixed" else [-1] * M
    punica.update_metadata(
        LoRAMapping(adapters, adapters, is_prefill=True), [1, 2], 2, 0
    )

    def lora_weights(slices, output_size, input_size):
        a = tuple(
            torch.randn(2, local_e, rank, input_size, device=device, dtype=dtype) / 8
            for _ in range(slices)
        )
        b = tuple(
            torch.randn(2, local_e, output_size, rank, device=device, dtype=dtype) / 8
            for _ in range(slices)
        )
        return a, b

    a13, b13 = lora_weights(num_slices, 2 * N // num_slices, K)
    a2, b2_lora = lora_weights(1, K, N)
    ctx = MoELoRAContext(
        a13,
        b13,
        a2,
        b2_lora,
        adapter_enabled=torch.ones(3, device=device, dtype=torch.int32),
        max_loras=2,
        top_k=topk,
        w13_num_slices=num_slices,
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=local_e,
        punica_wrapper=punica,
        use_tuned_config=False,
    )
    if lora_mode != "base":
        experts.set_lora_context(ctx)
    output = torch.empty_like(x)
    empty = torch.empty(0, device=device, dtype=dtype)

    def run():
        experts.apply(
            output,
            x,
            w1,
            w2,
            weights,
            ids,
            config.activation,
            E,
            expert_map,
            None,
            None,
            empty,
            empty,
            None,
            False,
        )

    run()
    if execution == "graph":
        torch.xpu.synchronize()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            run()
        x.mul_(0.5)
        ids.copy_(ids.roll(1, dims=0))
        weights.copy_(weights.flip(1))
        graph.replay()

    def unpack(w):
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=device,
        )
        return table[torch.stack((w & 15, w >> 4), dim=-1).long()].flatten(-2) / 8

    ref_w1, ref_w2 = unpack(w1), unpack(w2)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for token in range(M):
        for slot in range(topk):
            expert = int(ids[token, slot])
            expert = int(expert_map[expert]) if expert_map is not None else expert
            if expert < 0:
                continue
            h = (ref_w1[expert] @ x[token].float() + b1[expert].float()).to(dtype)
            adapter = adapters[token] - 1
            if adapter >= 0:
                delta = torch.cat(
                    [
                        b[adapter, expert].float()
                        @ (a[adapter, expert].float() @ x[token].float())
                        .to(dtype)
                        .float()
                        for a, b in zip(a13, b13)
                    ]
                )
                if num_slices == 2:
                    delta = delta.reshape(2, N).T.flatten()
                h = (h + delta.to(dtype)).to(dtype)
            gate, up = h[::2].float().clamp(max=7), h[1::2].float().clamp(-7, 7)
            act = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).to(dtype)
            y = (ref_w2[expert] @ act.float() + b2[expert].float()).to(dtype)
            if adapter >= 0:
                delta = (
                    b2_lora[0][adapter, expert].float()
                    @ (a2[0][adapter, expert].float() @ act.float()).to(dtype).float()
                )
                y = (y + delta.to(dtype)).to(dtype)
            expected[token] += weights[token, slot] * y.float()
    torch.testing.assert_close(output, expected.to(dtype), atol=0.06, rtol=0.03)


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
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_large_rank_fallback(
    max_lora_rank,
    add_inputs,
    num_slices,
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
    dtype = torch.bfloat16

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

    expected = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        num_slices,
    )
    if add_inputs:
        expected += residual

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
        config["SPLIT_K"],
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


@pytest.mark.parametrize("num_tokens", [16, 64])
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
    """Exercise the small-batch / NPID > 1 branch of the one-shot fast path.

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
@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_one_shot_early_exit(trigger, device):
    """one-shot must leave the residual byte-identical when every program
    must early-exit. Three trigger conditions are covered:

      - "sorted_lora_ids_neg":  sorted path, lora_ids all -1 (lora_id<0 check)
      - "naive_mapping_neg":    naive path, token_lora_mapping all -1
      - "naive_all_disabled":   naive path, adapter_enabled all 0
    """
    torch.set_default_device(device)
    set_random_seed(0)

    # Per-trigger shapes: naive_mapping_neg needs the naive dispatch gate
    # `num_tokens*top_k*8 <= num_experts*max_loras` to hold, hence the
    # larger E/max_loras and smaller num_tokens.
    if trigger == "naive_mapping_neg":
        num_tokens, top_k, E, max_loras, R = 8, 2, 64, 8, 16
    elif trigger == "naive_all_disabled":
        num_tokens, top_k, E, max_loras, R = 32, 2, 8, 4, 32
    else:  # sorted_lora_ids_neg
        num_tokens, top_k, E, max_loras, R = 32, 2, 8, 4, 16
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
    torch.testing.assert_close(output, residual, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_zero_grid_no_crash(device):
    """num_active_loras=0 (or num_slices=0) would otherwise launch a grid
    with a zero dimension. one-shot wrapper must short-circuit before launch."""
    torch.set_default_device(device)
    set_random_seed(0)
    num_tokens, top_k, E, max_loras, R, K, N = 8, 2, 8, 4, 16, 1024, 1024
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
    torch.testing.assert_close(output, residual, atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_moe_lora_kernel_rejects_bad_block_size_m(device):
    """one-shot must surface a clear assertion when shrink_block_size_m is not
    a power of 2 / less than 16, instead of the cryptic Triton compile
    failure (`arange's range must be a power of 2`)."""
    torch.set_default_device(device)
    set_random_seed(0)
    num_tokens, top_k, E, max_loras, R, K, N = 32, 2, 8, 4, 16, 1024, 1024
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
