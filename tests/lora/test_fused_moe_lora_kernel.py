# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.lora.ops.triton_ops import fused_moe_lora
from vllm.platforms import current_platform


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def assign_loras_to_tokens(num_tokens: int, num_sequences: int, max_loras: int):
    """
    Split `num_tokens` into `num_sequences` sequences.
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
    """
    For each token, randomly select `top_k_num` distinct experts out of `num_experts`,
    and assign normalized random weights that sum to 1.

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
    return topk_ids, topk_weights, token_lora_mapping


def use_fused_moe_lora_kernel(
    topk_ids,
    topk_weights,
    token_lora_mapping,
    max_lora_rank,
    top_k_num,
    lora_a_stacked,
    lora_b_stacked,
    hidden_states,
    output,
    max_loras,
    num_experts,
    block_size,
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
    lora_ids = torch.arange(max_loras + 2, dtype=torch.int32)

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
        "BLOCK_SIZE_M": 16,
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

    fused_moe_lora(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        max_lora_rank,
        top_k_num,
        lora_ids,
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
    )

    return output


def use_torch(
    hidden_states,
    token_lora_mapping,
    topk_ids,
    lora_a_stacked,
    lora_b_stacked,
    top_k_num,
):
    outputs = []
    for i in range(hidden_states.shape[0]):
        lora_idx = token_lora_mapping[i]
        expert_ids = topk_ids[i]
        lora_a = lora_a_stacked[0][lora_idx][expert_ids]
        lora_b = lora_b_stacked[0][lora_idx][expert_ids]
        tensors = [
            hidden_states[i] @ lora_a[x].T @ lora_b[x].T for x in range(top_k_num)
        ]
        outputs.append(torch.stack(tensors, dim=0))
    return torch.stack(outputs, dim=0)


DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"cuda:{0}"]
SEED = [42]


@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6, 12])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 6, 16])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32, 64])
@pytest.mark.parametrize("block_size", [16])
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
    dtype,
    device,
    seed,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)
    # the number of randomly generated sentences.
    num_sequences = 10
    # generate data
    topk_ids, topk_weights, token_lora_mapping = sample_data(
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
    ]
    lora_b_stacked = [
        torch.rand(
            (
                max_loras,
                num_experts,
                N,
                max_lora_rank,
            ),
            dtype=dtype,
        )
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
    )

    torch.testing.assert_close(output, output2, atol=1e-1, rtol=1e-1)
