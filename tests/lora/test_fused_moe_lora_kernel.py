# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import random

import pytest
import torch

from tests.utils import multi_gpu_test
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
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


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
    fully_sharded=False,
    offset=0,
):
    # Virtual expert approach: combine (lora_id, expert_id) into one index
    num_expert_lora = num_experts * max_loras
    token_lora_expanded = token_lora_mapping.unsqueeze(1)  # (num_tokens, 1)
    has_lora = token_lora_expanded >= 0
    topk_ids_lora = token_lora_expanded * num_experts + topk_ids
    topk_ids_lora = topk_ids_lora.masked_fill(~has_lora, -1)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_lora,
        block_size,
        num_expert_lora,
    )

    adapter_enabled = torch.ones(max_loras, dtype=torch.int32)

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
    )


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
    set_random_seed(seed)
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
    topk_ids, topk_weights, token_lora_mapping = sample_data(
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
    max_loras,
    num_experts,
    block_size,
    column_parallel,
):
    def _get_shard_slice(shard_size):
        return slice(local_rank * shard_size, (local_rank + 1) * shard_size)

    set_random_seed(seed)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method=init_method,
    )
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

    torch.testing.assert_close(output, ref_output, atol=1e-1, rtol=1e-1)
