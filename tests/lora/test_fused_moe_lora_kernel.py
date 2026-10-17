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


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU native MoE")
@pytest.mark.parametrize("recipe", ["fp8", "block_fp8", "int4"])
@pytest.mark.parametrize("logical_size", [96, 128])
@pytest.mark.parametrize("mapped_experts", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("activation_name", ["SILU", "SWIGLUOAI"])
@pytest.mark.parametrize("hidden", [256, 1024])
def test_xpu_quantized_moe_lora_reference(
    recipe, logical_size, mapped_experts, dtype, packed, activation_name, hidden
):
    """LoRA must follow token routes, padded gate/up halves and router weights."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.config import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper.punica_xpu import PunicaWrapperXPU
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a16_moe_quant_config,
        int4_w4a16_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.experts.lora_context import MoELoRAContext
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsBlockFp8,
        XPUExpertsFp8,
        XPUExpertsWNA16,
    )

    activation = MoEActivation[activation_name]
    interleaved = activation == MoEActivation.SWIGLUOAI
    block_fp8 = recipe == "block_fp8"
    torch.manual_seed(42)
    device = "xpu"
    tokens, experts, padded, rank = 4, 2, 128, 8

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=dtype) * 0.05

    # Keep projection variance comparable across hidden sizes.
    x = rand(tokens, hidden) * (10 * (256 / hidden) ** 0.5)
    w1 = rand(experts, hidden, 2 * padded)
    w2 = rand(experts, padded, hidden)
    w1[:, :, logical_size:padded] = 0
    w1[:, :, padded + logical_size :] = 0
    w2[:, logical_size:, :] = 0
    if interleaved:
        w1 = w1.unflatten(-1, (2, padded)).transpose(-1, -2).flatten(-2).contiguous()
    if recipe == "int4":

        def pack(w):
            q = (w.transpose(-1, -2) * 64).round().clamp(-8, 7).to(torch.int32)
            ref = (q.float() / 64).transpose(-1, -2)
            q = q + 8
            return (q[..., ::2] | (q[..., 1::2] << 4)).to(torch.uint8).contiguous(), ref

        w1, ref_w1 = pack(w1)
        w2, ref_w2 = pack(w2)
        s1 = torch.full(
            (experts, 2 * padded, hidden // 128), 1 / 64, device=device, dtype=dtype
        )
        s2 = torch.full((experts, hidden, 1), 1 / 64, device=device, dtype=dtype)
    else:
        w1, w2 = w1.to(torch.float8_e4m3fn), w2.to(torch.float8_e4m3fn)
        ref_w1, ref_w2 = w1.float(), w2.float()
        s1 = torch.ones(
            (experts, hidden // 128, 2) if block_fp8 else (experts,), device=device
        )
        s2 = torch.ones(
            (experts, 1, hidden // 128) if block_fp8 else (experts,), device=device
        )
    # Nonuniform scales expose expert/block-axis swaps in the native path.
    s1.mul_(torch.linspace(0.5, 1.5, s1.numel(), device=device).view_as(s1))
    s2.mul_(torch.linspace(1.5, 0.5, s2.numel(), device=device).view_as(s2))
    if recipe == "int4":
        ref_w1 *= (s1.float() * 64).transpose(-1, -2).repeat_interleave(128, 1)
        ref_w2 *= (s2.float() * 64).transpose(-1, -2).repeat_interleave(128, 1)
    elif block_fp8:
        ref_w1 *= s1.repeat_interleave(128, 1).repeat_interleave(128, 2)
        ref_w2 *= s2.repeat_interleave(128, 1).repeat_interleave(128, 2)
    else:
        ref_w1 *= s1[:, None, None]
        ref_w2 *= s2[:, None, None]
    config = make_dummy_moe_config(
        num_experts=4 if mapped_experts else experts,
        num_local_experts=experts,
        experts_per_token=2,
        hidden_dim=hidden,
        intermediate_size=padded,
        in_dtype=dtype,
        activation=activation,
    )
    config.is_lora_enabled = True
    if mapped_experts:
        config.moe_parallel_config.ep_size = 2
    quant = fp8_w8a16_moe_quant_config(
        s1, s2, block_shape=[128, 128] if block_fp8 else None
    )
    cls = XPUExpertsBlockFp8 if block_fp8 else XPUExpertsFp8
    if recipe == "int4":
        quant = int4_w4a16_moe_quant_config(s1, s2, block_shape=[0, 128])
        cls = XPUExpertsWNA16
    assert cls.supports_lora()
    impl = cls(config, quant)
    wrapper = PunicaWrapperXPU(
        32, 8, device, lora_config=LoRAConfig(max_loras=2, max_lora_rank=8)
    )
    mapping = (1, 2, 0, 1)
    wrapper.update_metadata(
        LoRAMapping(mapping, mapping, is_prefill=True), [1, 2, None], 3, 128
    )
    a13 = (rand(2, experts, rank, hidden), rand(2, experts, rank, hidden))
    b13 = (rand(2, experts, logical_size, rank), rand(2, experts, logical_size, rank))
    if packed:
        a13 = (a13[0],)
        b13 = (torch.cat(b13, dim=-2),)
    a2, b2 = (rand(2, experts, rank, logical_size),), (rand(2, experts, hidden, rank),)
    ctx = MoELoRAContext(
        a13,
        b13,
        a2,
        b2,
        torch.ones(3, device=device, dtype=torch.int32),
        2,
        2,
        len(a13),
        False,
        0,
        1,
        experts,
        wrapper,
        False,
    )
    # Adapter contributions must exceed the comparison error budget.
    for tensor in (*a13, *b13, *a2, *b2):
        tensor.mul_(3)
    if packed and interleaved:
        b13[0].copy_(
            b13[0].unflatten(-2, (2, logical_size)).transpose(-3, -2).flatten(-3, -2)
        )
    impl.set_lora_context(ctx)
    ids = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1]], device=device)
    expert_map = None
    if mapped_experts:
        expert_map = torch.tensor([1, 0, -1, -1], device=device, dtype=torch.int32)
        ids = torch.tensor([[1, 0], [0, -1], [2, 1], [-1, -1]], device=device)
    weights = torch.tensor(
        [[0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.9, 0.1]], device=device
    )
    shape13, shape2, _ = impl.workspace_shapes(
        tokens, 2 * padded, hidden, 2, config.num_experts, experts, None, activation
    )
    workspace13 = torch.empty(shape13, device=device, dtype=dtype)
    workspace2 = torch.empty(shape2, device=device, dtype=dtype)
    # Modular MoE may alias the output with workspace13. Exercise that case
    # as well as an independent output across the existing adapter matrix.
    output = torch.empty_like(x) if packed else workspace13[: x.numel()].view_as(x)
    impl.apply(
        output,
        x,
        w1,
        w2,
        weights,
        ids,
        activation,
        4 if mapped_experts else experts,
        expert_map,
        None,
        None,
        workspace13,
        workspace2,
        None,
        False,
    )
    graph = None
    if not packed and logical_size == 96:
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            impl.apply(
                output,
                x,
                w1,
                w2,
                weights,
                ids,
                activation,
                4 if mapped_experts else experts,
                expert_map,
                None,
                None,
                workspace13,
                workspace2,
                None,
                False,
            )
        x.mul_(0.75)
        weights.copy_(weights.flip(1))
        ids.copy_(ids.flip(1))
        graph.replay()

    def reference(with_w13=True, with_w2=True):
        result = torch.zeros_like(x, dtype=torch.float32)
        for t in range(tokens):
            for k in range(2):
                e = int(ids[t, k])
                if e < 0:
                    continue
                if expert_map is not None:
                    e = int(expert_map[e])
                if e < 0:
                    continue
                base13 = x[t].float() @ ref_w1[e]
                gate, up = (
                    (base13[::2], base13[1::2]) if interleaved else base13.chunk(2)
                )
                adapter = mapping[t] - 1
                if adapter >= 0 and with_w13:
                    for part, values in enumerate((gate, up)):
                        b = b13[0 if packed else part][adapter, e]
                        if packed:
                            b = b[part::2] if interleaved else b.chunk(2)[part]
                        values[:logical_size] += (
                            x[t].float()
                            @ a13[0 if packed else part][adapter, e].float().T
                            @ b.float().T
                        )
                if interleaved:
                    gate = gate.clamp(max=7)
                    act = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)
                else:
                    act = torch.nn.functional.silu(gate) * up
                y = act @ ref_w2[e]
                if adapter >= 0 and with_w2:
                    y += (
                        act[:logical_size]
                        @ a2[0][adapter, e].float().T
                        @ b2[0][adapter, e].float().T
                    )
                result[t] += weights[t, k] * y
        return result

    expected = reference()
    atol, rtol = (2e-3, 1e-2) if dtype == torch.float16 else (2e-2, 3e-2)
    torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)
    # Guard the oracle itself: neither omitted projection nor all-zero output
    # may fit inside the same error budget as a correct implementation.
    for wrong in (
        reference(False, True),
        reference(True, False),
        torch.zeros_like(expected),
    ):
        assert not torch.allclose(wrong, expected, atol=atol, rtol=rtol)

    # Zero deltas must preserve native routing/accumulation, especially in BF16.
    for b in (*b13, *b2):
        b.zero_()
    args = (
        x,
        w1,
        w2,
        weights,
        ids,
        activation,
        4 if mapped_experts else experts,
        expert_map,
        None,
        None,
        workspace13,
        workspace2,
        None,
        False,
    )
    impl.apply(output, *args)
    impl._lora_context = None
    native = torch.empty_like(output)
    impl.apply(native, *args)
    torch.testing.assert_close(output, native, atol=0, rtol=0)
    if graph is not None:
        graph.replay()
        torch.testing.assert_close(output, native, atol=0, rtol=0)


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU native MoE")
def test_xpu_moe_lora_support_boundaries():
    """The LoRA mixin must not enable unsupported native quantization paths."""
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsBlockFp8,
        XPUExpertsFp8,
        XPUExpertsMxFp4,
        XPUExpertsMxFp8,
        XPUExpertsWNA16,
    )
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
        backend_to_kernel_cls,
    )

    for cls in (XPUExpertsFp8, XPUExpertsBlockFp8, XPUExpertsWNA16):
        assert cls.supports_lora()
    for cls in (
        *backend_to_kernel_cls(UnquantizedMoeBackend.XPU),
        XPUExpertsMxFp4,
        XPUExpertsMxFp8,
    ):
        assert not cls.supports_lora()
