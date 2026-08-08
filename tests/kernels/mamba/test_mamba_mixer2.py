# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import unittest.mock

import pytest
import torch

from tests.utils import ensure_current_vllm_config, multi_gpu_test
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MambaMixer2,
    Mixer2RMSNormGated,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

_mixer = object.__new__(MambaMixer2)
save_block_aligned_ssm_states = _mixer._save_block_aligned_ssm_states


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "hidden_size_n_groups",
    [
        (64, 1),
        (64, 2),
        (64, 4),  # hidden_size be divisible by num_gpus
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mixer2_gated_norm_multi_gpu(
    batch_size: int,
    seq_len: int,
    hidden_size_n_groups: tuple[int, int],
    dtype: torch.dtype,
    device: str = "cuda",
):
    hidden_size, n_groups = hidden_size_n_groups
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                batch_size,
                seq_len,
                hidden_size,
                n_groups,
                dtype,
                device,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(mixer2_gated_norm_tensor_parallel, 2)


def mixer2_gated_norm_tensor_parallel(
    local_rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    n_groups: int,
    dtype: torch.dtype,
    device: str,
):
    set_random_seed(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12345",
        }
    )

    # initialize distributed
    init_distributed_environment()
    with ensure_current_vllm_config():
        initialize_model_parallel(tensor_model_parallel_size=world_size)

    # create random weights an inputs
    weight = torch.rand((hidden_size,), dtype=dtype, device=device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    gate_states = torch.randn(batch_size, seq_len, hidden_size)

    # create gated-norm with TP
    mixer = Mixer2RMSNormGated(
        full_hidden_size=hidden_size,
        full_n_groups=n_groups,
    )
    mixer.weight.weight_loader(mixer.weight, weight)  # load

    # create gated-norm without TP to compute reference
    # - utilize mock patching to disable TP when
    with (
        unittest.mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        unittest.mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_rank",
            return_value=0,
        ),
    ):
        mixer_single_gpu = Mixer2RMSNormGated(
            full_hidden_size=hidden_size,
            full_n_groups=n_groups,
        )
    # assign weight to single-gpu mixer
    mixer_single_gpu.weight.data = weight

    # generate and compare
    N = hidden_size // world_size
    output = mixer(
        hidden_states[..., local_rank * N : (local_rank + 1) * N],
        gate_states[..., local_rank * N : (local_rank + 1) * N],
    )
    ref_output = mixer_single_gpu(hidden_states, gate_states)
    torch.testing.assert_close(
        output,
        ref_output[..., local_rank * N : (local_rank + 1) * N],
        atol=5e-3,
        rtol=1e-3,
    )


def _reference_save_block_aligned_ssm_states(
    ssm_state: torch.Tensor,
    varlen_states: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    block_idx_first_scheduled_token_p: torch.Tensor,
    block_idx_last_scheduled_token_p: torch.Tensor,
    last_chunk_indices_p: torch.Tensor,
    num_computed_tokens_p: torch.Tensor,
    mamba_block_size: int,
    chunk_size: int,
) -> None:
    """Original per-sequence loop, kept verbatim as the reference."""
    chunk_stride = mamba_block_size // chunk_size
    num_prefills = state_indices_tensor_p.size(0)

    for seq_idx in range(num_prefills):
        # Block index for the first scheduled token
        block_idx_first_scheduled_token = block_idx_first_scheduled_token_p[seq_idx]

        # Block index for the last scheduled token
        block_idx_last_scheduled_token = block_idx_last_scheduled_token_p[seq_idx]

        # Number of blocks that need to be written
        n_blocks_to_fill = (
            block_idx_last_scheduled_token - block_idx_first_scheduled_token
        )

        # Skip sequences that don't have any blocks to fill
        if n_blocks_to_fill == 0:
            continue

        # Look up the state indices
        cache_blocks_to_fill = state_indices_tensor_p[
            seq_idx,
            block_idx_first_scheduled_token:block_idx_last_scheduled_token,
        ]

        # First chunk index for this sequence
        if seq_idx == 0:  # noqa: SIM108 - kept verbatim
            first_chunk = 0
        else:
            first_chunk = 1 + last_chunk_indices_p[seq_idx - 1]

        # First chunk that is aligned on the mamba block boundary
        first_aligned_chunk = first_chunk + chunk_stride - 1

        # Calculate the number of computed tokens that were not
        # already cached
        num_unaligned_computed_tokens = num_computed_tokens_p[seq_idx] % (
            mamba_block_size
        )

        if num_unaligned_computed_tokens > 0:
            # If the number of computed tokens is not block aligned,
            # then we need to shift the index accordingly
            first_aligned_chunk -= num_unaligned_computed_tokens // chunk_size

        # Get states to write
        from_where = varlen_states[
            first_aligned_chunk : first_aligned_chunk
            + n_blocks_to_fill * chunk_stride : chunk_stride
        ]

        # Write the states
        ssm_state[cache_blocks_to_fill] = from_where


def _make_prefill_workload(
    rng: random.Random,
    num_prefills: int,
    mamba_block_size: int,
    chunk_size: int,
    aligned: bool | None = None,
    max_new_tokens: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """Build an ssm_state cache plus internally consistent prefill metadata,
    mirroring how the mamba attention metadata builder derives the chunk
    layout and the prefix-caching block indices."""
    num_computed = []
    new_tokens = []
    for _ in range(num_prefills):
        if aligned is True:
            computed = rng.randint(0, 3) * mamba_block_size
        elif aligned is False:
            computed = rng.randint(0, 3) * mamba_block_size + rng.randint(
                1, mamba_block_size - 1
            )
        else:
            computed = rng.randint(0, 3 * mamba_block_size)
        num_computed.append(computed)
        new_tokens.append(rng.randint(1, max_new_tokens or 4 * mamba_block_size))

    last_chunk_indices = []
    total_chunks = 0
    for computed, scheduled in zip(num_computed, new_tokens):
        remaining = scheduled
        n_chunks = 0
        # if computed tokens are not chunk-aligned, the first chunk
        # finishes off the partial chunk
        if computed % chunk_size != 0:
            realign_len = min(
                cdiv(computed, chunk_size) * chunk_size - computed, remaining
            )
            remaining -= realign_len
            n_chunks += 1
        n_chunks += cdiv(remaining, chunk_size)
        total_chunks += n_chunks
        last_chunk_indices.append(total_chunks - 1)

    block_idx_first_scheduled_token = [
        cdiv(computed + 1, mamba_block_size) - 1 for computed in num_computed
    ]
    block_idx_last_scheduled_token = [
        max(cdiv(computed + scheduled, mamba_block_size) - 1, 0)
        for computed, scheduled in zip(num_computed, new_tokens)
    ]

    # Distinct cache block per (sequence, block index) pair, as guaranteed
    # by the block manager
    max_blocks_per_seq = max(block_idx_last_scheduled_token) + 1
    num_blocks = num_prefills * max_blocks_per_seq
    block_ids = list(range(num_blocks))
    rng.shuffle(block_ids)
    state_indices_tensor = torch.tensor(block_ids, dtype=torch.int32).view(
        num_prefills, max_blocks_per_seq
    )

    nheads, headdim, dstate = 2, 4, 8
    ssm_state = torch.randn(num_blocks, nheads, headdim, dstate)
    workload = dict(
        varlen_states=torch.randn(total_chunks, nheads, headdim, dstate),
        state_indices_tensor_p=state_indices_tensor,
        block_idx_first_scheduled_token_p=torch.tensor(
            block_idx_first_scheduled_token, dtype=torch.int32
        ),
        block_idx_last_scheduled_token_p=torch.tensor(
            block_idx_last_scheduled_token, dtype=torch.int32
        ),
        last_chunk_indices_p=torch.tensor(last_chunk_indices, dtype=torch.int32),
        num_computed_tokens_p=torch.tensor(num_computed, dtype=torch.int32),
        mamba_block_size=mamba_block_size,
        chunk_size=chunk_size,
    )
    return ssm_state, workload


def _assert_matches_reference(ssm_state: torch.Tensor, workload: dict) -> None:
    ssm_state_ref = ssm_state.clone()
    ssm_state_out = ssm_state.clone()
    _reference_save_block_aligned_ssm_states(ssm_state_ref, **workload)
    save_block_aligned_ssm_states(ssm_state_out, **workload)
    assert torch.equal(ssm_state_out, ssm_state_ref)


@pytest.mark.parametrize(
    "mamba_block_size,chunk_size",
    [(64, 64), (64, 32), (64, 16), (512, 256)],  # chunk_stride 1, 2, 4, 2
)
@pytest.mark.parametrize("num_prefills", [1, 2, 8, 32])
@pytest.mark.parametrize("aligned", [None, True, False])
def test_save_block_aligned_random_workloads(
    mamba_block_size: int, chunk_size: int, num_prefills: int, aligned: bool | None
):
    """Randomized batches; alignment mode steers the chunk-shift branch."""
    rng = random.Random(mamba_block_size * 1000 + num_prefills)
    for _ in range(5):
        ssm_state, workload = _make_prefill_workload(
            rng, num_prefills, mamba_block_size, chunk_size, aligned=aligned
        )
        _assert_matches_reference(ssm_state, workload)


def test_save_block_aligned_zero_fill_mix():
    """Short sequences interleave with filling ones, including trailing
    sequences whose first_aligned_chunk lands past the end of varlen_states
    but is dropped by their zero fill count."""
    rng = random.Random(0)
    for _ in range(10):
        ssm_state, workload = _make_prefill_workload(
            rng, 16, 512, 256, max_new_tokens=512
        )
        _assert_matches_reference(ssm_state, workload)


def test_save_block_aligned_nothing_to_fill():
    """Block-aligned sequences shorter than a mamba block leave the cache
    untouched (the batched path early-outs)."""
    rng = random.Random(0)
    ssm_state, workload = _make_prefill_workload(
        rng, 8, 512, 256, aligned=True, max_new_tokens=512
    )
    ssm_state_before = ssm_state.clone()
    save_block_aligned_ssm_states(ssm_state, **workload)
    assert torch.equal(ssm_state, ssm_state_before)


def test_save_block_aligned_worked_example():
    """Hand-computed batch (mamba_block_size=512, chunk_size=256):
    seq A fresh with 1300 tokens (chunks 0-5) fills block columns 0-1,
    seq B resumed mid-block at 768 with 512 tokens (chunks 6-7) fills
    column 1 via the unaligned shift, and seq C fresh with 200 tokens
    (chunk 8) fills nothing. The whole batch collapses to
    ssm_state[[17, 4, 11]] = varlen_states[[1, 3, 6]]."""
    ssm_state = torch.randn(24, 2, 4, 8)
    varlen_states = torch.randn(9, 2, 4, 8)
    workload = dict(
        varlen_states=varlen_states,
        state_indices_tensor_p=torch.tensor(
            [[17, 4, 9], [3, 11, 22], [7, 19, 20]], dtype=torch.int32
        ),
        block_idx_first_scheduled_token_p=torch.tensor([0, 1, 0], dtype=torch.int32),
        block_idx_last_scheduled_token_p=torch.tensor([2, 2, 0], dtype=torch.int32),
        last_chunk_indices_p=torch.tensor([5, 7, 8], dtype=torch.int32),
        num_computed_tokens_p=torch.tensor([0, 768, 0], dtype=torch.int32),
        mamba_block_size=512,
        chunk_size=256,
    )
    expected = ssm_state.clone()
    expected[[17, 4, 11]] = varlen_states[[1, 3, 6]]
    save_block_aligned_ssm_states(ssm_state, **workload)
    assert torch.equal(ssm_state, expected)
