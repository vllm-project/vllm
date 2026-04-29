# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import pytest
import torch

from tests.utils import ensure_current_vllm_config, multi_gpu_test
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    Mixer2RMSNormGated,
    _gather_decode_state_indices,
)
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec


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


def test_gather_decode_state_indices_no_spec():
    n, max_blocks = 3, 5
    state_indices = torch.arange(n * max_blocks, dtype=torch.int32).reshape(
        n, max_blocks
    )
    last_computed = torch.tensor([0, 1, 2], dtype=torch.int32)
    last_scheduled = torch.tensor([1, 2, 3], dtype=torch.int32)

    in_slots, out_slots = _gather_decode_state_indices(
        state_indices, last_computed, last_scheduled, num_spec_tokens=0
    )

    assert in_slots.shape == (n,)
    assert out_slots.shape == (n,)
    torch.testing.assert_close(in_slots, torch.tensor([0, 6, 12], dtype=torch.int32))
    torch.testing.assert_close(out_slots, torch.tensor([1, 7, 13], dtype=torch.int32))


def test_gather_decode_state_indices_with_spec_matches_align_layout():
    n, max_blocks_full, num_spec_tokens, block_size = 3, 5, 1, 16
    full_block_table = torch.arange(n * max_blocks_full, dtype=torch.int32).reshape(
        n, max_blocks_full
    )
    last_scheduled = torch.tensor([1, 2, 3], dtype=torch.int32)
    seq_lens = (last_scheduled.to(torch.int64) + 1) * block_size

    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=num_spec_tokens,
    )
    align_indices = mamba_get_block_table_tensor(
        full_block_table, seq_lens, spec, "align"
    )

    in_slots, _ = _gather_decode_state_indices(
        full_block_table,
        block_idx_last_computed_token_d=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_idx_last_scheduled_token_d=last_scheduled,
        num_spec_tokens=num_spec_tokens,
    )

    assert in_slots.shape == align_indices.shape
    torch.testing.assert_close(in_slots.to(align_indices.dtype), align_indices)


def test_gather_decode_state_indices_with_spec():
    n, max_blocks, num_spec_tokens = 3, 5, 1
    state_indices = torch.arange(n * max_blocks, dtype=torch.int32).reshape(
        n, max_blocks
    )
    last_computed = torch.tensor([0, 1, 2], dtype=torch.int32)
    last_scheduled = torch.tensor([1, 2, 3], dtype=torch.int32)

    in_slots, out_slots = _gather_decode_state_indices(
        state_indices,
        last_computed,
        last_scheduled,
        num_spec_tokens=num_spec_tokens,
    )

    assert in_slots.shape == (n, 1 + num_spec_tokens)
    assert out_slots.shape == (n, 1 + num_spec_tokens)
    expected = torch.tensor([[1, 2], [7, 8], [13, 14]], dtype=torch.int32)
    torch.testing.assert_close(in_slots, expected)
    torch.testing.assert_close(out_slots, expected)
