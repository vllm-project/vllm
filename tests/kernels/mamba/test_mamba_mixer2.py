# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import pytest
import torch

from tests.utils import multi_gpu_test
from vllm.distributed.parallel_state import (init_distributed_environment,
                                             initialize_model_parallel)
from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated
from vllm.platforms import current_platform
from vllm.utils import update_environment_variables


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize(
    "hidden_size_n_groups",
    [
        (64, 1),
        (64, 2),
        (64, 4),  # hidden_size be divisible by num_gpus
        (100, 5),  # and n_groups must divide hidden_size
    ])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mixer2_gated_norm_multi_gpu(
    batch_size: int,
    seq_len: int,
    hidden_size_n_groups: tuple[int, int],
    dtype: torch.dtype,
    device: str = 'cuda',
):
    hidden_size, n_groups = hidden_size_n_groups
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(fn,
                                    args=(
                                        num_processes,
                                        batch_size,
                                        seq_len,
                                        hidden_size,
                                        n_groups,
                                        dtype,
                                        device,
                                    ),
                                    nprocs=nprocs)

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
    current_platform.seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables({
        'RANK': str(local_rank),
        'LOCAL_RANK': str(local_rank),
        'WORLD_SIZE': str(world_size),
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12345',
    })

    # initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # create random weights an inputs
    weight = torch.rand((hidden_size, ), dtype=dtype, device=device)
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
    with (unittest.mock.patch(
            "vllm.model_executor.layers.mamba.mamba_mixer2."
            "get_tensor_model_parallel_world_size",
            return_value=1),
          unittest.mock.patch(
              "vllm.model_executor.layers.mamba.mamba_mixer2."
              "get_tensor_model_parallel_rank",
              return_value=0)):
        mixer_single_gpu = Mixer2RMSNormGated(
            full_hidden_size=hidden_size,
            full_n_groups=n_groups,
        )
    # assign weight to single-gpu mixer
    mixer_single_gpu.weight.data = weight

    # generate and compare
    N = hidden_size // world_size
    output = mixer(
        hidden_states[..., local_rank * N:(local_rank + 1) * N],
        gate_states[..., local_rank * N:(local_rank + 1) * N],
    )
    ref_output = mixer_single_gpu(hidden_states, gate_states)
    torch.testing.assert_close(output,
                               ref_output[...,
                                          local_rank * N:(local_rank + 1) * N],
                               atol=5e-3,
                               rtol=1e-3)
