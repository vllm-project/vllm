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
from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed


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

    # Spawned workers need the config during module construction and execution.
    with ensure_current_vllm_config():
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


@pytest.mark.parametrize("tp_rank", range(8))
@pytest.mark.parametrize("scalar_shape", [(), (1,)])
def test_mamba_sharded_loader_replicates_per_tensor_scale(
    tp_rank, scalar_shape, monkeypatch
):
    """Every TP rank receives the full scalar, including scalar-shaped checkpoints."""
    from vllm.model_executor.layers.mamba.mamba_mixer2 import (
        mamba_v2_sharded_weight_loader,
    )
    from vllm.model_executor.parameter import PerTensorScaleParameter

    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 8
    )
    param = PerTensorScaleParameter(data=torch.empty(1), weight_loader=lambda: None)
    loaded = torch.full(scalar_shape, 0.25)
    loader = mamba_v2_sharded_weight_loader([(64, 0, 1), (64, 32, 4)], 8, tp_rank)
    loader(param, loaded)
    torch.testing.assert_close(param.data, torch.tensor([0.25]))
