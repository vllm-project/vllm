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


@pytest.mark.parametrize("n_groups", [1, 8])
@pytest.mark.parametrize("gated", [True, False])
def test_batch_invariant_norm_runs_fp32_grouped_kernel_through_dispatch(
    monkeypatch, n_groups, gated
):
    """With VLLM_BATCH_INVARIANT the engine config enables the mixer norm custom
    op, so the module's forward (not just forward_cuda) reaches the fp32 grouped
    Triton kernel for any group count, with or without a gate."""
    import vllm.envs as envs
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    config = VllmConfig(
        model_config=ModelConfig(
            model="AntonV/mamba2-130m-hf",
            hf_overrides={"architectures": ["Mamba2ForCausalLM"]},
            max_model_len=2048,
        )
    )
    assert "+mixer2_gated_rms_norm" in config.compilation_config.custom_ops
    device = torch.device("cuda")
    hidden_size, tokens = 1024, 37
    with (
        set_current_vllm_config(config),
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
        norm = Mixer2RMSNormGated(full_hidden_size=hidden_size, full_n_groups=n_groups)
        norm.weight.data = torch.rand(hidden_size, dtype=torch.bfloat16, device=device)
        x = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
        gate = torch.randn_like(x) if gated else None
        out = norm(x, gate)
    ref = rms_norm_gated(
        x,
        norm.weight.data,
        bias=None,
        z=gate,
        eps=norm.variance_epsilon,
        group_size=hidden_size // n_groups,
        norm_before_gate=False,
    )
    assert out.dtype == x.dtype
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("n_groups", [1, 8])
def test_norm_torch_fallback_without_gate_keeps_fp32_variance(n_groups):
    """forward_native stays reachable (groups split across ranks, or one group
    with TP > 1). Without a gate it must still compute the variance in fp32 and
    only round when casting back; the weight multiply then runs in the input
    dtype, which is why this path is not part of the training-alignment claim."""
    device = torch.device("cuda")
    hidden_size, tokens = 1024, 37
    with (
        ensure_current_vllm_config(),
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
        norm = Mixer2RMSNormGated(full_hidden_size=hidden_size, full_n_groups=n_groups)
    norm.weight.data = torch.rand(hidden_size, dtype=torch.bfloat16, device=device)
    x = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device=device)
    out = norm.forward_native(x, None)
    x32 = x.float().view(tokens, n_groups, -1)
    normed = x32 * torch.rsqrt(
        x32.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon
    )
    expected = norm.weight.data * normed.view(tokens, hidden_size).to(x.dtype)
    assert out.dtype == x.dtype
    assert torch.equal(out, expected)
    # a bf16 variance would not reproduce this
    xb = x.view(tokens, n_groups, -1)
    bf16_variant = norm.weight.data * (
        xb * torch.rsqrt(xb.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    ).view(tokens, hidden_size)
    assert not torch.equal(out, bf16_variant)
