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


@pytest.mark.parametrize("disable_marlin", [False, True])
def test_mamba_nvfp4_backend_selection(
    tmp_path, dist_init, monkeypatch, disable_marlin
):
    """Mamba changes its projection backends without changing unrelated linears."""
    from transformers import NemotronHConfig

    from vllm.config import (
        CacheConfig,
        KernelConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.model_executor.kernels.linear.nvfp4.b12x import (
        B12xNvFp4W4A16LinearKernel,
    )
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptLinearMethod,
        ModelOptNvFp4Config,
    )
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_default_torch_dtype

    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() not in (120, 121):
        pytest.skip("requires SM120 or SM121")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    if disable_marlin:
        monkeypatch.setenv("VLLM_DISABLED_KERNELS", "MarlinNvFp4LinearKernel")
    else:
        monkeypatch.delenv("VLLM_DISABLED_KERNELS", raising=False)
    NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        layers_block_type=["linear_attention"],
        vocab_size=128,
        max_position_embeddings=1024,
    ).save_pretrained(tmp_path)
    model = ModelConfig(
        model=str(tmp_path),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        max_model_len=1024,
    )
    cache = CacheConfig(mamba_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float16")
    config = VllmConfig(
        model_config=model,
        cache_config=cache,
        scheduler_config=SchedulerConfig(
            max_num_seqs=1, max_model_len=1024, is_encoder_decoder=False
        ),
        kernel_config=KernelConfig(linear_backend="b12x", moe_backend="b12x"),
    )
    quant = ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device("meta"),
    ):
        layer = MambaMixer2(
            hidden_size=4096,
            ssm_state_size=128,
            conv_kernel_size=4,
            intermediate_size=8192,
            use_conv_bias=True,
            use_bias=False,
            n_groups=8,
            model_config=model,
            cache_config=cache,
            quant_config=quant,
            prefix="model.layers.0.mixer",
        )
        attention = MergedColumnParallelLinear(
            input_size=4096,
            output_sizes=[4096],
            params_dtype=torch.bfloat16,
            quant_config=quant,
            prefix="model.layers.1.attention.q_proj",
        )
    expected = B12xNvFp4W4A16LinearKernel if disable_marlin else MarlinNvFp4LinearKernel
    for projection, kernel_type in (
        (layer.in_proj, expected),
        (layer.out_proj, expected),
        (attention, B12xNvFp4W4A16LinearKernel),
    ):
        method = projection.quant_method
        assert isinstance(method, ModelOptLinearMethod)
        assert type(method.kernel) is kernel_type
    assert config.kernel_config.linear_backend == "b12x"
    assert config.kernel_config.moe_backend == "b12x"


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
    with ensure_current_vllm_config():
        mixer = Mixer2RMSNormGated(
            full_hidden_size=hidden_size,
            full_n_groups=n_groups,
        )
    mixer.weight.weight_loader(mixer.weight, weight)  # load

    # create gated-norm without TP to compute reference
    # - utilize mock patching to disable TP when
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
