# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.compilation.backends import disable_benchmarks_for_oversized_inputs
from vllm.platforms import current_platform


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_benchmarks_disabled_only_when_an_input_cannot_be_copied():
    device = torch.device("cuda")
    small = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    config = {"benchmark_combo_kernel": True}

    disable_benchmarks_for_oversized_inputs(config, [small, 3])
    assert config == {"benchmark_combo_kernel": True}

    # Leaves less free memory than the tensor itself, like Gemma-4's per-layer
    # embedding table after the model has loaded on a 16 GB GPU.
    free = current_platform.free_device_memory(device)
    large = torch.empty(int(free * 0.6), dtype=torch.uint8, device=device)

    disable_benchmarks_for_oversized_inputs(config, [small, large])
    assert config == {
        "benchmark_combo_kernel": False,
        "triton.autotune_at_compile_time": False,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_benchmarks_disabled_for_cpu_offloaded_weights():
    # A device copy of a CPU-offloaded weight undoes the offload, whatever its
    # size: it is counted as peak activation memory and taken from the KV cache.
    from vllm.model_executor.offloader.uva import UVAOffloader
    from vllm.utils.platform_utils import is_uva_available

    if not is_uva_available():
        pytest.skip("requires UVA")
    linear = torch.nn.Linear(32, 32, device="cuda")
    (linear,) = UVAOffloader(cpu_offload_max_bytes=1 << 30).wrap_modules(
        module for module in [linear]
    )
    assert getattr(linear.weight, "_vllm_is_uva_offloaded", False)
    config = {"benchmark_combo_kernel": True}

    disable_benchmarks_for_oversized_inputs(config, [linear.weight])
    assert config == {
        "benchmark_combo_kernel": False,
        "triton.autotune_at_compile_time": False,
    }
