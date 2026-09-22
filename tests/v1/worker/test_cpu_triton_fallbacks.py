# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU Triton fallbacks must land on the dispatchers that launch the kernels.

`CPUModelRunner._postprocess_triton` swaps Triton kernels for C++ CPU
implementations when Triton is unavailable (arm64, macOS). Kernels wrapped by
`triton_kernel_dispatcher_with_warmup` are captured on the dispatcher instance
at import time, so rebinding the module-level kernel name never reaches the
launch path and the fallback is silently not installed.
"""

import pytest

import vllm.utils.cpu_triton_utils as cpu_tl
import vllm.v1.sample.rejection_sampler as rejection_sampler
import vllm.v1.spec_decode.utils as spec_decode_utils
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

# (module, dispatcher attribute, expected CPU fallback in cpu_triton_utils)
DISPATCHER_FALLBACKS = [
    (
        spec_decode_utils,
        "_eagle_prepare_inputs_padded",
        "eagle_prepare_inputs_padded_kernel",
    ),
    (
        spec_decode_utils,
        "_eagle_prepare_next_token_padded",
        "eagle_prepare_next_token_padded_kernel",
    ),
    (
        spec_decode_utils,
        "_copy_and_expand_eagle_inputs",
        "copy_and_expand_eagle_inputs_kernel",
    ),
    (
        spec_decode_utils,
        "_copy_and_expand_dflash_inputs",
        "copy_and_expand_dflash_inputs_kernel",
    ),
    (
        spec_decode_utils,
        "_eagle_step_slot_mapping_metadata",
        "eagle_step_slot_mapping_metadata_kernel",
    ),
    (rejection_sampler, "_rejection_greedy_sample", "rejection_greedy_sample_kernel"),
    (rejection_sampler, "_rejection_random_sample", "rejection_random_sample_kernel"),
    (rejection_sampler, "_expand", "expand_kernel"),
    (rejection_sampler, "_sample_recovered_tokens", "sample_recovered_tokens_kernel"),
]


@pytest.fixture
def restore_patched_kernels(monkeypatch):
    """Undo the global kernel swaps `_postprocess_triton` performs."""
    import vllm.v1.worker.block_table as block_table
    import vllm.v1.worker.mamba_utils as mamba_utils

    for module, dispatcher_name, _ in DISPATCHER_FALLBACKS:
        dispatcher = getattr(module, dispatcher_name)
        monkeypatch.setattr(dispatcher, "kernel", dispatcher.kernel)
    monkeypatch.setattr(
        block_table._COMPUTE_SLOT_MAPPING_KERNEL,
        "kernel",
        block_table._COMPUTE_SLOT_MAPPING_KERNEL.kernel,
    )
    monkeypatch.setattr(
        mamba_utils, "batch_memcpy_kernel", mamba_utils.batch_memcpy_kernel
    )


@pytest.mark.parametrize(
    "module, dispatcher_name, fallback_name", DISPATCHER_FALLBACKS
)
def test_postprocess_triton_installs_cpu_fallback(
    monkeypatch, restore_patched_kernels, module, dispatcher_name, fallback_name
):
    """Each dispatcher launches the CPU fallback once Triton is unavailable."""
    monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", False)

    # `_postprocess_triton` reads no instance state, so it can run unbound.
    CPUModelRunner._postprocess_triton(None)

    dispatcher = getattr(module, dispatcher_name)
    assert dispatcher.kernel is getattr(cpu_tl, fallback_name)
