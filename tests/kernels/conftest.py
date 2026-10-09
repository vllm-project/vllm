# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch


@pytest.fixture(autouse=True)
def reset_default_torch_device():
    """Several kernel tests call torch.set_default_device without restoring
    it, which poisons subsequent tests in the same pytest run (e.g. CPU
    tensors silently created on CUDA). Restore the factory default after
    every test.
    """
    yield
    torch.set_default_device(None)


@pytest.fixture
def batch_invariant_kernel() -> None:
    """Require the batch-invariant kernels in the stable CUDA extension."""
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        require_batch_invariant_quant_kernel,
    )

    require_batch_invariant_quant_kernel()
    assert hasattr(torch.ops._C, "deterministic_top_k_per_row_prefill")
