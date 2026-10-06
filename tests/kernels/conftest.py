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
