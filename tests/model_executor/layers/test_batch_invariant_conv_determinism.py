# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers import batch_invariant as bi


def test_init_batch_invariance_sets_cudnn_deterministic(monkeypatch):
    """VLLM_BATCH_INVARIANT=1 must force deterministic cuDNN/MIOpen conv
    algorithm selection, not just constrain rounding precision, so that
    models with a convolution in their forward path (e.g. vision
    patch-embed conv3d, diffusion VAE blocks) stay batch-invariant."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(bi, "_batch_invariant_MODE", False)
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_benchmark = torch.backends.cudnn.benchmark
    try:
        bi.init_batch_invariance()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = prev_deterministic
        torch.backends.cudnn.benchmark = prev_benchmark
