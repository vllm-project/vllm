# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_ubatch_wrapper import _slice_is_padding


def test_slice_is_padding_for_ubatch_context():
    is_padding = torch.tensor([False, False, False, True, True])

    torch.testing.assert_close(
        _slice_is_padding(is_padding, slice(0, 3)),
        torch.tensor([False, False, False]),
    )
    torch.testing.assert_close(
        _slice_is_padding(is_padding, slice(3, 5)),
        torch.tensor([True, True]),
    )
    assert _slice_is_padding(None, slice(0, 3)) is None


def test_prepare_moe_padding_mask_only_when_needed():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.use_moe_padding_mask = True
    runner.is_padding = SimpleNamespace(gpu=torch.ones(8, dtype=torch.bool))

    assert runner._prepare_moe_padding_mask(4, 4, CUDAGraphMode.NONE) is None
    assert runner.is_padding.gpu.all()

    mask = runner._prepare_moe_padding_mask(3, 6, CUDAGraphMode.NONE)
    torch.testing.assert_close(
        mask,
        torch.tensor([False, False, False, True, True, True]),
    )

    # A captured graph must retain the mask pointer even when its dummy input
    # has no padding; a later replay of the same shape may have padded rows.
    mask = runner._prepare_moe_padding_mask(4, 4, CUDAGraphMode.FULL)
    torch.testing.assert_close(mask, torch.zeros(4, dtype=torch.bool))

    runner.use_moe_padding_mask = False
    assert runner._prepare_moe_padding_mask(3, 6, CUDAGraphMode.FULL) is None
