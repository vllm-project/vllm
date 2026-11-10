# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for inferring an encoder's `dynamic_arg_dims` from its `forward`."""

import pytest
import torch
from torch import nn

from vllm.model_executor.models.transformers.utils import infer_dynamic_arg_dims


# Stubs mirror HF vision encoder signatures, independent of the installed version
class UnannotatedEncoder(nn.Module):
    """Like `SiglipVisionModel`."""

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False, **kw):
        pass


class LegacyTensorEncoder(nn.Module):
    """Like `Idefics3VisionTransformer`."""

    def forward(
        self, pixel_values, patch_attention_mask: torch.BoolTensor | None = None
    ):
        pass


class NonTensorEncoder(nn.Module):
    """Containers of tensors and non-tensor types are left static."""

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_sizes: tuple[int, int],
        tiles: list[torch.Tensor] | None = None,
        *args,
    ):
        pass


@pytest.mark.parametrize(
    ("encoder_cls", "expected"),
    [
        (UnannotatedEncoder, {"pixel_values": 0}),
        (LegacyTensorEncoder, {"pixel_values": 0, "patch_attention_mask": 0}),
        (NonTensorEncoder, {"pixel_values": 0}),
    ],
)
def test_infer_dynamic_arg_dims(encoder_cls: type[nn.Module], expected: dict):
    assert infer_dynamic_arg_dims(encoder_cls) == expected
