# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for inferring an encoder's `dynamic_arg_dims` from its `forward`."""

import pytest
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.models.transformers.utils import infer_dynamic_arg_dims


# The stubs mirror the `forward` signatures of the HF vision encoders they are named
# after, so that the test does not depend on the installed `transformers` version.
class UnannotatedEncoder(nn.Module):
    """Like `SiglipVisionModel`: the tensor input is unannotated."""

    def forward(
        self, pixel_values, interpolate_pos_encoding: bool | None = False, **kw
    ):
        pass


class LegacyTensorEncoder(nn.Module):
    """Like `Idefics3VisionTransformer`: unannotated and legacy tensor inputs."""

    def forward(
        self, pixel_values, patch_attention_mask: torch.BoolTensor | None = None, **kw
    ):
        pass


class AnnotatedEncoder(nn.Module):
    """Like `InternVLVisionModel`: annotated and legacy tensor inputs."""

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        **kw,
    ):
        pass


class NonTensorEncoder(nn.Module):
    """Arguments which are annotated as anything but a tensor are left static."""

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_sizes: tuple[int, int],
        num_tiles: int | None = None,
        *args,
        **kw,
    ):
        pass


@pytest.mark.parametrize(
    "encoder_cls,expected",
    [
        (UnannotatedEncoder, {"pixel_values": 0}),
        (LegacyTensorEncoder, {"pixel_values": 0, "patch_attention_mask": 0}),
        (AnnotatedEncoder, {"pixel_values": 0, "bool_masked_pos": 0}),
        (NonTensorEncoder, {"pixel_values": 0}),
    ],
)
def test_infer_dynamic_arg_dims(encoder_cls, expected):
    assert infer_dynamic_arg_dims(encoder_cls) == expected


@pytest.mark.parametrize("encoder_cls", [UnannotatedEncoder, LegacyTensorEncoder])
def test_inferred_dims_are_accepted_by_support_torch_compile(encoder_cls):
    """`dynamic_arg_dims=None` raises `ValueError` for these encoders."""
    with pytest.raises(ValueError, match="No dynamic dimensions found"):
        support_torch_compile(dynamic_arg_dims=None)(type("E", (encoder_cls,), {}))
    support_torch_compile(dynamic_arg_dims=infer_dynamic_arg_dims(encoder_cls))(
        type("E", (encoder_cls,), {})
    )
