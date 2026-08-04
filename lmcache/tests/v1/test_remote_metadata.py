# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.protocol import (
    RemoteMetadata,
    get_remote_metadata_bytes,
    init_remote_metadata_info,
    pad_shape_to_4d,
    strip_shape_padding,
)


@pytest.mark.parametrize("num_groups", [1, 2, 3])
def test_serialize_and_deserialize(num_groups):
    all_shapes = [
        torch.Size([1, 2, 3, 4]),
        torch.Size([5, 6, 7, 8]),
        torch.Size([9, 10, 11, 12]),
    ]
    all_dtypes = [torch.uint8, torch.float16, torch.float32]

    shapes = all_shapes[:num_groups]
    dtypes = all_dtypes[:num_groups]

    # init remote metadata
    init_remote_metadata_info(num_groups)

    origin_metadata = RemoteMetadata(
        100,
        shapes,
        dtypes,
        MemoryFormat.KV_MLA_FMT,
    )

    meta_bytes = origin_metadata.serialize()
    assert len(meta_bytes) == get_remote_metadata_bytes()
    new_metadata = RemoteMetadata.deserialize(meta_bytes)
    assert origin_metadata.length == new_metadata.length
    assert origin_metadata.shapes == new_metadata.shapes
    assert origin_metadata.dtypes == new_metadata.dtypes
    assert origin_metadata.fmt == new_metadata.fmt


def test_pad_shape_to_4d_already_4d():
    shape = torch.Size([2, 4, 8, 16])
    assert pad_shape_to_4d(shape) == [2, 4, 8, 16]


def test_pad_shape_to_4d_1d():
    shape = torch.Size([42])
    assert pad_shape_to_4d(shape) == [42, 0, 0, 0]


def test_pad_shape_to_4d_3d():
    shape = torch.Size([1, 2, 3])
    assert pad_shape_to_4d(shape) == [1, 2, 3, 0]


def test_pad_shape_too_large():
    with pytest.raises(AssertionError):
        pad_shape_to_4d(torch.Size([1, 2, 3, 4, 5]))


def test_strip_shape_padding_no_zeros():
    assert strip_shape_padding([2, 4, 8, 16]) == torch.Size([2, 4, 8, 16])


def test_strip_shape_padding_trailing_zeros():
    assert strip_shape_padding([42, 0, 0, 0]) == torch.Size([42])


def test_strip_shape_padding_preserves_one_dim():
    assert strip_shape_padding([0, 0, 0, 0]) == torch.Size([0])


# ---- Round-trip for sub-4D shapes in RemoteMetadata ----


@pytest.mark.parametrize(
    "shape",
    [
        torch.Size([128]),  # 1D (e.g. MLA)
        torch.Size([4, 64]),  # 2D
        torch.Size([2, 8, 128]),  # 3D
        torch.Size([2, 4, 8, 128]),  # 4D (existing case)
    ],
)
def test_remote_metadata_roundtrip_sub4d(shape):
    init_remote_metadata_info(1)
    original = RemoteMetadata(
        100,
        [shape],
        [torch.float16],
        MemoryFormat.KV_MLA_FMT,
    )
    restored = RemoteMetadata.deserialize(original.serialize())
    assert restored.shapes[0] == shape
