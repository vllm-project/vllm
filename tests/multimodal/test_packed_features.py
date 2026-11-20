# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from collections.abc import Callable

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.packed_features import PackedMMFeatures
from vllm.multimodal.utils import get_mm_features_in_window

pytestmark = pytest.mark.cpu_test


def _item(**tensors: torch.Tensor) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            key: MultiModalFieldElem(data=tensor, field=MultiModalBatchedField())
            for key, tensor in tensors.items()
        }
    )


def _feature(
    offset: int,
    data: MultiModalKwargsItem | None,
    length: int = 4,
    is_embed: torch.Tensor | None = None,
) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=data,
        modality="audio",
        identifier=f"audio-{offset}",
        mm_position=PlaceholderRange(offset=offset, length=length, is_embed=is_embed),
        mm_hash=f"hash-{offset}" if offset % 20 else None,
    )


def _features() -> list[MultiModalFeatureSpec]:
    """One feature of each kind that is packed differently, sorted by offset."""
    datas = [
        _item(audio=torch.arange(160, dtype=torch.float32)),
        None,
        _item(audio=torch.empty(0)),
        _item(
            ids=torch.arange(5),
            mask=torch.tensor([True, False, True]),
            embeds=torch.arange(12).reshape(4, 3).to(torch.bfloat16),
            scale=torch.tensor(0.5),
            strided=torch.arange(24.0).reshape(4, 6)[:, ::2],
        ),
        MultiModalKwargsItem(
            {
                "pixels": MultiModalFieldElem(
                    data=torch.ones(2, 3, dtype=torch.float16),
                    field=MultiModalSharedField(batch_size=1),
                )
            }
        ),
        # Kept whole: not a tensor.
        MultiModalKwargsItem(
            {
                "nested": MultiModalFieldElem(
                    data=[torch.ones(2), torch.zeros(3)],
                    field=MultiModalBatchedField(),
                )
            }
        ),
        _item(audio=torch.arange(160, 320, dtype=torch.float32)),
    ]
    features = [_feature(10 * i, data, length=1 + i) for i, data in enumerate(datas)]
    # Kept whole: the embedding mask is a tensor.
    features.append(
        _feature(
            10 * len(datas),
            _item(audio=torch.ones(3)),
            is_embed=torch.tensor([True, False, True, True]),
        )
    )
    features.append(_feature(10 * len(datas) + 10, None))
    return features


NUM_FEATURES = len(_features())


@pytest.mark.parametrize("num_packed", range(NUM_FEATURES + 1))
def test_packed_features_equal_originals(num_packed: int):
    features = _features()
    packed = PackedMMFeatures(list(features))

    assert (
        packed.pack_while(lambda i, f: f is features[i] and i < num_packed)
        == num_packed
    )
    assert packed.num_packed == num_packed
    assert len(packed) == NUM_FEATURES
    assert list(packed) == features
    assert [packed[i] for i in range(-NUM_FEATURES, NUM_FEATURES)] == features * 2
    assert packed[1:-1:2] == features[1:-1:2]
    with pytest.raises(IndexError):
        packed[NUM_FEATURES]

    # Feature equality only compares the types of the fields.
    for feature, original in zip(packed, features):
        if feature.data is not None:
            for key, elem in feature.data.items():
                assert elem.field == original.data[key].field
    # Packing copies the tensors, so later writes to the originals don't leak.
    features[0].data["audio"].data.fill_(-1)
    assert (packed[0] == features[0]) == (num_packed == 0)
    # Features kept whole are returned as is.
    assert all(packed[i] is features[i] for i in (5, 7))


@pytest.mark.parametrize("num_packed", [0, 4, NUM_FEATURES])
def test_window_matches_unpacked_list(num_packed: int):
    features = _features()
    packed = PackedMMFeatures(list(features))
    packed.pack_while(lambda i, _: i < num_packed)

    end = features[-1].mm_position.offset + features[-1].mm_position.length
    for start in range(end + 2):
        for stop in range(start, end + 2):
            assert get_mm_features_in_window(
                packed, start, stop
            ) == get_mm_features_in_window(features, start, stop)


_Edit = Callable[[list[MultiModalFeatureSpec] | PackedMMFeatures], None]


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda f: f.__setitem__(2, _feature(25, None)), id="set"),
        pytest.param(lambda f: f.__setitem__(-8, _feature(25, None)), id="set_neg"),
        pytest.param(lambda f: f.__delitem__(1), id="del"),
        pytest.param(lambda f: f.insert(3, _feature(25, None)), id="insert"),
        pytest.param(lambda f: f.insert(-100, _feature(25, None)), id="insert_lo"),
        pytest.param(lambda f: f.append(_feature(500, None)), id="append"),
        pytest.param(
            lambda f: f.__setitem__(slice(1, 3), [_feature(25, None)]), id="set_slice"
        ),
        pytest.param(lambda f: f.__delitem__(slice(None, None, 2)), id="del_slice"),
    ],
)
def test_edits_match_list(edit: _Edit):
    features = _features()
    packed = PackedMMFeatures(list(features))
    packed.pack_while(lambda i, _: i < 5)

    edit(features)
    edit(packed)
    assert list(packed) == features

    packed.pack_while(lambda *_: True)
    assert packed.num_packed == len(features)
    assert list(packed) == features


def test_too_many_layouts_are_kept_whole():
    features = [
        _feature(
            4 * i,
            MultiModalKwargsItem(
                {
                    "audio": MultiModalFieldElem(
                        data=torch.ones(i + 1),
                        field=MultiModalFlatField(slices=[slice(0, i + 1)]),
                    )
                }
            ),
        )
        for i in range(32)
    ]
    packed = PackedMMFeatures(list(features))
    packed.pack_while(lambda *_: True)

    assert list(packed) == features
    assert [packed[i] is features[i] for i in range(32)] == [False] * 16 + [True] * 16


def _num_tracked_after_packing(num_features: int) -> int:
    gc.collect()
    before = len(gc.get_objects())
    packed = PackedMMFeatures(
        [
            _feature(4 * i, _item(audio=torch.full((160,), float(i))))
            for i in range(num_features)
        ]
    )
    packed.pack_while(lambda *_: True)
    gc.collect()
    num_tracked = len(gc.get_objects()) - before
    assert len(packed) == num_features
    return num_tracked


def test_tracked_objects_do_not_grow_with_packed_features():
    _num_tracked_after_packing(1)
    assert _num_tracked_after_packing(1000) == _num_tracked_after_packing(10)
