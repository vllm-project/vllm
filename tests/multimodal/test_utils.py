# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.multimodal.utils import (
    allocate_gpu_mm_processors,
    argsort_mm_positions,
    group_and_batch_mm_items,
)


@pytest.mark.parametrize(
    "case",
    [
        # Basic
        dict(
            mm_processor_device="cuda",
            mm_processor_count=0,
            available_device_count=1,
            engine_device_count=1,
            expected_gpu_allocation=[],
        ),
        dict(
            mm_processor_device="cuda",
            mm_processor_count=1,
            available_device_count=1,
            engine_device_count=1,
            expected_gpu_allocation=["cuda:0"],
        ),
        # Use Engine GPUs
        dict(
            mm_processor_device="cuda",
            mm_processor_count=2,
            available_device_count=1,
            engine_device_count=1,
            expected_gpu_allocation=["cuda:0", "cuda:0"],
        ),
        dict(
            mm_processor_device="cuda",
            mm_processor_count=2,
            available_device_count=1,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:0", "cuda:0"],
        ),
        dict(
            mm_processor_device="cuda",
            mm_processor_count=2,
            available_device_count=2,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:0", "cuda:1"],
        ),
        dict(
            mm_processor_device="cuda",
            mm_processor_count=3,
            available_device_count=2,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:0", "cuda:1", "cuda:0"],
        ),
        # Use excess GPUs
        dict(
            mm_processor_device="cuda",
            mm_processor_count=2,
            available_device_count=3,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:2", "cuda:2"],
        ),
        dict(
            mm_processor_device="cuda",
            mm_processor_count=2,
            available_device_count=4,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:2", "cuda:3"],
        ),
        dict(
            mm_processor_device="cuda",
            mm_processor_count=3,
            available_device_count=4,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:2", "cuda:3", "cuda:2"],
        ),
        # Specific device
        dict(
            mm_processor_device="cuda:0",
            mm_processor_count=2,
            available_device_count=4,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:0", "cuda:0"],
        ),
        dict(
            mm_processor_device="cuda:2",
            mm_processor_count=2,
            available_device_count=4,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:2", "cuda:2"],
        ),
        # Out-of-bounds device
        dict(
            mm_processor_device="cuda:4",
            mm_processor_count=2,
            available_device_count=4,
            engine_device_count=2,
            expected_gpu_allocation=["cuda:4", "cuda:4"],
        ),
    ],
)
def test_allocate_gpu_mm_processors(case):
    mm_processor_device = case["mm_processor_device"]
    mm_processor_count = case["mm_processor_count"]
    available_device_count = case["available_device_count"]
    engine_device_count = case["engine_device_count"]
    expected_gpu_allocation = case["expected_gpu_allocation"]

    gpu_allocation = allocate_gpu_mm_processors(
        mm_processor_device,
        mm_processor_count,
        available_device_count=available_device_count,
        engine_device_count=engine_device_count,
    )

    assert gpu_allocation == expected_gpu_allocation


@pytest.mark.parametrize(
    "case",
    [
        # Single modality
        ## Internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=3, length=2),
                ]
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Internally unsorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=3, length=2),
                    PlaceholderRange(offset=0, length=2),
                ]
            },
            expected_modality_idxs=[
                ("image", 1),
                ("image", 0),
            ],
        ),
        # Two modalities
        ## Internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=7, length=4),
                    PlaceholderRange(offset=11, length=5),
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3),
                ],
            },
            expected_modality_idxs=[
                ("audio", 0),
                ("audio", 1),
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Interleaved, internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=4),
                    PlaceholderRange(offset=8, length=2),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                    PlaceholderRange(offset=11, length=4),
                ],
            },
            expected_modality_idxs=[
                ("image", 0),
                ("audio", 0),
                ("image", 1),
                ("audio", 1),
            ],
        ),
        ## Interleaved, internally unsorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=8, length=2),
                    PlaceholderRange(offset=0, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=11, length=4),
                    PlaceholderRange(offset=5, length=2),
                ],
            },
            expected_modality_idxs=[
                ("image", 1),
                ("audio", 1),
                ("image", 0),
                ("audio", 0),
            ],
        ),
        # Three modalities
        ## Internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=15, length=7),
                    PlaceholderRange(offset=22, length=8),
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=3, length=4),
                    PlaceholderRange(offset=7, length=5),
                    PlaceholderRange(offset=12, length=6),
                ],
            },
            expected_modality_idxs=[
                ("audio", 0),
                ("video", 0),
                ("video", 1),
                ("video", 2),
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Interleaved, internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3),
                    PlaceholderRange(offset=20, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=8, length=5),
                ],
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 1),
                ("audio", 0),
                ("video", 0),
                ("image", 2),
            ],
        ),
        ## Interleaved, internally unsorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=20, length=4),
                    PlaceholderRange(offset=2, length=3),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=8, length=5),
                ],
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 2),
                ("audio", 0),
                ("video", 0),
                ("image", 1),
            ],
        ),
    ],
)
def test_argsort_mm_positions(case):
    mm_positions = case["mm_positions"]
    expected_modality_idxs = case["expected_modality_idxs"]

    modality_idxs = argsort_mm_positions(mm_positions)

    assert modality_idxs == expected_modality_idxs


def test_group_and_batch_mm_items_split_by_fieldset():
    elem = MultiModalFieldElem(
        data=torch.empty(1, dtype=torch.uint8),
        field=MultiModalBatchedField(),
    )
    item1 = MultiModalKwargsItem({"x": elem, "y": elem})
    item2 = MultiModalKwargsItem({"y": elem, "x": elem})
    item3 = MultiModalKwargsItem({"x": elem, "y": elem, "z": elem})
    item4 = MultiModalKwargsItem({"x": elem})
    item5 = MultiModalKwargsItem({"x": elem, "y": elem})

    res = group_and_batch_mm_items([item1, item2, item3, item4, item5])
    assert [num_items for num_items, _ in res] == [2, 1, 1, 1]


def test_group_and_batch_mm_items_split_by_shared_data():
    elem1 = MultiModalFieldElem(
        data=torch.zeros(1, dtype=torch.uint8),
        field=MultiModalSharedField(batch_size=1),
    )
    elem2 = MultiModalFieldElem(
        data=torch.zeros(2, dtype=torch.uint8),
        field=MultiModalSharedField(batch_size=1),
    )
    item1 = MultiModalKwargsItem({"x": elem1})
    item2 = MultiModalKwargsItem({"x": elem1})
    item3 = MultiModalKwargsItem({"x": elem2})
    item4 = MultiModalKwargsItem({"x": elem1})
    item5 = MultiModalKwargsItem({"x": elem2})

    res = group_and_batch_mm_items([item1, item2, item3, item4, item5])
    assert [num_items for num_items, _ in res] == [2, 1, 1, 1]
