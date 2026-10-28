# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from typing import Literal

from vllm.inputs import MultiModalDataDict, MultiModalUUIDDict

ModalityType = Literal["image", "video", "audio"]
MultiModalDataCounts = Mapping[ModalityType, int]


def assert_mm_uuids(
    mm_uuids: MultiModalUUIDDict | None,
    media_count: int,
    expected_uuids: list[str | None],
    modality: str = "image",
) -> None:
    if len(expected_uuids) > 0:
        assert mm_uuids is not None
        assert modality in mm_uuids

        image_uuids = mm_uuids.get(modality)
        assert image_uuids is not None

        assert isinstance(image_uuids, list) and len(image_uuids) == media_count

        assert image_uuids == expected_uuids
    else:
        assert mm_uuids is None


def assert_mm_data_inputs(
    mm_data: MultiModalDataDict | None,
    data_count: MultiModalDataCounts,
    skipped_media_indices: dict[str, list] | None = None,  # modality -> list[int]
) -> None:
    assert mm_data is not None
    assert set(data_count.keys()) == (set(mm_data.keys()))

    for modality, n in data_count.items():
        modality_data = mm_data.get(modality)
        assert modality_data is not None
        assert isinstance(modality_data, list) and len(modality_data) == n

        if skipped_media_indices is not None:
            skipped_media_indices_for_modality = skipped_media_indices.get(modality)
            assert skipped_media_indices_for_modality is not None
            for i in skipped_media_indices_for_modality:
                assert modality_data[i] is None


def assert_mm_data_is_image_input(
    mm_data: MultiModalDataDict | None,
    image_count: int,
    skipped_image_indices: list | None = None,
) -> None:
    assert mm_data is not None
    assert set(mm_data.keys()) == {"image"}

    image_data = mm_data.get("image")
    assert image_data is not None

    assert isinstance(image_data, list) and len(image_data) == image_count
    if skipped_image_indices is not None:
        for i in skipped_image_indices:
            assert image_data[i] is None
