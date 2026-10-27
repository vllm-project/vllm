# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPD metadata-only parser contract shared by GLM-4V and GLM-5.3-Flash."""

import pytest
import torch

from vllm.model_executor.models.glm4_1v import Glm4vMultiModalDataParser


@pytest.mark.parametrize("allow_missing", [False, True])
def test_glm5_image_metadata_requires_ec_consumer(allow_missing):
    parser = Glm4vMultiModalDataParser(
        spatial_merge_size=2,
        allow_missing_mm_embeddings=allow_missing,
    )
    data = {"image_grid_thw": torch.tensor([[1, 4, 6], [1, 8, 8]])}
    if not allow_missing:
        with pytest.raises(ValueError, match="image_embeds"):
            parser.parse_mm_data({"image": data})
        return

    items = parser.parse_mm_data({"image": data})["image"]
    assert len(items) == 2
    assert items.get(1)["image_grid_thw"].tolist() == [1, 8, 8]
    assert items.get_processor_data() == {}
    assert parser.placeholder_metadata_fields("image") == {"image_grid_thw"}
    assert parser.placeholder_metadata_fields("video") == {
        "video_grid_thw",
        "timestamps",
    }


@pytest.mark.parametrize("allow_missing", [False, True])
def test_glm5_video_metadata_requires_timestamps(allow_missing):
    parser = Glm4vMultiModalDataParser(
        spatial_merge_size=2,
        video_needs_metadata=True,
        allow_missing_mm_embeddings=allow_missing,
    )
    data = {
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
        "timestamps": torch.tensor([[0, 1]]),
    }
    if not allow_missing:
        with pytest.raises(ValueError, match="video_embeds"):
            parser.parse_mm_data({"video": data})
        return

    items = parser.parse_mm_data({"video": data})["video"]
    assert len(items) == 1
    assert items.get(0)["timestamps"].tolist() == [0, 1]
    assert items.get_processor_data() == {}
