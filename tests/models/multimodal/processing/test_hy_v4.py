# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.models.hy_v4.common.mm_preprocess import HYV4VLDataParser
from vllm.transformers_utils.configs.hy_v4 import HYV4VLVisionConfig

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("modality", "metadata"),
    [
        ("image", {"image_grid_thw": torch.tensor([[1, 8, 12]])}),
        (
            "video",
            {
                "video_grid_thw": torch.tensor([[3, 8, 12], [1, 8, 8]]),
                "video_timestamps": torch.tensor([0.25, 1.75, 2.25]),
            },
        ),
    ],
)
def test_hy_v4_accepts_placeholder_metadata_on_consumer(modality, metadata):
    parser = HYV4VLDataParser(
        HYV4VLVisionConfig(temporal_patch_size=2),
        max_model_len=512,
        allow_missing_mm_embeddings=True,
    )
    items = parser.parse_mm_data({modality: metadata})[modality]
    assert parser.placeholder_metadata_fields(modality) == metadata.keys()
    assert items.get_count() == len(metadata[f"{modality}_grid_thw"])
    assert set(items.get(0)) == metadata.keys()
    if modality == "video":
        assert items.get(0)["video_timestamps"].tolist() == [0.25, 1.75]
        assert items.get(1)["video_timestamps"].tolist() == [2.25]


@pytest.mark.parametrize(
    ("grid", "allow_metadata"),
    [
        ([1, 8, 12], False),
        ([1, 0, 12], True),
        ([1, 7, 12], True),
        ([1, 1024, 1024], True),
    ],
)
def test_hy_v4_rejects_invalid_or_disabled_placeholder_metadata(grid, allow_metadata):
    parser = HYV4VLDataParser(
        HYV4VLVisionConfig(spatial_merge_size=2),
        max_model_len=512,
        allow_missing_mm_embeddings=allow_metadata,
    )
    with pytest.raises(ValueError):
        parser.parse_mm_data({"image": {"image_grid_thw": torch.tensor([grid])}})


@pytest.mark.parametrize("timestamps", [[0.25], [0.25, float("nan")]])
def test_hy_v4_rejects_invalid_video_timestamps(timestamps):
    parser = HYV4VLDataParser(
        HYV4VLVisionConfig(temporal_patch_size=2),
        max_model_len=512,
        allow_missing_mm_embeddings=True,
    )
    with pytest.raises(ValueError, match="timestamps"):
        parser.parse_mm_data(
            {
                "video": {
                    "video_grid_thw": torch.tensor([[3, 8, 12]]),
                    "video_timestamps": torch.tensor(timestamps),
                }
            }
        )


@pytest.mark.parametrize("projector_type", [None, "rmsnorm_mlp_adapter"])
def test_hy_v4_projector_outputs_contiguous_embeddings(projector_type):
    from vllm.models.hy_v4.common.vision import HYV4VisionPatchMerger

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config), torch.inference_mode():
        projector = HYV4VisionPatchMerger(
            HYV4VLVisionConfig(
                hidden_size=8,
                out_hidden_size=16,
                projector_type=projector_type,
                use_image_newline_token=False,
            )
        )
        embeddings = projector(torch.ones(1, 4, 6, 8))
    assert embeddings.shape == (6, 16)
    assert embeddings.is_contiguous()
