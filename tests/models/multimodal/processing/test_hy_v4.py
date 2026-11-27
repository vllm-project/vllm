# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.models.hy_v4.common.mm_preprocess import (
    HYV4VLDataParser,
    HYV4VLDummyInputsBuilder,
    HYV4VLMultiModalProcessor,
)
from vllm.models.hy_v4_flash.common.vl_model import HYV4FlashVLProcessingInfo
from vllm.transformers_utils.configs.hy_v4 import HYV4VLVisionConfig

from ...utils import build_model_context

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


@pytest.mark.parametrize("configured", [False, True])
def test_hy_v4_flash_preserves_timing_across_epd_partition(configured):
    intervals = {"second_per_grid_ts": [0.5, 1.25]}
    ctx = build_model_context(
        "tencent/Hy4-flash",
        model_config_kwargs={"max_model_len": 4096},
        limit_mm_per_prompt={"video": 2},
        mm_processor_kwargs=intervals if configured else None,
    )
    mm_config = ctx.model_config.get_multimodal_config()
    mm_config.enable_mm_embeds = True
    mm_config.allow_missing_mm_embeddings = True
    info = HYV4FlashVLProcessingInfo(ctx)
    processor = HYV4VLMultiModalProcessor(info, HYV4VLDummyInputsBuilder(info))
    placeholder = info.get_hf_processor().get_placeholder("video")
    frames = np.zeros((8, 112, 112, 3), dtype=np.uint8)
    video = (
        frames,
        {
            "fps": 4.0,
            "duration": 2.0,
            "total_num_frames": 8,
            "frames_indices": list(range(8)),
            "do_sample_frames": False,
            "video_backend": "opencv",
        },
    )
    kwargs = {} if configured else intervals

    def run(data, count, options):
        return processor(
            placeholder * count,
            info.parse_mm_data({"video": data}),
            {"video": ["same-video"] * count},
            hf_processor_mm_kwargs=options,
        )

    parent = run([video, video], 2, kwargs)
    keys = parent["mm_hashes"]["video"]
    assert keys[0] != keys[1]
    for index in range(2):
        child = run([video], 1, {**kwargs, "_epd_video_partition": [index, 2]})
        assert child["mm_hashes"]["video"] == keys[index : index + 1]
        torch.testing.assert_close(
            child["mm_kwargs"]["video"][0]["video_timestamps"].data,
            parent["mm_kwargs"]["video"][index]["video_timestamps"].data,
        )
    data = parent["mm_kwargs"].get_data()
    metadata = {key: data[key] for key in ("video_grid_thw", "video_timestamps")}
    assert run(metadata, 2, kwargs)["mm_hashes"]["video"] == keys
    with pytest.raises(ValueError, match="must match video count"):
        run([video, video], 2, {"second_per_grid_ts": [0.5]})


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
