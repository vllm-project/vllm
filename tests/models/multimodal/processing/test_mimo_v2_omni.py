# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-V2 omni video timestamps must follow the sampled frames' media time.

Uses the public `XiaomiMiMo/MiMo-V2.5` config and tokenizer (no weights).
"""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.model_executor.models.mimo_v2_omni import _video_timestamps
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MediaWithBytes

pytestmark = pytest.mark.cpu_test

MODEL = "XiaomiMiMo/MiMo-V2.5"


@pytest.fixture(scope="module")
def processor():
    model_config = ModelConfig(
        MODEL,
        tokenizer=MODEL,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _metadata(frames_indices, fps=25.0):
    return {
        "fps": fps,
        "duration": (frames_indices[-1] + 1) / fps,
        "total_num_frames": frames_indices[-1] + 1,
        "frames_indices": frames_indices,
        "video_backend": "opencv",
        "do_sample_frames": False,
    }


def _process(processor, video):
    mm_items = processor.info.parse_mm_data({"video": [video]})
    out = processor("<|vision_start|><|video_pad|><|vision_end|>", mm_items=mm_items)
    data = out["mm_kwargs"]["video"][0].get_data()
    text = processor.info.get_tokenizer().decode(out["prompt_token_ids"])
    return data, text


@pytest.mark.parametrize("wrap_bytes", [False, True])
def test_labels_use_frame_media_time(processor, wrap_bytes):
    # 16 frames sampled 8 s apart from a 120 s clip at 25 fps.
    video = (
        np.zeros((16, 56, 56, 3), dtype=np.uint8),
        _metadata([i * 200 for i in range(16)]),
    )
    if wrap_bytes:  # how the serving path delivers decoded videos
        video = MediaWithBytes(video, b"encoded")

    data, text = _process(processor, video)

    assert float(data["second_per_grid_ts"]) == pytest.approx(16.0)
    assert "00:16" in text and "01:52" in text
    assert "00:14" not in text  # last label when frames were stamped 1 s apart


def test_videos_without_timing_keep_1fps_labels(processor):
    frames = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    no_indices = _metadata([0] * 16)
    del no_indices["frames_indices"]
    for video in (frames, [Image.fromarray(f) for f in frames], (frames, no_indices)):
        data, _ = _process(processor, video)
        assert float(data["second_per_grid_ts"]) == pytest.approx(2.0)


def test_prebuilt_mimo_video_passes_through(processor):
    frames = torch.zeros(16, 3, 56, 56)
    timestamps = 8.0 * torch.arange(16.0)
    data, _ = _process(processor, (frames, timestamps))
    assert float(data["second_per_grid_ts"]) == pytest.approx(16.0)


def test_timestamps_from_metadata():
    indices = [500 + i * 200 for i in range(8)]
    torch.testing.assert_close(
        _video_timestamps(8, _metadata(indices)), 20.0 + 8.0 * torch.arange(8.0)
    )
    indices = np.array([0, 25, 50, 75])
    torch.testing.assert_close(
        _video_timestamps(4, _metadata([0]) | {"frames_indices": indices}),
        torch.arange(4.0),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"fps": 25.0},
        {"frames_indices": [0, 50, 100, 150]},
        {"frames_indices": [0, 50, 100, 150], "fps": 0.0},
        {"frames_indices": [0, 50, 100], "fps": 25.0},
        {"frames_indices": [0, 0, 0, 0], "fps": 25.0},
    ],
)
def test_timestamps_without_frame_timing(metadata):
    torch.testing.assert_close(_video_timestamps(4, metadata), torch.arange(4.0))
