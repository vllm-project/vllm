# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for https://github.com/vllm-project/vllm/issues/48383

`use_audio_in_video=True` can be requested by a caller that doesn't know in
advance whether a given video actually has an audio track (e.g. a generic
client that always sets the flag). When the video has no audio, no audio
items are extracted, but Qwen3-Omni's `_call_hf_processor` used to forward
`use_audio_in_video=True` to the underlying HF processor regardless.

The HF `Qwen3OmniMoeProcessor` assumes `use_audio_in_video=True` implies at
least one audio item per video and raises an opaque `StopIteration` from
`replace_multimodal_special_tokens` otherwise, which vLLM's processing
context then wraps in a confusing `ValueError`.

Without the fix, this test raises `ValueError: Failed to apply
Qwen3OmniMoeProcessor on data=... with kwargs={'use_audio_in_video': True, ...}`.
With the fix, a clear `ValueError` is raised instead, naming the actual
mismatch rather than an opaque wrapped `StopIteration`.
"""

import numpy as np
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....multimodal.utils import random_video
from ...utils import build_model_context

MODELS = [
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
]


@pytest.mark.parametrize("model_id", MODELS)
def test_use_audio_in_video_without_audio_track(model_id: str) -> None:
    """
    A video with no audio track, combined with `use_audio_in_video=True`,
    must raise a clear `ValueError` naming the mismatch, rather than an
    opaque `StopIteration`-derived error from the underlying HF processor.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"audio": 0, "image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config, cache=None)
    video_token_id = processor.info.get_hf_config().video_token_id

    rng = np.random.RandomState(0)
    video = random_video(rng, min_frames=8, max_frames=9, min_wh=64, max_wh=65)

    # No "audio" key at all: this is the "video has no audio track" case.
    mm_data = {"video": [video]}

    with pytest.raises(ValueError, match="doesn't have audio track"):
        processor(
            [video_token_id],
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={"use_audio_in_video": True},
        )
