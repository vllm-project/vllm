# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import numpy as np

from vllm.models.dots3_note.common.processor import Dots3NoteMultiModalProcessor
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import MultiModalDataParser


def test_processor_cache_miss_preserves_video_source_bytes():
    parser = MultiModalDataParser()
    processor = object.__new__(Dots3NoteMultiModalProcessor)
    processor.info = Mock()
    processor.info.parse_mm_data.side_effect = lambda data, **_: (
        parser.parse_mm_data(data)
    )

    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    source_bytes = b"encoded video with audio"
    video = MediaWithBytes((frames, {"fps": 1.0}), source_bytes)
    mm_items = parser.parse_mm_data({"video": [video]})

    cache = Mock()
    cache.is_cached.return_value = [False]
    _, missing_items = processor._get_cache_missing_items(
        cache,
        mm_items,
        {"video": ["video-hash"]},
    )

    missing_video = missing_items["video"].data[0]
    assert isinstance(missing_video, MediaWithBytes)
    assert missing_video.original_bytes == source_bytes
    assert missing_items["video"].metadata == [{"fps": 1.0}]
