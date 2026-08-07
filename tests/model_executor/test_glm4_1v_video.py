# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
from types import SimpleNamespace

from vllm.model_executor.models.glm4_1v import Glm4vProcessingInfo


def _make_info():
    info = object.__new__(Glm4vProcessingInfo)
    info.get_video_processor = lambda: SimpleNamespace(
        temporal_patch_size=1,
        max_frames=640,
    )
    return info


def _guard_large_range(monkeypatch):
    original_range = builtins.range

    def guarded_range(*args):
        stop = args[0] if len(args) == 1 else args[1]
        assert stop <= 640
        return original_range(*args)

    monkeypatch.setattr(
        "vllm.model_executor.models.glm4_1v.range",
        guarded_range,
        raising=False,
    )


def test_glm46v_metadata_sampling_stays_bounded(monkeypatch):
    _guard_large_range(monkeypatch)
    info = _make_info()

    result = info._get_video_second_idx_glm46v(
        {
            "fps": 1.0,
            "duration": 100_000.0,
            "total_num_frames": 100_000,
            "do_sample_frames": True,
        },
        total_frames=1,
    )

    assert len(result) <= 640


def test_glmga_metadata_sampling_stays_bounded(monkeypatch):
    _guard_large_range(monkeypatch)
    info = _make_info()

    result = info._get_video_second_idx_glmga(
        {
            "fps": 1.0,
            "duration": 100_000.0,
            "total_num_frames": 100_000,
            "do_sample_frames": True,
        },
        total_frames=1,
    )

    assert len(result) <= 640
