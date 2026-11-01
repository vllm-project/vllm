# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.renderers.base import _mm_executor_workers


def _config(renderer_num_workers: int, mm_processor_cache_gb: float | None):
    multimodal_config = (
        None
        if mm_processor_cache_gb is None
        else SimpleNamespace(mm_processor_cache_gb=mm_processor_cache_gb)
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=multimodal_config,
            renderer_num_workers=renderer_num_workers,
        )
    )


@pytest.mark.parametrize(
    ("renderer_num_workers", "mm_processor_cache_gb", "expected"),
    [
        # Text-only model: nothing to preprocess, keep one worker.
        (8, None, 1),
        # Processor cache on (default 4 GB): keep the cache order (#38418).
        (8, 4, 1),
        (8, 0.5, 1),
        # Cache off: the executor follows renderer_num_workers.
        (8, 0, 8),
        (1, 0, 1),
    ],
)
def test_mm_executor_workers(renderer_num_workers, mm_processor_cache_gb, expected):
    config = _config(renderer_num_workers, mm_processor_cache_gb)
    assert _mm_executor_workers(config) == expected
