# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
import textwrap
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


# Runs in a subprocess: without serialization, numba's workqueue threading
# layer aborts the whole process ("Concurrent access has been detected").
_KIMI_FUSED_THREADS_SCRIPT = textwrap.dedent(
    """
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    from PIL import Image

    from vllm.transformers_utils.processors.kimi_k25_vision_fused import (
        KimiK25FusedVisionProcessor,
    )

    processor = KimiK25FusedVisionProcessor(
        {
            "patch_size": 14,
            "merge_kernel_size": 2,
            "temporal_merge_kernel_size": 4,
            "in_patch_limit": 16384,
            "patch_limit_on_one_side": 512,
            "fixed_output_tokens": None,
            "in_patch_limit_each_frame": None,
            "in_patch_limit_video": None,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    )
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8))
    medias = [{"type": "image", "image": image}]

    def run(_):
        return processor.preprocess(medias, return_tensors="pt")["pixel_values"]

    expected = run(None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(64)))
    assert all(torch.equal(result, expected) for result in results)
    """
)


def test_kimi_fused_preprocess_from_concurrent_workers():
    """The fused Kimi processor runs a numba parallel kernel; MM preprocessing
    with several renderer workers must not abort the process."""
    pytest.importorskip("numba")
    result = subprocess.run(
        [sys.executable, "-c", _KIMI_FUSED_THREADS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
