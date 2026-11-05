# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from vllm.renderers.base import BaseRenderer
from vllm.renderers.hf import HfRenderer


@dataclass
class _ModelConfig:
    multimodal_config: Any
    renderer_num_workers: int
    supports_multimodal_inputs: bool = False
    hf_config: Any = field(default_factory=SimpleNamespace)


@dataclass
class _VllmConfig:
    model_config: _ModelConfig
    parallel_config: Any = field(
        default_factory=lambda: SimpleNamespace(_api_process_rank=0)
    )


def _peak_concurrent_mm_requests(
    renderer_num_workers: int, mm_processor_cache_gb: float | None
) -> int:
    """Send two MM preprocessing requests at once; return how many ran
    concurrently."""
    multimodal_config = (
        None
        if mm_processor_cache_gb is None
        else SimpleNamespace(mm_processor_cache_gb=mm_processor_cache_gb)
    )
    lock = threading.Lock()
    both_running = threading.Event()
    in_flight = peak = 0

    def fake_process_multimodal(self, *args, **kwargs):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
            if in_flight == 2:
                both_running.set()
        both_running.wait(timeout=0.5)
        with lock:
            in_flight -= 1

    with patch.object(BaseRenderer, "_process_multimodal", fake_process_multimodal):
        config = _VllmConfig(_ModelConfig(multimodal_config, renderer_num_workers))
        renderer = HfRenderer(config, None)

    async def send_two():
        await asyncio.gather(
            renderer._process_multimodal_async(),
            renderer._process_multimodal_async(),
        )

    try:
        asyncio.run(send_two())
    finally:
        renderer.shutdown()
    return peak


@pytest.mark.parametrize(
    ("renderer_num_workers", "mm_processor_cache_gb", "expected_peak"),
    [
        # Text-only model: no processor cache config, keep one worker.
        (2, None, 1),
        # Processor cache on: requests stay ordered with the cache (#38418).
        (2, 4, 1),
        # Cache off: preprocessing follows renderer_num_workers.
        (2, 0, 2),
        (1, 0, 1),
    ],
)
def test_mm_preprocessing_concurrency(
    renderer_num_workers, mm_processor_cache_gb, expected_peak
):
    peak = _peak_concurrent_mm_requests(renderer_num_workers, mm_processor_cache_gb)
    assert peak == expected_peak


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
