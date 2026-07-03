# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
]


def _shutdown_llm(llm: Any, gpu_memory_utilization: float) -> None:
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.platforms import current_platform

    try:
        shutdown_timeout = 60.0 if current_platform.is_rocm() else None
        llm.llm_engine.engine_core.shutdown(timeout=shutdown_timeout)
    except Exception:
        pass

    del llm

    try:
        import torch

        torch._dynamo.reset()
    except Exception:
        pass

    cleanup_dist_env_and_memory()

    if current_platform.is_rocm():
        from tests.utils import wait_for_rocm_memory_to_settle

        wait_for_rocm_memory_to_settle(threshold_ratio=1.0 - gpu_memory_utilization)


@contextmanager
def managed_llm(*args: Any, **kwargs: Any) -> Iterator[Any]:
    from vllm import LLM

    llm = LLM(*args, **kwargs)
    gpu_memory_utilization = (
        llm.llm_engine.vllm_config.cache_config.gpu_memory_utilization
    )
    try:
        yield llm
    finally:
        _shutdown_llm(llm, gpu_memory_utilization)


def _make_managed_llm_factory() -> Iterator[Callable[..., Any]]:
    from vllm import LLM

    llms: list[tuple[Any, float]] = []

    def make_llm(*args: Any, **kwargs: Any) -> Any:
        llm = LLM(*args, **kwargs)
        gpu_memory_utilization = (
            llm.llm_engine.vllm_config.cache_config.gpu_memory_utilization
        )
        llms.append((llm, gpu_memory_utilization))
        return llm

    try:
        yield make_llm
    finally:
        while llms:
            llm, gpu_memory_utilization = llms.pop()
            _shutdown_llm(llm, gpu_memory_utilization)


@pytest.fixture
def multimodal_llm_factory() -> Iterator[Callable[..., Any]]:
    yield from _make_managed_llm_factory()
