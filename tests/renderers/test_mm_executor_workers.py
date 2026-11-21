# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many multimodal requests the renderer preprocesses at the same time."""

import asyncio
import contextlib
import threading

import pytest
from PIL import Image

from vllm.assets.image import ImageAsset
from vllm.config import ModelConfig, VllmConfig
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers.registry import cached_tokenizer_from_config

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class _GatedImage(Image.Image):
    """An image that calls `gate` the first time its pixels are read, which
    happens on the thread that preprocesses the request."""

    def __init__(self, image: Image.Image, gate) -> None:
        super().__init__()
        self.__dict__.update(image.copy().__dict__)
        self._gate = gate
        self._gated = False

    def load(self):
        if not self._gated:
            self._gated = True
            self._gate()
        return super().load()


@pytest.mark.parametrize(
    ("mm_processor_cache_gb", "expected_workers"),
    [
        (None, 1),  # default: processor cache on, keep its order (#38418)
        (0, 2),  # cache off: renderer_num_workers requests at a time
    ],
)
def test_mm_preprocessing_concurrency(mm_processor_cache_gb, expected_workers):
    renderer_num_workers = 2
    cache_kwargs = (
        {}
        if mm_processor_cache_gb is None
        else {"mm_processor_cache_gb": mm_processor_cache_gb}
    )
    model_config = ModelConfig(
        model=MODEL,
        max_model_len=4096,
        renderer_num_workers=renderer_num_workers,
        **cache_kwargs,
    )
    renderer = HfRenderer(
        VllmConfig(model_config=model_config),
        cached_tokenizer_from_config(model_config),
    )

    # Each request waits here until `expected_workers` requests are being
    # preprocessed at once (or the timeout expires), then records its thread.
    barrier = threading.Barrier(expected_workers, timeout=60)
    threads: set[int] = set()

    def gate() -> None:
        threads.add(threading.get_ident())
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait()

    image = ImageAsset("cherry_blossom").pil_image.resize((224, 224))
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    prompt_token_ids = renderer.get_tokenizer().encode(
        f"{image_token}What is in this image?"
    )

    async def render(num_requests: int):
        return await asyncio.gather(
            *(
                renderer.process_for_engine_async(
                    {
                        "prompt_token_ids": prompt_token_ids,
                        "multi_modal_data": {"image": _GatedImage(image, gate)},
                    },
                    arrival_time=0.0,
                )
                for _ in range(num_requests)
            )
        )

    try:
        outputs = asyncio.run(render(2 * renderer_num_workers))
    finally:
        renderer.shutdown()

    assert len(threads) == expected_workers
    assert all(
        out["prompt_token_ids"] == outputs[0]["prompt_token_ids"] for out in outputs
    )
