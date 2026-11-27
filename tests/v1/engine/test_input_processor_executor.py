# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`InputProcessor.process_inputs_async` routes raw prompts with multimodal
data to the renderer's single-worker MM executor.

The renderer thread pool can have several workers, but the multimodal
processor cache and caller-owned media (e.g. a lazily decoded PIL image shared
by concurrent requests) are not thread-safe.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm.utils.async_utils import make_async
from vllm.v1.engine.input_processor import InputProcessor, _has_multimodal_data


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("hello", False),
        ([1, 2, 3], False),
        ({"prompt": "hello"}, False),
        ({"prompt_token_ids": [1, 2]}, False),
        ({"prompt": "hello", "multi_modal_data": None}, False),
        ({"prompt": "hello", "multi_modal_data": {"image": object()}}, True),
        ({"prompt_token_ids": [1], "multi_modal_data": {"image": None}}, True),
        (
            {
                "encoder_prompt": {"prompt": "a", "multi_modal_data": {"audio": 0}},
                "decoder_prompt": "b",
            },
            True,
        ),
        ({"encoder_prompt": "a", "decoder_prompt": None}, False),
    ],
)
def test_has_multimodal_data(prompt, expected):
    assert _has_multimodal_data(prompt) is expected


def test_process_inputs_async_routes_by_modality():
    text_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="text")
    mm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mm")

    processor = object.__new__(InputProcessor)

    def fake_process_inputs(request_id, prompt, *args, **kwargs):
        return threading.current_thread().name

    processor.process_inputs = fake_process_inputs  # type: ignore[method-assign]
    processor._process_text_inputs_async = make_async(
        fake_process_inputs, executor=text_pool
    )
    processor._process_mm_inputs_async = make_async(
        fake_process_inputs, executor=mm_pool
    )

    async def run():
        text = await processor.process_inputs_async("r0", {"prompt": "hi"}, None)
        mm = await processor.process_inputs_async(
            "r1", {"prompt": "hi", "multi_modal_data": {"image": object()}}, None
        )
        return text, mm

    try:
        text_thread, mm_thread = asyncio.run(run())
    finally:
        text_pool.shutdown()
        mm_pool.shutdown()

    assert text_thread.startswith("text")
    assert mm_thread.startswith("mm")
