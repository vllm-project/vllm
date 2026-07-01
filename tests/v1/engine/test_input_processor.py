# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import input_processor as input_processor_mod
from vllm.v1.engine.input_processor import InputProcessor


def test_mm_processor_cache_cleared_after_post_render_error(monkeypatch):
    processor = object.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
    )

    class FakeRenderer:
        mm_processor_cache = object()

        def __init__(self):
            self.clear_count = 0

        def clear_mm_cache(self):
            self.clear_count += 1

    fake_renderer = FakeRenderer()
    processor.renderer = fake_renderer
    processor._validate_params = lambda *args, **kwargs: None
    processor._validate_lora = lambda *args, **kwargs: None

    def raise_after_render(*args, **kwargs):
        raise ValueError("post-render validation failed")

    processor._validate_model_inputs = raise_after_render
    monkeypatch.setattr(
        input_processor_mod.current_platform,
        "validate_request",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="post-render validation failed"):
        processor.process_inputs(
            request_id="req-mm-cache-cleanup",
            prompt={"type": "token", "prompt_token_ids": [1]},
            params=SamplingParams(),
            supported_tasks=("generate",),
        )

    assert fake_renderer.clear_count == 1
