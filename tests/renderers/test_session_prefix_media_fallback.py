# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pre-image stages must preserve native image processing and histories."""

import os
from typing import Any, cast

import pytest

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tokenizers import get_tokenizer


@pytest.mark.asyncio
@pytest.mark.parametrize("token_reuse", [False, True])
async def test_multimodal_session_falls_back_to_native(token_reuse):
    """Image edits retain native IDs, media tensors and native block hashes."""
    import io

    import pybase64
    import torch
    from PIL import Image

    from vllm import SamplingParams
    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.multimodal.cache import engine_receiver_cache_from_config
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm.v1.request import Request

    config = VllmConfig(
        model_config=ModelConfig(
            model=os.environ.get(
                "VLLM_TEST_QWEN25_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"
            ),
            revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
            tokenizer_revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
            dtype="float16",
            max_model_len=8192,
            renderer_num_workers=1,
            session_token_cache_mb=8 if token_reuse else 0,
            limit_mm_per_prompt={"image": 4},
            mm_processor_cache_gb=0.1,
            mm_processor_cache_type="lru",
            mm_processor_device="cpu",
        ),
        cache_config=CacheConfig(enable_prefix_caching=True, block_size=16),
        scheduler_config=SchedulerConfig(
            is_encoder_decoder=False,
            max_model_len=8192,
            max_num_seqs=2,
            max_num_batched_tokens=2048,
        ),
    )
    renderer = HfRenderer(
        config,
        get_tokenizer(
            config.model_config.model,
            revision=config.model_config.tokenizer_revision,
        ),
    )
    online = OnlineRenderer(
        config.model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_mm_kwargs=True,
    )
    processor = InputProcessor(config, renderer)
    receiver = engine_receiver_cache_from_config(config)
    assert receiver is not None
    init_none_hash(sha256)
    hasher = get_request_block_hasher(16, sha256)
    images = []
    for color, size in (("red", (56, 56)), ("blue", (56, 56)), ("green", (84, 56))):
        stream = io.BytesIO()
        Image.new("RGB", size, color).save(stream, format="PNG")
        images.append(
            "data:image/png;base64," + pybase64.b64encode(stream.getvalue()).decode()
        )
    cases = [(0,), (0,), (1,), (2,), (0, 1), (1, 0), (0,), (), (0,)]
    try:
        if token_reuse:
            assert renderer.session_token_cache is not None
            assert renderer.session_token_cache.enabled
        for index, selected in enumerate(cases):
            content: list[dict[str, Any]] = [
                {"type": "text", "text": "Describe 截图 e\u0301 👩‍💻."}
            ]
            content.extend(
                {"type": "image_url", "image_url": {"url": images[item]}}
                for item in selected
            )
            messages = [
                {"role": "system", "content": "Preserve the visible history. " * 40},
                {"role": "user", "content": content},
            ]
            request = ChatCompletionRequest(
                model=config.model_config.model,
                messages=cast(Any, messages),
                max_tokens=1,
            )
            _, reference = await online.render_chat(request.model_copy(deep=True))
            cache = renderer.session_token_cache
            retained = cache.store.bytes_used if cache else 0
            _, candidate = await online.render_chat(
                request.model_copy(deep=True), session_id="images"
            )
            if selected and cache:
                assert cache.store.bytes_used == retained
            expected_input = processor.process_inputs(
                f"reference-{index}",
                reference[0],
                SamplingParams(max_tokens=1),
                ("generate",),
            )
            incoming = processor.process_inputs(
                f"candidate-{index}",
                candidate[0],
                SamplingParams(max_tokens=1),
                ("generate",),
                session_id="images",
            )
            assert incoming.prompt_token_ids == expected_input.prompt_token_ids
            if expected_input.mm_features:
                receiver.get_and_update_features(expected_input.mm_features)
            if incoming.mm_features:
                receiver.get_and_update_features(incoming.mm_features)
            assert incoming.mm_features == expected_input.mm_features
            actual = Request.from_engine_core_request(incoming, hasher)
            expected = Request.from_engine_core_request(expected_input, hasher)
            assert actual.block_hashes == expected.block_hashes
    finally:
        renderer.shutdown()
    assert not torch.cuda.is_initialized()
