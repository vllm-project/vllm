# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reloading weights must invalidate the scheduler's encoder cache too. The
model runner already clears its own cache during `reload_weights`, but before
the fix the scheduler's `EncoderCacheManager` still counted pre-reload hashes
as cache hits, so the next request reusing a previously-seen image was
scheduled as a hit and the engine died with "Encoder cache miss" (#67088)."""

import pytest
from PIL import Image, ImageDraw

from vllm import SamplingParams

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "Describe this image.<|im_end|>\n<|im_start|>assistant\n"
)


def _test_image() -> Image.Image:
    img = Image.new("RGB", (448, 448), "white")
    ImageDraw.Draw(img).rectangle([40, 60, 130, 150], fill=(220, 30, 30))
    return img


def test_reload_weights_then_same_image_succeeds(vllm_runner):
    with vllm_runner(
        MODEL,
        max_model_len=4096,
        enforce_eager=True,
    ) as runner:
        llm = runner.llm
        params = SamplingParams(temperature=0.0, max_tokens=16)
        request = {
            "prompt": PROMPT,
            "multi_modal_data": {"image": _test_image()},
        }

        first = llm.generate([request], params)
        assert first[0].outputs[0].token_ids

        # Before the fix this second generate raised EngineDeadError:
        # "Encoder cache miss for <mm_hash>".
        llm.collective_rpc("reload_weights")
        second = llm.generate([request], params)
        assert second[0].outputs[0].token_ids


@pytest.mark.parametrize("reset_prefix_cache_first", [False, True])
def test_reload_weights_survives_prefix_cache_reset(
    reset_prefix_cache_first, vllm_runner
):
    """The repro in #67088 crashed even with an explicit prefix-cache reset
    after the reload: `reset_prefix_cache` does not touch the encoder cache
    manager, so both orderings must work."""
    with vllm_runner(
        MODEL,
        max_model_len=4096,
        enforce_eager=True,
    ) as runner:
        llm = runner.llm
        params = SamplingParams(temperature=0.0, max_tokens=16)
        request = {
            "prompt": PROMPT,
            "multi_modal_data": {"image": _test_image()},
        }

        assert llm.generate([request], params)[0].outputs[0].token_ids

        if reset_prefix_cache_first:
            llm.reset_prefix_cache()
        llm.collective_rpc("reload_weights")
        if not reset_prefix_cache_first:
            llm.reset_prefix_cache()

        assert llm.generate([request], params)[0].outputs[0].token_ids
