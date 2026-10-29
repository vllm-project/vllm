# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight image PCP regression, selected by the four-GPU PCP nightly."""

import re

import pytest

from vllm import SamplingParams
from vllm.multimodal.utils import encode_image_url
from vllm.platforms import current_platform
from vllm.renderers.chat_utils import ChatCompletionMessageParam

from ..conftest import ImageTestAssets, VllmRunner
from ..models.registry import HF_EXAMPLE_MODELS


@pytest.mark.distributed(num_gpus=4)
def test_deepseek_v41_image_pcp(
    vllm_runner: type[VllmRunner], image_assets: ImageTestAssets, monkeypatch
):
    """Read image content across PCP chunks, then reuse it in a mixed batch."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = HF_EXAMPLE_MODELS.get_hf_info("DeepseekV41ForCausalLM").default
    image_chat: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_url(image_assets[0].pil_image)},
                },
                {
                    "type": "text",
                    "text": "Read the prominent English text in the image. "
                    "Reply with that text only.",
                },
            ],
        }
    ]
    sampling = SamplingParams(temperature=0, max_tokens=32)
    with vllm_runner(
        model,
        tokenizer_mode="deepseek_v41",
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        enable_expert_parallel=True,
        enforce_eager=True,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
        attention_backend="FLASHMLA_SPARSE_DSV41",
        moe_backend="humming",
        kv_cache_dtype="fp8_ds_mla",
        block_size=64 if current_platform.is_device_capability_family(90) else 128,
        max_model_len=4096,
        max_num_batched_tokens=128,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        swa_bounded_replay=True,
        disable_hybrid_kv_cache_manager=False,
        limit_mm_per_prompt={"image": 1},
    ) as runner:
        cold = runner.llm.chat(
            image_chat, sampling, chat_template_kwargs={"thinking": False}
        )[0]
        assert cold.finished
        assert cold.outputs[0].finish_reason in ("stop", "length")
        assert cold.prompt_token_ids is not None
        assert cold.prompt_token_ids.count(129264) > 128
        assert re.search(r"\bstop\b", cold.outputs[0].text, re.IGNORECASE)

        mixed = runner.llm.chat(
            [
                image_chat,
                [{"role": "user", "content": "What is 2 + 2? Reply with one digit."}],
            ],
            sampling,
            chat_template_kwargs={"thinking": False},
        )
        assert all(output.finished for output in mixed)
        assert all(
            output.outputs[0].finish_reason in ("stop", "length") for output in mixed
        )
        assert re.search(r"\bstop\b", mixed[0].outputs[0].text, re.IGNORECASE)
        assert re.search(r"\b4\b", mixed[1].outputs[0].text)
        assert mixed[0].num_cached_tokens is not None
        assert mixed[0].num_cached_tokens > 0
