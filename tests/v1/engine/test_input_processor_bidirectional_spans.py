# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation of bidirectional attention spans in InputProcessor."""

import asyncio

import numpy as np
import pytest
import torch
from PIL import Image

from vllm import PoolingParams, SamplingParams
from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import (
    EngineInput,
    TextPrompt,
    TokensPrompt,
    embeds_input,
    tokens_input,
)
from vllm.renderers import TokenizeParams
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.v1.engine.input_processor import InputProcessor

pytestmark = pytest.mark.cpu_test

MODEL = "google/gemma-4-E2B-it"
SPANS = {
    "bidirectional_span_layer_types": ["full_attention"],
    "bidirectional_span_max_length": 4096,
}


def make_processor(**scheduler_kwargs) -> InputProcessor:
    return InputProcessor(
        VllmConfig(
            model_config=ModelConfig(
                MODEL,
                hf_overrides={"text_config": SPANS},
                max_model_len=8192,
                limit_mm_per_prompt={"image": 1, "audio": 0, "video": 0},
            ),
            scheduler_config=SchedulerConfig(
                max_model_len=8192,
                is_encoder_decoder=False,
                **{"max_num_batched_tokens": 3000, **scheduler_kwargs},
            ),
        )
    )


@pytest.fixture(scope="module")
def processor() -> InputProcessor:
    return make_processor()


def process(
    processor: InputProcessor,
    engine_input: EngineInput,
    params: SamplingParams | PoolingParams | None = None,
    resumable: bool = False,
):
    return processor.process_inputs(
        "req",
        engine_input,
        params or SamplingParams(max_tokens=1),
        supported_tasks=("generate",),
        resumable=resumable,
    )


def with_spans(spans, num_tokens: int = 40) -> EngineInput:
    engine_input = tokens_input(list(range(100, 100 + num_tokens)))
    if spans is not None:
        engine_input["bidirectional_spans"] = spans
    return engine_input


def test_spans_are_returned_without_one_token_spans(processor):
    request = process(processor, with_spans([[0, 1], [1, 9], (9, 10), [10, 40]]))
    assert request.bidirectional_spans == [(1, 9), (10, 40)]
    # An empty list states a causal request.
    assert process(processor, with_spans([])).bidirectional_spans == []


def test_a_checkpoint_that_declares_spans_requires_them(processor):
    with pytest.raises(VLLMValidationError, match="must set bidirectional_spans"):
        process(processor, with_spans(None))


@pytest.mark.parametrize(
    ("spans", "match"),
    [
        ([[0, 5001]], "prompt length"),
        ([[5, 5]], "0 <= start < end"),
        ([[-1, 3]], "0 <= start < end"),
        ([[8, 16], [0, 8]], "sorted and disjoint"),
        ([[0, 10], [9, 16]], "sorted and disjoint"),
        ([[0, True]], "pair of integers"),
        ([[0, 1.5]], "pair of integers"),
        ([[0, 1, 2]], "pair of integers"),
        ([3], "pair of integers"),
        # More than the 3000 tokens of --max-num-batched-tokens.
        ([[0, 3001]], "in one step"),
        ([[0, 4097]], "at most 4096"),
    ],
)
def test_invalid_spans_are_refused(processor, spans, match):
    with pytest.raises(VLLMValidationError, match=match):
        process(processor, with_spans(spans, num_tokens=5000))


def test_spans_are_refused_with_streaming_input_and_prompt_logprobs(processor):
    with pytest.raises(VLLMValidationError, match="streaming input"):
        process(processor, with_spans([[0, 8]]), resumable=True)
    with pytest.raises(VLLMValidationError, match="prompt logprobs"):
        process(
            processor,
            with_spans([[0, 8]]),
            SamplingParams(max_tokens=1, prompt_logprobs=0),
        )


def test_spans_reach_pooling_requests(processor):
    request = processor.process_inputs(
        "req",
        with_spans([[0, 8]]),
        PoolingParams(task="embed"),
        supported_tasks=("embed",),
    )
    assert request.bidirectional_spans == [(0, 8)]


@pytest.mark.parametrize(
    "scheduler_kwargs",
    [
        {"max_num_scheduled_tokens": 1000},
        {"long_prefill_token_threshold": 1000},
        {"max_num_batched_tokens": 1000},
    ],
)
def test_each_span_fits_one_scheduler_step(scheduler_kwargs):
    """The scheduler prefills the rest of a span in one step."""
    processor = make_processor(**scheduler_kwargs)
    process(processor, with_spans([[0, 1000]], num_tokens=1200))
    with pytest.raises(VLLMValidationError, match="more than the 1000 tokens"):
        process(processor, with_spans([[0, 1001]], num_tokens=1200))


def test_spans_are_refused_with_prompt_embeds(processor):
    engine_input = embeds_input(torch.zeros(10, 8))
    engine_input["bidirectional_spans"] = [(0, 8)]  # type: ignore[typeddict-unknown-key]
    with pytest.raises(VLLMValidationError, match="prompt embeds"):
        process(processor, engine_input)


def image_input(processor: InputProcessor, uuid: str | None = None) -> EngineInput:
    prompt = {
        "prompt": "<|image|> Describe the image.",
        "multi_modal_data": {"image": Image.new("RGB", (64, 64), "teal")},
    }
    if uuid is not None:
        prompt["multi_modal_uuids"] = {"image": [uuid]}
    renderer = processor.renderer
    (engine_input,) = renderer.render_cmpl(
        [parse_model_prompt(processor.model_config, prompt)],
        renderer.default_cmpl_tok_params,
    )
    assert engine_input["type"] == "multimodal"
    return engine_input


def test_server_set_spans_hold_each_multimodal_item_whole(processor):
    engine_input = image_input(processor)
    (position,) = engine_input["mm_placeholders"]["image"]
    end = position.offset + position.length
    num_tokens = len(engine_input["prompt_token_ids"])

    engine_input["bidirectional_spans"] = [(0, end), (end, num_tokens)]
    request = process(processor, engine_input)
    assert request.bidirectional_spans == [(0, end), (end, num_tokens)]

    # The image straddles the boundary.
    engine_input["bidirectional_spans"] = [(0, end - 1), (end - 1, num_tokens)]
    with pytest.raises(VLLMValidationError, match="inside one bidirectional span"):
        process(processor, engine_input)

    # No span holds the image.
    engine_input["bidirectional_spans"] = [(end, num_tokens)]
    with pytest.raises(VLLMValidationError, match="inside one bidirectional span"):
        process(processor, engine_input)


def test_the_items_of_a_span_fit_the_encoder_cache_together():
    """The scheduler encodes a span's items in the step that runs it, so they
    must fit the encoder cache together. A cache that only takes one item at a
    time would never fit them, and the request would hold the queue."""
    spans = {**SPANS, "bidirectional_span_max_length": 8192}

    def make(encoder_cache_size: int | None) -> InputProcessor:
        return InputProcessor(
            VllmConfig(
                model_config=ModelConfig(
                    MODEL,
                    hf_overrides={"text_config": spans},
                    max_model_len=8192,
                    limit_mm_per_prompt={"image": 2, "audio": 0, "video": 0},
                ),
                scheduler_config=SchedulerConfig(
                    max_model_len=8192,
                    is_encoder_decoder=False,
                    max_num_batched_tokens=8192,
                    # vLLM raises it to the largest item.
                    encoder_cache_size=encoder_cache_size,
                ),
            )
        )

    small, default = make(1), make(None)
    rng = np.random.default_rng(0)
    images = [
        Image.fromarray(rng.integers(0, 256, (1536, 1536, 3), dtype=np.uint8))
        for _ in range(2)
    ]
    renderer = small.renderer
    (engine_input,) = renderer.render_cmpl(
        [
            parse_model_prompt(
                small.model_config,
                {
                    "prompt": "<|image|><|image|> Compare the images.",
                    "multi_modal_data": {"image": images},
                },
            )
        ],
        renderer.default_cmpl_tok_params,
    )
    positions = engine_input["mm_placeholders"]["image"]
    num_embeds = sum(position.get_num_embeds() for position in positions)
    assert max(p.get_num_embeds() for p in positions) <= small.mm_encoder_cache_size
    assert small.mm_encoder_cache_size < num_embeds <= default.mm_encoder_cache_size
    num_tokens = len(engine_input["prompt_token_ids"])
    split = positions[1].offset

    engine_input["bidirectional_spans"] = [(0, num_tokens)]
    with pytest.raises(VLLMValidationError, match="--encoder-cache-size"):
        process(small, engine_input)
    assert process(default, engine_input).bidirectional_spans == [(0, num_tokens)]
    # One item per span fits.
    engine_input["bidirectional_spans"] = [(0, split), (split, num_tokens)]
    assert process(small, engine_input).bidirectional_spans == [
        (0, split),
        (split, num_tokens),
    ]


def test_multimodal_prompts_can_be_causal(processor):
    """An empty list sends a multimodal prompt without spans; no item then
    needs to lie inside one."""
    engine_input = image_input(processor)
    engine_input["bidirectional_spans"] = []
    assert process(processor, engine_input).bidirectional_spans == []


def test_server_set_spans_refuse_caller_media_ids(processor):
    engine_input = image_input(processor, uuid="my-image")
    engine_input["bidirectional_spans"] = [(0, len(engine_input["prompt_token_ids"]))]
    with pytest.raises(VLLMValidationError, match="multi_modal_uuids"):
        process(processor, engine_input)


def test_models_that_do_not_declare_spans_refuse_them():
    processor = InputProcessor(VllmConfig(model_config=ModelConfig(MODEL)))
    with pytest.raises(VLLMValidationError, match="does not support"):
        process(processor, with_spans([[0, 8]]))
    assert process(processor, with_spans(None)).bidirectional_spans is None


@pytest.mark.parametrize("use_async", [False, True])
def test_client_set_spans_need_an_unchanged_token_id_prompt(processor, use_async):
    renderer = processor.renderer

    def render(prompt, tok_params=None):
        prompts = [parse_model_prompt(processor.model_config, prompt)]
        tok_params = tok_params or renderer.default_cmpl_tok_params
        if use_async:
            return asyncio.run(renderer.render_cmpl_async(prompts, tok_params))
        return renderer.render_cmpl(prompts, tok_params)

    ids = list(range(100, 140))
    (engine_input,) = render(
        TokensPrompt(prompt_token_ids=ids, bidirectional_spans=[(0, 20)])
    )
    assert engine_input["bidirectional_spans"] == [(0, 20)]

    # The client cannot know the positions of tokens the server tokenizes.
    text = TextPrompt(prompt="Describe the state.")
    text["bidirectional_spans"] = [(0, 2)]  # type: ignore[typeddict-unknown-key]
    with pytest.raises(VLLMValidationError, match="token-id prompt"):
        render(text)
    # Truncation and placeholder expansion move them.
    with pytest.raises(VLLMValidationError, match="truncation"):
        render(
            TokensPrompt(prompt_token_ids=ids, bidirectional_spans=[(0, 20)]),
            TokenizeParams(max_total_tokens=4096, truncate_prompt_tokens=30),
        )
    with pytest.raises(VLLMValidationError, match="multimodal data"):
        render(
            TokensPrompt(
                prompt_token_ids=[
                    *ids,
                    processor.model_config.hf_config.image_token_id,
                ],
                multi_modal_data={"image": Image.new("RGB", (64, 64))},
                bidirectional_spans=[(0, 20)],
            )
        )
