# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma 4 with bidirectional attention spans, against a Hugging Face
reference built from the same masks, with prefix caching and chunked prefill on.

The checkpoint is a stock Gemma 4; `hf_overrides` declares spans on its
full-attention layers, or on every layer, which builds its sliding-window layers
without a window, as a checkpoint trained with them would in its config.
On such a checkpoint every request states its spans, and an empty list makes
it causal. For local validation, BIDIRECTIONAL_SPANS_MODEL,
BIDIRECTIONAL_SPANS_DRAFTER, BIDIRECTIONAL_SPANS_LENGTH and
BIDIRECTIONAL_SPANS_ENGINE_ARGS (JSON, for example
`{"tensor_parallel_size": 2}`) select another checkpoint, drafter, span length
or engine arguments. The cache tests need prompts longer than the checkpoint's
prefix-cache hit alignment.
"""

import gc
import json
import math
import os
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import regex as re
import torch
from PIL import Image

from tests.models.utils import check_logprobs_close
from vllm import LLM, PoolingParams, SamplingParams, TokensPrompt
from vllm.config import CompilationConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.platforms import current_platform
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.metrics.reader import Counter

MODEL = os.environ.get("BIDIRECTIONAL_SPANS_MODEL", "google/gemma-4-E2B-it")
DRAFTER = os.environ.get(
    "BIDIRECTIONAL_SPANS_DRAFTER", "google/gemma-4-E2B-it-assistant"
)
SPAN_LENGTH = int(os.environ.get("BIDIRECTIONAL_SPANS_LENGTH", "64"))
ENGINE_ARGS: dict[str, Any] = json.loads(
    os.environ.get("BIDIRECTIONAL_SPANS_ENGINE_ARGS", "{}")
)
TOP_K = 20
# Two top-1 tokens are a near tie when each is in the other's top CLOSE_K.
CLOSE_K = 5
MAX_STATE_TOKENS = 3000
NUM_GENERATED = 16
# The image cases' state holds an image, so their spans are longer.
IMAGE_SPAN_LENGTH = 1024
# Spans on every layer, with the sliding-window layers built without a window.
ALL_LAYERS = ("full_attention", "sliding_attention")
SLIDING_LAYERS = ("sliding_attention",)

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="needs a CUDA GPU"),
    pytest.mark.slow_test,
]


@dataclass(frozen=True)
class Layout:
    span_layer_types: tuple[str, ...] = ("full_attention",)
    drafter: bool = False

    @property
    def unwindowed(self) -> bool:
        """Whether the sliding-window layers apply spans, without a window."""
        return "sliding_attention" in self.span_layer_types


STATES = [
    (
        "The agent opened the settings page, changed the shipping address to "
        "221B Baker Street, saved the form and saw a confirmation banner. It then "
        "returned to the cart, where the order total still showed the old address."
    ),
    (
        "A user asked for a refund on order 1182. The agent looked up the order, "
        "found that it was delivered nine days ago, checked the policy (refunds "
        "within 30 days) and issued the refund, but did not email the user."
    ),
    (
        "The script read 4,000 rows from the input file, dropped 12 rows with "
        "missing dates, wrote 3,988 rows to the output table and exited with "
        "status 0. The monitoring job reported 3,990 rows in the table an hour "
        "later."
    ),
    (
        "During the meeting the team agreed to move the launch to Thursday. Maria "
        "will update the release notes, Ken will tell support, and nobody has yet "
        "been assigned to update the status page or the public calendar."
    ),
]
QUESTIONS = [
    "Did the agent complete the task? Answer yes or no.",
    "Is any follow-up action still needed? Answer yes or no.",
]
# With a drafter, several tokens are generated, so the answer is a sentence.
DRAFTER_QUESTIONS = ["In one sentence, what is still left to do?"]
IMAGE_STATE = "The screenshot shows the order page after the agent saved the form."


def render(tokenizer, state: str, question: str) -> tuple[list[int], int]:
    """Token ids of the chat-templated prompt and the state's token count."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{state}\n\nQuestion:\n{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    boundary = text.index("\n\nQuestion:\n")
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    num_state = sum(1 for _, end in enc["offset_mapping"] if end <= boundary)
    return enc["input_ids"], num_state


def chunked(num_state: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size, num_state)) for s in range(0, num_state, size)]


def text_cases(
    questions: list[str], window: int | None = None
) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """States of two to four fixed-size spans, so that several spans and
    prefill chunks are involved whatever the span length, then the question as
    a span of its own, which holds the last prompt token. With a `window`, each
    state is one span, longer than the window, which would cut the span and the
    question's view of the state."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    out = []
    for i, state in enumerate(STATES):
        if window is None:
            target = min(int((2 + i / 2) * SPAN_LENGTH), MAX_STATE_TOKENS)
        else:
            target = min(window + 64 + i * window // 4, MAX_STATE_TOKENS)
        repeats = 1
        while len(tokenizer(" ".join([state] * repeats))["input_ids"]) < target:
            repeats += 1
        state = " ".join([state] * repeats)
        for question in questions:
            ids, num_state = render(tokenizer, state, question)
            assert len(ids) - num_state <= SPAN_LENGTH, "the question needs one span"
            if window is None:
                state_spans = chunked(num_state, SPAN_LENGTH)
            else:
                assert window < num_state <= 2 * window, (num_state, window)
                state_spans = [(0, num_state)]
            out.append((ids, state_spans + [(num_state, len(ids))]))
    return out


def span_masks(
    n: int,
    spans,
    layer_types,
    span_layer_types,
    window: int,
    image_ranges=(),
    windowed=False,
):
    """Masks per layer type: the listed types apply the spans and no image
    ranges, without a window unless `windowed`; the other sliding-window layers
    keep their window, with the inclusive image ranges bidirectional inside it
    (Gemma 4's "vision")."""
    idx = torch.arange(n, device="cuda")
    causal = idx[None, :] <= idx[:, None]

    def same_range(ranges, inclusive: bool):
        range_id = torch.full((n,), -1, device="cuda")
        for i, (start, end) in enumerate(ranges):
            range_id[start : end + 1 if inclusive else end] = i
        return (range_id[:, None] == range_id[None, :]) & (range_id[:, None] >= 0)

    in_window = (idx[:, None] - idx[None, :]) < window
    masks = {}
    for layer_type in set(layer_types):
        if layer_type in span_layer_types:
            mask = causal | same_range(spans, inclusive=False)
            if windowed and layer_type == "sliding_attention":
                mask = mask & in_window
        elif layer_type == "sliding_attention":
            mask = (causal | same_range(image_ranges, inclusive=True)) & in_window
        else:
            mask = causal
        masks[layer_type] = mask[None, None]
    return masks


def hf_forward(
    model,
    text_config,
    ids,
    spans,
    image_ranges=(),
    span_layer_types=("full_attention",),
    windowed=False,
    **mm_inputs,
):
    """Last-position logits and final hidden state of the transformers model
    under `span_masks`."""
    n = len(ids)
    out = model(
        input_ids=torch.tensor([ids], device="cuda"),
        attention_mask=span_masks(
            n,
            spans,
            text_config.layer_types,
            span_layer_types,
            text_config.sliding_window,
            image_ranges,
            windowed,
        ),
        position_ids=torch.arange(n, device="cuda")[None],
        output_hidden_states=True,
        **mm_inputs,
    )
    return out.logits[0, -1].float(), out.hidden_states[-1][0, -1].float()


def image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))


def image_overrides() -> dict[str, Any]:
    return {
        "text_config": {
            "use_bidirectional_attention": "vision",
            "bidirectional_span_layer_types": ["full_attention"],
            "bidirectional_span_max_length": IMAGE_SPAN_LENGTH,
        }
    }


def image_cases() -> list[dict[str, Any]]:
    """Image prompts rendered by vLLM, with the state (image included) and the
    question as one span each, the image range vLLM gives the sliding-window
    layers, and the transformers processor's inputs."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL)
    renderer = renderer_from_config(
        VllmConfig(
            model_config=ModelConfig(
                MODEL,
                hf_overrides=image_overrides(),
                limit_mm_per_prompt={"image": 1, "audio": 0},
            )
        )
    )
    out = []
    for i, question in enumerate(QUESTIONS):
        text = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{IMAGE_STATE}\n{question}"},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        img = image(i)
        prompt = TokensPrompt(
            prompt_token_ids=processor.tokenizer(text, add_special_tokens=False)[
                "input_ids"
            ],
            multi_modal_data={"image": img},
        )
        (engine_input,) = renderer.render_cmpl(
            [parse_model_prompt(renderer.model_config, prompt)]
        )
        hf_inputs = processor(
            text=text, images=[img], add_special_tokens=False, return_tensors="pt"
        )
        ids = engine_input["prompt_token_ids"]
        assert hf_inputs["input_ids"][0].tolist() == ids
        (position,) = engine_input["mm_placeholders"]["image"]
        image_end = position.offset + position.length
        state_end = image_end + (len(ids) - image_end) // 2
        mm_inputs = {
            k: v
            for k, v in hf_inputs.items()
            if k not in ("input_ids", "attention_mask")
        }
        out.append(
            {
                "prompt": prompt,
                "ids": ids,
                "spans": [(0, state_end), (state_end, len(ids))],
                "image_ranges": position.extract_embeds_range(),
                "mm_inputs": mm_inputs,
            }
        )
    return out


@pytest.fixture(scope="module")
def text_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODEL).text_config


def text_references(
    model, text_config, layout: Layout, with_drafter_cases: bool
) -> dict[Any, Any]:
    """The text cases of a layout, their next-token probabilities and final
    hidden states with and without spans, and, for a layout with a drafter
    class, greedy continuations of the drafter cases. With sliding-window
    layers listed, also the probabilities with their window kept."""
    window = text_config.sliding_window if layout.unwindowed else None
    cases = text_cases(QUESTIONS, window)
    drafter_cases = text_cases(DRAFTER_QUESTIONS, window) if with_drafter_cases else []
    ref: dict[Any, Any] = {"cases": cases, "drafter_cases": drafter_cases}

    def forward(ids, spans, windowed=False):
        return hf_forward(
            model,
            text_config,
            ids,
            spans,
            span_layer_types=layout.span_layer_types,
            windowed=windowed,
        )

    for i, (ids, spans) in enumerate(cases):
        for mode, mode_spans in (("spans", spans), ("causal", [])):
            logits, hidden = forward(ids, mode_spans)
            ref["text", i, mode] = torch.softmax(logits, -1).cpu()
            ref["pooled", i, mode] = torch.nn.functional.normalize(hidden, dim=-1).cpu()
            if layout.unwindowed:
                logits, _ = forward(ids, mode_spans, windowed=True)
                ref["windowed", i, mode] = torch.softmax(logits, -1).cpu()
    for i, (ids, spans) in enumerate(drafter_cases):
        for mode, mode_spans in (("spans", spans), ("causal", [])):
            logits, _ = forward(ids, mode_spans)
            ref["first", i, mode] = torch.softmax(logits, -1).cpu()
        tokens: list[int] = []
        logprobs: list[dict[int, float]] = []
        for _ in range(NUM_GENERATED):
            logits, _ = forward(ids + tokens, spans)
            best = torch.log_softmax(logits, -1).topk(TOP_K)
            logprobs.append(dict(zip(best.indices.tolist(), best.values.tolist())))
            tokens.append(int(best.indices[0]))
        ref["greedy", i] = tokens, logprobs
    return ref


@pytest.fixture(scope="module")
def all_references(text_config):
    """From transformers, in one load of the model: the text references of
    each layout, by its layer types, and with the full-attention layout the
    image cases."""
    from transformers import AutoModelForImageTextToText

    # The boolean masks need SDPA on the language model. The audio tower has
    # no SDPA path and keeps its default.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation={"text_config": "sdpa"}
    )
    model = model.cuda().eval()
    refs: dict[tuple[str, ...], dict[Any, Any]] = {}
    with torch.no_grad():
        for layout, with_drafter_cases in (
            (Layout(), True),
            (Layout(ALL_LAYERS), True),
            (Layout(SLIDING_LAYERS), False),
        ):
            refs[layout.span_layer_types] = text_references(
                model, text_config, layout, with_drafter_cases
            )
        ref = refs[Layout().span_layer_types]
        ref["images"] = image_cases()
        for case in ref["images"]:
            mm_inputs = {k: v.cuda() for k, v in case.pop("mm_inputs").items()}
            for mode, mode_spans, image_ranges in (
                ("spans", case["spans"], case["image_ranges"]),
                ("causal", [], case["image_ranges"]),
            ):
                logits, _ = hf_forward(
                    model,
                    text_config,
                    case["ids"],
                    mode_spans,
                    image_ranges,
                    **mm_inputs,
                )
                case[f"ref_{mode}"] = torch.softmax(logits, -1).cpu()
    del model
    free_gpu_memory()
    return refs


def free_gpu_memory() -> None:
    gc.collect()
    torch.accelerator.empty_cache()


def make_llm(**kwargs):
    """An LLM with the given arguments, overridden by the local ones, and a
    proxy to it for the tests, so that the engine is released with its class."""
    engine = LLM(MODEL, **{**kwargs, **ENGINE_ARGS})
    return engine, weakref.proxy(engine)


def span_overrides(layout: Layout, max_length: int) -> dict[str, Any]:
    return {
        "text_config": {
            "bidirectional_span_layer_types": list(layout.span_layer_types),
            "bidirectional_span_max_length": max_length,
        }
    }


@pytest.fixture(scope="class")
def layout(request) -> Layout:
    return request.cls.layout


@pytest.fixture(scope="class")
def references(layout, all_references):
    return all_references[layout.span_layer_types]


@pytest.fixture(scope="class")
def cases(references):
    return references["cases"]


@pytest.fixture(scope="class")
def drafter_cases(references):
    return references["drafter_cases"]


def span_max_length(layout: Layout, text_config) -> int:
    """Spans of SPAN_LENGTH tokens, or without a window, states longer than
    the window."""
    if layout.unwindowed:
        return max(SPAN_LENGTH, 2 * text_config.sliding_window)
    return SPAN_LENGTH


@pytest.fixture(scope="class")
def llm(layout, references, text_config):
    # A per-step budget just above the longest span, so that prompts are split
    # at span boundaries, with room for the drafter's slots next to a span.
    max_length = span_max_length(layout, text_config)
    engine, proxy = make_llm(
        hf_overrides=span_overrides(layout, max_length),
        max_model_len=MAX_STATE_TOKENS + 512,
        max_num_batched_tokens=max(128, max_length + 16),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        limit_mm_per_prompt={"image": 0, "audio": 0},
        speculative_config=(
            {"method": "mtp", "model": DRAFTER, "num_speculative_tokens": 2}
            if layout.drafter
            else None
        ),
        disable_log_stats=not layout.drafter,
    )
    yield proxy
    del engine
    free_gpu_memory()


PARAMS = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP_K)


def first_logprobs(output) -> dict[int, float]:
    return {t: lp.logprob for t, lp in output.outputs[0].logprobs[0].items()}


def run(llm, prompts):
    outputs = llm.generate(prompts, PARAMS, use_tqdm=False)
    return [(first_logprobs(o), o.num_cached_tokens or 0) for o in outputs]


def distance(a: dict[int, float], ref: torch.Tensor) -> float:
    """Sum of absolute probability differences over vLLM's top tokens."""
    return sum(abs(math.exp(lp) - ref[t].item()) for t, lp in a.items())


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def top(dist: dict[int, float] | torch.Tensor, k: int) -> dict[int, float]:
    """The k most likely tokens of a log-probability dict or a probability
    tensor, as a log-probability dict."""
    if isinstance(dist, torch.Tensor):
        values, indices = dist.log().topk(k)
        return dict(zip(indices.tolist(), values.tolist()))
    return dict(sorted(dist.items(), key=lambda item: -item[1])[:k])


def check_top1_close(results: list[dict[int, float]], refs: list[torch.Tensor]):
    """Each top-1 token is the reference's, or the two are near ties."""
    check_logprobs_close(
        outputs_0_lst=[([int(ref.argmax())], "", [top(ref, CLOSE_K)]) for ref in refs],
        outputs_1_lst=[
            ([max(lp, key=lp.get)], "", [top(lp, CLOSE_K)]) for lp in results
        ],
        name_0="hf",
        name_1="vllm",
    )


def prompt(ids, spans):
    return TokensPrompt(prompt_token_ids=ids, bidirectional_spans=spans)


def causal(ids):
    return prompt(ids, [])


def check_against_references(results, plain, refs, refs_causal, bound: float):
    """Spans served by vLLM match the span reference about as closely as causal
    requests match the causal one, and move the answer by more than that."""
    check_top1_close([lp for lp, _ in results], refs)
    check_top1_close([lp for lp, _ in plain], refs_causal)
    span_err = [distance(lp, ref) for (lp, _), ref in zip(results, refs)]
    causal_err = [distance(lp, ref) for (lp, _), ref in zip(plain, refs_causal)]
    effect = [distance(lp, ref) for (lp, _), ref in zip(results, refs_causal)]
    assert mean(span_err) < max(2 * mean(causal_err), bound), (span_err, causal_err)
    assert mean(effect) > 2 * mean(span_err), (effect, span_err)


class SpanTests:
    """The tests of every text layout without a drafter."""

    reference_bound = 0.02

    def test_spans_match_the_reference(self, llm, cases, references):
        llm.reset_prefix_cache()
        with_spans = run(llm, [prompt(ids, spans) for ids, spans in cases])
        plain = run(llm, [causal(ids) for ids, _ in cases])
        check_against_references(
            with_spans,
            plain,
            [references["text", i, "spans"] for i in range(len(cases))],
            [references["text", i, "causal"] for i in range(len(cases))],
            bound=self.reference_bound,
        )

    def test_cached_spans_match_the_reference(self, llm, cases, references):
        """Span requests served from the prefix cache match the reference as
        closely as uncached ones do."""
        span_prompts = [prompt(ids, spans) for ids, spans in cases]
        plain_prompts = [causal(ids) for ids, _ in cases]
        # Cases of one state share its tokens, so each half runs alone.
        uncached: list[Any] = [None] * len(cases)
        for parity in (0, 1):
            llm.reset_prefix_cache()
            indices = range(parity, len(cases), 2)
            for i, result in zip(indices, run(llm, [span_prompts[i] for i in indices])):
                uncached[i] = result
        assert all(num_cached == 0 for _, num_cached in uncached)
        llm.reset_prefix_cache()
        run(llm, span_prompts + plain_prompts)
        cached = [run(llm, span_prompts), run(llm, plain_prompts)]
        assert all(num_cached > 0 for _, num_cached in cached[0])
        for hits, mode in zip(cached, ("spans", "causal")):
            check_top1_close(
                [lp for lp, _ in hits],
                [references["text", i, mode] for i in range(len(cases))],
            )
        # Hits are not bitwise equal to full prefills (other kernels), so the
        # bar is the uncached requests' distance to the same reference.
        refs = [references["text", i, "spans"] for i in range(len(cases))]
        cached_err = [distance(lp, ref) for (lp, _), ref in zip(cached[0], refs)]
        uncached_err = [distance(lp, ref) for (lp, _), ref in zip(uncached, refs)]
        assert mean(cached_err) < max(2 * mean(uncached_err), 0.05), (
            cached_err,
            uncached_err,
        )

    def test_grown_prompts_reuse_complete_spans_only(self, llm):
        """A state that grows inside its last span reuses the blocks of its
        complete spans and nothing of the span that changed."""
        # The complete spans end at 2,048 tokens, a multiple of the KV cache's
        # hit alignment, so the hit is exactly there. With short spans the state
        # may end within one alignment of it, where only the other-bounds test
        # tells stale reuse apart.
        num_complete = 2048
        assert num_complete % SPAN_LENGTH == 0, SPAN_LENGTH
        tokenizer = llm.get_tokenizer()
        text = tokenizer(" ".join(STATES), add_special_tokens=False)["input_ids"]
        state = (text * (num_complete // len(text) + 2))[
            : num_complete + SPAN_LENGTH // 2
        ]
        question = tokenizer(QUESTIONS[0], add_special_tokens=False)["input_ids"]

        def with_question(state_ids):
            ids = state_ids + question
            return prompt(
                ids, chunked(len(state_ids), SPAN_LENGTH) + [(len(state_ids), len(ids))]
            )

        llm.reset_prefix_cache()
        run(llm, [with_question(state)])
        ((_, cached),) = run(llm, [with_question(state + state[-5:])])
        assert cached == num_complete, cached

    def test_other_span_bounds_share_no_blocks(self, llm, cases):
        ids, spans = cases[-1]
        llm.reset_prefix_cache()
        run(llm, [prompt(ids, spans)])
        num_state = spans[-2][1]
        other = chunked(num_state, max(SPAN_LENGTH // 2, 2)) + [spans[-1]]
        ((_, cached),) = run(llm, [prompt(ids, other)])
        assert cached == 0

    def test_mixed_batches_match_their_references(self, llm, cases, references):
        """A batch of span and causal requests matches the references as
        closely as batches of one kind do. This checks that requests of one
        kind do not change those of the other; the other tests check spans."""
        prompts = [
            prompt(ids, spans) if i % 2 else causal(ids)
            for i, (ids, spans) in enumerate(cases)
        ]
        modes = ["spans" if i % 2 else "causal" for i in range(len(cases))]
        refs = [references["text", i, mode] for i, mode in enumerate(modes)]
        llm.reset_prefix_cache()
        mixed = run(llm, prompts)
        alone: list[Any] = [None] * len(cases)
        for parity in (0, 1):
            llm.reset_prefix_cache()
            indices = range(parity, len(cases), 2)
            for i, result in zip(indices, run(llm, [prompts[i] for i in indices])):
                alone[i] = result
        check_top1_close([lp for lp, _ in mixed], refs)
        mixed_err = [distance(lp, ref) for (lp, _), ref in zip(mixed, refs)]
        alone_err = [distance(lp, ref) for (lp, _), ref in zip(alone, refs)]
        assert mean(mixed_err) < max(2 * mean(alone_err), 0.02), (mixed_err, alone_err)


class TestSpansOnFullAttentionLayers(SpanTests):
    layout = Layout()

    def test_requests_without_spans_are_refused(self, llm, cases):
        ids, _ = cases[0]
        with pytest.raises(VLLMValidationError, match="must set bidirectional_spans"):
            llm.generate([TokensPrompt(prompt_token_ids=ids)], PARAMS, use_tqdm=False)

    def test_engine_requests_without_spans_are_refused(self, llm, cases):
        # Built without InputProcessor, as the Rust frontend builds them.
        ids, _ = cases[0]
        params = SamplingParams(max_tokens=1)
        params.update_from_generation_config({}, eos_token_id=1)
        request = EngineCoreRequest(
            request_id="no-spans",
            prompt_token_ids=ids,
            mm_features=None,
            sampling_params=params,
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        try:
            llm.llm_engine.add_request("no-spans", request, params)
        except VLLMValidationError as e:
            # An in-process engine core refuses it at once.
            assert "must set bidirectional_spans" in str(e)
            return
        outputs: list[Any] = []
        while llm.llm_engine.has_unfinished_requests():
            outputs += llm.llm_engine.step()
        (output,) = outputs
        assert output.outputs[0].finish_reason == "error"
        assert not output.outputs[0].token_ids


class TestSpansWithMTP:
    layout = Layout(drafter=True)
    first_step_bound = 0.05

    def test_drafted_answers_match_the_reference(self, llm, drafter_cases, references):
        # Drafts are verified, so the greedy answers are the target's.
        params = SamplingParams(
            max_tokens=NUM_GENERATED, temperature=0.0, logprobs=TOP_K, ignore_eos=True
        )
        outputs = llm.generate(
            [prompt(ids, spans) for ids, spans in drafter_cases],
            params,
            use_tqdm=False,
        )
        tokenizer = llm.get_tokenizer()
        hf_outputs = []
        for i in range(len(drafter_cases)):
            tokens, logprobs = references["greedy", i]
            hf_outputs.append((tokens, tokenizer.decode(tokens), logprobs))
        vllm_outputs = [
            (
                list(o.outputs[0].token_ids),
                o.outputs[0].text,
                [{t: lp.logprob for t, lp in s.items()} for s in o.outputs[0].logprobs],
            )
            for o in outputs
        ]
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )
        # The first step tells span requests apart from causal ones.
        first = [
            {t: lp.logprob for t, lp in o.outputs[0].logprobs[0].items()}
            for o in outputs
        ]
        span_err = [
            distance(lp, references["first", i, "spans"]) for i, lp in enumerate(first)
        ]
        effect = [
            distance(lp, references["first", i, "causal"]) for i, lp in enumerate(first)
        ]
        assert mean(span_err) < self.first_step_bound, span_err
        assert mean(effect) > 2 * mean(span_err), (effect, span_err)
        counters = {
            m.name: m.value for m in llm.get_metrics() if isinstance(m, Counter)
        }
        assert counters["vllm:spec_decode_num_accepted_tokens"] > 0, counters


class TestSpansOnAllLayers(SpanTests):
    """Spans on every layer, which builds the sliding-window layers without a
    window, over states longer than the window."""

    layout = Layout(ALL_LAYERS)
    # On these long states span and causal requests reach errors of 0.065 and
    # 0.083, and the causal requests' error, the default yardstick, varies
    # between runs on the same prompts (0.03 to 0.08 at TP=2). The window and
    # the causal mask are at least 0.2 away.
    reference_bound = 0.1

    def test_sliding_window_layers_have_no_window(self, llm, cases, references):
        """Span and causal requests are closer to the reference without the
        window than to the one with it."""
        llm.reset_prefix_cache()
        for mode in ("spans", "causal"):
            results = run(
                llm,
                [prompt(ids, spans if mode == "spans" else []) for ids, spans in cases],
            )
            err = [
                distance(lp, references["text", i, mode])
                for i, (lp, _) in enumerate(results)
            ]
            windowed = [
                distance(lp, references["windowed", i, mode])
                for i, (lp, _) in enumerate(results)
            ]
            assert mean(windowed) > 2 * mean(err), (mode, windowed, err)


class TestSpansOnSlidingWindowLayers(TestSpansOnAllLayers):
    """Spans on the sliding-window layers only, without their window; the
    full-attention layers stay causal."""

    layout = Layout(SLIDING_LAYERS)


class TestSpansOnAllLayersWithMTP(TestSpansWithMTP):
    """The drafter's layers keep their own windows and read the target's KV
    cache, which keeps every token."""

    layout = Layout(ALL_LAYERS, drafter=True)
    # The first step's error reaches 0.06 on these states, about that of
    # causal first steps; the spans move it by at least 0.4.
    first_step_bound = 0.1


class TestServerSetSpansOverImages:
    """A server sets spans on a rendered multimodal engine input, after the
    placeholders are expanded. The checkpoint also sets Gemma 4's "vision"
    attention, so the sliding-window layers take the image range and the
    full-attention layers the spans. On E2B, which was not trained with that
    attention, the references with and without the image range are close, so
    this does not tell whether the sliding-window layers got it; the attention
    group unit tests cover that routing."""

    layout = Layout()

    @pytest.fixture(scope="class")
    def image_llm(self, references):
        # "vision" turns off chunked multimodal input, so a step must hold the
        # largest image Gemma 4 can produce (2,496 tokens).
        engine, proxy = make_llm(
            hf_overrides=image_overrides(),
            max_model_len=4096,
            max_num_batched_tokens=2560,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.5,
            limit_mm_per_prompt={"image": 1, "audio": 0},
        )
        yield proxy
        del engine
        free_gpu_memory()

    def test_spans_over_an_image_match_the_reference(self, image_llm, references):
        cases = references["images"]
        results: dict[str, list] = {"spans": [], "causal": []}
        for n, case in enumerate(cases):
            for mode, spans in (("spans", case["spans"]), ("causal", [])):
                (engine_input,) = image_llm.renderer.render_cmpl(
                    [parse_model_prompt(image_llm.model_config, case["prompt"])]
                )
                engine_input["bidirectional_spans"] = spans
                image_llm.llm_engine.add_request(f"{mode}-{n}", engine_input, PARAMS)
                outputs: list[Any] = []
                while image_llm.llm_engine.has_unfinished_requests():
                    outputs += [o for o in image_llm.llm_engine.step() if o.finished]
                (output,) = outputs
                results[mode].append((first_logprobs(output), 0))
        check_against_references(
            results["spans"],
            results["causal"],
            [case["ref_spans"] for case in cases],
            [case["ref_causal"] for case in cases],
            bound=0.05,
        )


class TestSpansOnPoolingRequests:
    layout = Layout()

    def test_pooled_states_match_the_reference(self, cases, references):
        engine, llm = make_llm(
            runner="pooling",
            convert="embed",
            hf_overrides=span_overrides(Layout(), SPAN_LENGTH),
            max_model_len=MAX_STATE_TOKENS + 512,
            max_num_batched_tokens=max(128, SPAN_LENGTH),
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.5,
            limit_mm_per_prompt={"image": 0, "audio": 0},
        )
        try:
            outputs = {
                mode: llm.embed(
                    [
                        prompt(ids, spans if mode == "spans" else [])
                        for ids, spans in cases
                    ],
                    pooling_params=PoolingParams(),
                    use_tqdm=False,
                )
                for mode in ("spans", "causal")
            }
        finally:
            del llm, engine
            free_gpu_memory()

        def error(output, ref: torch.Tensor) -> float:
            embedding = torch.nn.functional.normalize(
                torch.tensor(output.outputs.embedding), dim=-1
            )
            return 1 - torch.dot(embedding, ref).item()

        err = {
            mode: [
                error(o, references["pooled", i, mode])
                for i, o in enumerate(outputs[mode])
            ]
            for mode in ("spans", "causal")
        }
        refs = [references["pooled", i, "spans"] for i in range(len(cases))]
        effect = [
            1 - torch.dot(ref, references["pooled", i, "causal"]).item()
            for i, ref in enumerate(refs)
        ]
        # The error is about that of causal requests and a small fraction of
        # what the spans change. The spans move a pooled state by as little as
        # 0.009 (1 - cos), where numerics alone move it by up to 0.001.
        assert mean(err["spans"]) < 2 * mean(err["causal"]), err
        assert mean(err["spans"]) < 0.1 * mean(effect), (err["spans"], effect)
        # Each span request is closer to its span reference than to the causal
        # one.
        to_causal = [
            error(o, references["pooled", i, "causal"])
            for i, o in enumerate(outputs["spans"])
        ]
        assert all(c > 2 * e for c, e in zip(to_causal, err["spans"])), (
            to_causal,
            err["spans"],
        )


@pytest.mark.parametrize(
    ("configs", "setting"),
    [
        # A captured prefill step would replay without the spans.
        ({"compilation_config": {"cudagraph_mode": "FULL"}}, "cudagraph_mode=FULL"),
        (
            # Decode context parallelism needs more TP ranks than KV heads.
            {"parallel_config": {"decode_context_parallel_size": 2}},
            "context parallelism",
        ),
        (
            {"parallel_config": {"prefill_context_parallel_size": 2}},
            "context parallelism",
        ),
    ],
    ids=["full-cuda-graphs", "dcp", "pcp"],
)
def test_settings_that_cannot_apply_spans_are_refused(configs, setting):
    model_config = ModelConfig(MODEL, hf_overrides=span_overrides(Layout(), 64))
    built: dict[str, Any] = {}
    if "compilation_config" in configs:
        built["compilation_config"] = CompilationConfig(**configs["compilation_config"])
    if "parallel_config" in configs:
        parallel = dict(configs["parallel_config"])
        if "decode_context_parallel_size" in parallel:
            parallel["tensor_parallel_size"] = 2 * model_config.get_total_num_kv_heads()
        # The multiprocessing backend skips the check for enough local GPUs.
        built["parallel_config"] = ParallelConfig(
            distributed_executor_backend="mp", **parallel
        )
    match = "declares bidirectional attention spans, which are not supported with "
    with pytest.raises(ValueError, match=match + re.escape(setting)):
        VllmConfig(model_config=model_config, **built)
