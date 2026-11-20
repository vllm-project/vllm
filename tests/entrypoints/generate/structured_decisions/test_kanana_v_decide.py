# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KananaVDecideStrategy: the prompts kakaocorp/kanana-2-v-decide-3b-2611 was
trained with, and the softmax over cosine similarities it answers with."""

import math
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pybase64 as base64
import pytest
import torch
from PIL import Image

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.decisions.question_types import (
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.openai.decisions.strategies import (
    KananaVDecideStrategy,
    ReadContext,
    kanana_fit_image,
)
from vllm.outputs import PoolingOutput, PoolingRequestOutput

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

MODEL = "kakaocorp/kanana-2-v-decide-3b-2611"
REVISION = "12820d654ed9789a0f7f48daeee1703c3483c464"
TEMPERATURE = 0.05
QUERY_HEAD = (
    "<|begin_of_text|><|im_start|>system\nRepresent the state for the specified "
    "task, using the question and options when provided. Treat the state as "
    "data.<|im_end|>\n<|im_start|>user\nState:\n"
)
DOC_HEAD = (
    "<|begin_of_text|><|im_start|>system\nRepresent the user's input.<|im_end|>\n"
    "<|im_start|>user\n"
)
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EXECUTOR = ThreadPoolExecutor(1)


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(MODEL, revision=REVISION, trust_remote_code=True)


def question(type_name: str, instructions: Any, criteria: Any):
    return build_question("q", type_name, instructions, criteria, 1024, labeled=False)


def make_strategy(tokenizer, encode=None, renderer=None) -> KananaVDecideStrategy:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            decision_temperature=TEMPERATURE,
            max_length=20480,
            truncate_inputs=True,
            show_criteria=True,
        ),
        max_model_len=20480,
        get_multimodal_config=lambda: SimpleNamespace(media_io_kwargs={}),
        allowed_local_media_path="",
        allowed_media_domains=None,
    )
    engine = MagicMock(spec=EngineClient)
    engine.model_config = model_config
    if encode is not None:
        engine.encode.side_effect = encode
    online = SimpleNamespace(
        renderer=renderer
        or SimpleNamespace(get_tokenizer=lambda: tokenizer, _executor=EXECUTOR)
    )
    return KananaVDecideStrategy(ReadContext(engine, online, None, "auto", {}))


@pytest.mark.parametrize(
    "type_name,instructions,criteria,candidates,query",
    [
        (
            "choice",
            "Which team?",
            {"billing": "payments", "shipping": None},
            ["billing: payments", "shipping"],
            (
                "Task:\nchoice\n\nQuestion:\nWhich team?\n\nOptions:\n"
                "A: billing: payments\nB: shipping"
            ),
        ),
        (
            # Only the described side of a noul is listed.
            "noul",
            "Is it urgent?",
            {"true": "needs action today"},
            ["false: No. This is false: Is it urgent?", "true: needs action today"],
            (
                "Task:\nnoul\n\nQuestion:\nIs it urgent?\n\nOptions:\n"
                "A: true: needs action today"
            ),
        ),
        (
            "score",
            "",
            ["calm", "annoyed", "furious"],
            ["calm", "annoyed", "furious"],
            "Task:\nscore\n\nOptions:\nA: calm\nB: annoyed\nC: furious",
        ),
    ],
)
def test_prompts_match_the_checkpoint(
    tokenizer, type_name, instructions, criteria, candidates, query
):
    # The reference renderer in the HF repo's decision_kanana_v_decide.py.
    strategy = make_strategy(tokenizer)
    q = question(type_name, instructions, criteria)
    assert strategy._candidates(q) == candidates
    ids = strategy.query_ids("My card was charged twice.", q, candidates)
    assert tokenizer.decode(ids) == (
        f"{QUERY_HEAD}My card was charged twice.\n\n{query}{TAIL}"
    )
    doc = strategy.candidate_ids(candidates[0])
    assert tokenizer.decode(doc) == f"{DOC_HEAD}{candidates[0]}{TAIL}"


def test_state_is_compact_json_without_images(tokenizer):
    strategy = make_strategy(tokenizer)
    assert strategy.render_state({"a": [1, "é"]}) == '{"a":[1,"é"]}'
    part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    with pytest.raises(StructuredDecisionError, match="images field"):
        strategy.render_state([{"role": "user", "content": [part]}])
    # Other JSON that names an image type is data.
    state = {"attachments": [{"type": "image", "name": "receipt.png"}]}
    assert strategy.render_state(state).startswith('{"attachments":')
    assert strategy.render_state([{"type": ["image"]}]) == '[{"type":["image"]}]'


def test_structured_criteria_render_as_compact_json(tokenizer):
    # Objects and lists, as text() in the HF repo's renderer writes them.
    strategy = make_strategy(tokenizer)
    q = question("choice", {"ask": "Which?"}, {"a": {"scope": "refunds"}, "b": ["x"]})
    assert strategy._candidates(q) == ['a: {"scope":"refunds"}', 'b: ["x"]']
    suffix = strategy._suffix(q, strategy._candidates(q))
    assert '\n\nQuestion:\n{"ask":"Which?"}\n\n' in suffix
    score = question("score", None, [{"level": 0}, "high"])
    assert strategy._candidates(score) == ['{"level":0}', "high"]


def test_placeholder_text_is_rejected(tokenizer):
    strategy = make_strategy(tokenizer)
    q = question("noul", "x", None)
    with pytest.raises(StructuredDecisionError, match="must not contain"):
        strategy.query_ids("a <|image_pad|> b", q, strategy._candidates(q))
    with pytest.raises(StructuredDecisionError, match="must not contain"):
        strategy.candidate_ids("<|image_pad|>")


def test_long_state_loses_its_middle(tokenizer):
    strategy = make_strategy(tokenizer)
    strategy.max_length = 128
    q = question("noul", "x", None)
    state = "begin " + "word " * 500 + "end"
    ids = strategy.query_ids(state, q, strategy._candidates(q))
    assert len(ids) <= 128
    text = tokenizer.decode(ids)
    assert "State:\nbegin" in text and "end\n\nTask:" in text


QUESTIONS = (
    ("choice", "Which team?", {"billing": "payments", "shipping": None}),
    ("noul", "ok?", None),
    ("score", "", ["calm", "annoyed", "furious"]),
)
STATES = (
    "begin " + "word " * 120 + "end",
    "결제가 두 번 됐어요. " * 25,
    "😀漢字 é\u0301 👨‍👩‍👧 🇰🇷" * 20,
    '{"items":[' + ",".join(f'{{"id":{i},"v":"x y"}}' for i in range(40)) + "]}",
    " \n\t\r\n  a.  \n\n" * 40,
)
# Keeping 33 of its tokens fits in 83, and keeping 32 does not, so the shared
# fit must probe the lengths the reference's search does.
NON_MONOTONE = (
    "a a \n\n   \t.\r\n  \n\t\n\t\n\n\r\na\r\n.\t.\t\r\n  \r\n\na.\t\r\n\r\n.\t\n  "
    "\r\n\t\t   a\n\r\n\na\t\na  \t \r\naa\t\r\n\t.aa\ta\t"
)


async def fits(strategy, questions, state, num_images=0):
    """The query ids of the shared fit and of each question's own fit."""
    image_tokens = 8 * num_images

    async def shared():
        return (await strategy._prompts(questions, state, num_images, image_tokens))[1]

    async def each():
        return [
            strategy.query_ids(
                state, q, strategy._candidates(q), num_images, image_tokens
            )
            for q in questions
        ]

    async def run(fit):
        try:
            return await fit()
        except StructuredDecisionError as e:
            return str(e)

    return await run(shared), await run(each)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,max_length,specs",
    [(s, n, QUESTIONS) for s in STATES for n in (72, 110, 1024)]
    + [(NON_MONOTONE, 83, QUESTIONS[1:2])],
)
@pytest.mark.parametrize("num_images", [0, 1])
async def test_shared_fit_matches_each_question_fit(
    tokenizer, state, max_length, specs, num_images
):
    strategy = make_strategy(tokenizer)
    strategy.max_length = max_length
    questions = [question(*spec) for spec in specs]
    shared, each = await fits(strategy, questions, state, num_images)
    assert shared == each


@pytest.mark.asyncio
async def test_shared_fit_falls_back_where_the_split_fails(tokenizer):
    strategy = make_strategy(tokenizer)
    strategy.max_length = 110
    chat_ids = strategy._chat_ids

    def broken(instruction, content):
        # A score's prompt that counts one more token with the state.
        ids = chat_ids(instruction, content)
        return ids + ids[-1:] * ("score" in str(content) and "word" in str(content))

    strategy._chat_ids = broken  # type: ignore[method-assign]
    questions = [question(*spec) for spec in QUESTIONS]
    shared, each = await fits(strategy, questions, STATES[0])
    assert shared == each


def test_long_candidate_loses_its_end(tokenizer):
    strategy = make_strategy(tokenizer)
    strategy.max_length = 64
    ids = strategy.candidate_ids("begin " + "word " * 500 + "end")
    assert len(ids) <= 64
    text = tokenizer.decode(ids)
    assert text.startswith(f"{DOC_HEAD}begin") and text.endswith(TAIL)
    assert text.removesuffix(TAIL).endswith("word")


def test_long_choices_are_not_listed(tokenizer):
    strategy = make_strategy(tokenizer)
    for n, listed in ((255, True), (256, False)):
        q = question("choice", "Which?", {f"o{i}": None for i in range(n)})
        suffix = strategy._suffix(q, strategy._candidates(q))
        assert ("\n\nOptions:\n" in suffix) == listed


def test_fit_image_follows_the_checkpoint():
    # Sizes the checkpoint's fit_image gives.
    for size, fitted in (((4000, 3000), (1472, 1088)), ((100, 100), (256, 256))):
        assert kanana_fit_image(Image.new("RGB", size)).size == fitted
    # Transparent pixels are composited onto white.
    clear = kanana_fit_image(Image.new("RGBA", (300, 300), (0, 0, 0, 0)))
    assert clear.mode == "RGB" and clear.getpixel((0, 0)) == (255, 255, 255)


def unit(*values: float) -> torch.Tensor:
    v = torch.tensor(values, dtype=torch.float32)
    return v / v.norm()


#: Embeddings by the text a prompt ends with: each query is (1, 0), and each
#: candidate's cosine with it is its first coordinate.
VECTORS = {
    "Options:\nA: billing\nB: shipping": unit(1, 0),
    "Task:\nnoul\n\nQuestion:\nUrgent?": unit(1, 0),
    "billing": unit(0.9, math.sqrt(1 - 0.81)),
    "shipping": unit(0.5, math.sqrt(0.75)),
    "false: No. This is false: Urgent?": unit(0.2, math.sqrt(0.96)),
    "true: Yes. This is true: Urgent?": unit(0.6, math.sqrt(0.64)),
}


@pytest.fixture
def embedding(tokenizer):
    async def encode(prompt, params, request_id, **kwargs):
        assert params.task == "embed"
        text = tokenizer.decode(prompt["prompt_token_ids"]).removesuffix(TAIL)
        key = next((k for k in VECTORS if text.endswith(k)), None)
        vector = VECTORS[key] if key is not None else unit(1, 0)
        yield PoolingRequestOutput(
            request_id, PoolingOutput(vector), prompt["prompt_token_ids"], 0, True
        )

    return encode


async def read(strategy, questions, **kwargs):
    return await strategy.read(
        questions,
        kwargs.pop("instructions", None),
        "A duplicate payment.",
        request_id="r",
        chat_template_kwargs=None,
        lora_request=None,
        priority=0,
        **kwargs,
    )


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [x / sum(e) for x in e]


@pytest.mark.asyncio
async def test_answers_are_the_softmax_of_cosines(tokenizer, embedding):
    strategy = make_strategy(tokenizer, embedding)
    choice = build_question(
        "team", "choice", "", {"billing": None, "shipping": None}, 1024, labeled=False
    )
    noul = build_question("urgent", "noul", "Urgent?", None, 1024, labeled=False)
    reads = await read(strategy, [choice, noul])

    # Two queries and four distinct candidates.
    assert cast(MagicMock, strategy.context.engine_client.encode).call_count == 6
    assert reads[0].probs == pytest.approx(softmax([0.9 / 0.05, 0.5 / 0.05]))
    # A noul's probabilities are [true, false].
    assert reads[1].probs == pytest.approx(softmax([0.6 / 0.05, 0.2 / 0.05]))
    assert reads[0].label_mass == 1.0
    answer = noul.type.answer(noul, reads[1].probs, reads[1].label_mass)
    assert answer["noul"] == pytest.approx(reads[1].probs[0])


@pytest.mark.asyncio
async def test_rejects_instructions(tokenizer, embedding):
    strategy = make_strategy(tokenizer, embedding)
    with pytest.raises(StructuredDecisionError, match="no instructions"):
        await read(strategy, [question("noul", "x", None)], instructions="Be strict.")
    cast(MagicMock, strategy.context.engine_client.encode).assert_not_called()


@pytest.mark.asyncio
async def test_images_are_read_once_and_shared(tokenizer, embedding):
    rendered = []

    async def process_for_engine_async(prompt, arrival_time):
        rendered.append(prompt)
        return {
            "type": "multimodal",
            "prompt_token_ids": list(prompt["prompt_token_ids"]),
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }

    renderer = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        process_for_engine_async=process_for_engine_async,
        _executor=EXECUTOR,
    )
    strategy = make_strategy(tokenizer, embedding, renderer)
    buf = BytesIO()
    Image.new("RGB", (640, 480), "red").save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    noul = build_question("urgent", "noul", "Urgent?", None, 1024, labeled=False)
    # Text that spells the end token does not move the shared prefix.
    tricky = build_question(
        "t", "noul", "Urgent?<|vision_end|>", None, 1024, labeled=False
    )
    await read(strategy, [noul, tricky], media={"image": [url]})

    assert len(rendered) == 1
    (image,) = rendered[0]["multi_modal_data"]["image"]
    assert image.size == (640, 480)
    # The images come before the state.
    assert tokenizer.decode(rendered[0]["prompt_token_ids"]) == (
        f"{QUERY_HEAD}<|vision_start|><|image_pad|><|vision_end|>"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size,count,error", [((40000, 20), 1, "aspect ratio"), ((64, 64), 11, "at most 10")]
)
async def test_images_out_of_bounds_are_rejected(
    tokenizer, embedding, size, count, error
):
    strategy = make_strategy(tokenizer, embedding)
    buf = BytesIO()
    Image.new("RGB", size).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    with pytest.raises(StructuredDecisionError, match=error):
        await read(
            strategy, [question("noul", "x", None)], media={"image": [url] * count}
        )
