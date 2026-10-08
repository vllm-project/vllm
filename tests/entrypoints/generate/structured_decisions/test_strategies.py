# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import math
from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.datastructures import State

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.generate.structured_decisions.api_router import (
    register_structured_decisions_api_router,
)
from vllm.entrypoints.generate.structured_decisions.protocol import (
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from vllm.entrypoints.generate.structured_decisions.serving import (
    ServingStructuredDecisions,
    state_text,
)
from vllm.entrypoints.openai.decisions.api_router import register_decisions_api_router
from vllm.entrypoints.openai.decisions.question_types import (
    LABELS,
    StructuredDecisionError,
)
from vllm.entrypoints.openai.decisions.serving import OpenAIServingDecisions
from vllm.entrypoints.openai.decisions.state import init_decisions_state
from vllm.entrypoints.openai.decisions.strategies import (
    DiffusionGemmaCanvasStrategy,
    NextTokenStrategy,
    ReadContext,
    reply_label_ids,
    select_read_strategy,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.exception_handling.register import init_exception_handler
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.inputs import tokens_input
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer

TRACE_HEADERS = {
    "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
    "tracestate": "vendor=value",
}

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def model(architecture: str, logprobs_mode: str = "raw_logprobs") -> Any:
    return SimpleNamespace(architecture=architecture, logprobs_mode=logprobs_mode)


def test_strategy_selection():
    qwen = "Qwen3ForCausalLM"
    assert select_read_strategy(model(qwen)) is NextTokenStrategy
    assert select_read_strategy(model(qwen, "processed_logprobs")) is NextTokenStrategy
    assert (
        select_read_strategy(model("DiffusionGemmaForBlockDiffusion"))
        is DiffusionGemmaCanvasStrategy
    )
    with pytest.raises(ValueError, match="does not support LlamaForCausalLM"):
        select_read_strategy(model("LlamaForCausalLM"))
    with pytest.raises(ValueError, match="not raw_logits"):
        select_read_strategy(model(qwen, "raw_logits"))


def test_route_is_registered_by_default():
    app = FastAPI()
    register_structured_decisions_api_router(app)
    register_decisions_api_router(app)
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/v1/systemone" in paths
    assert "/v1/decisions" in paths


def test_unsupported_model_returns_501():
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.serving_structured_decisions = None
    init_exception_handler(app)
    register_structured_decisions_api_router(app)

    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "unsupported",
                "state": "x",
                "questions": {"answer": {"type": "choice", "criteria": {"yes": None}}},
            },
        )
    assert response.status_code == 501


def test_decisions_route_returns_501_without_supported_strategy():
    app = FastAPI()
    app.state.args = SimpleNamespace(log_error_stack=False)
    app.state.openai_serving_decisions = None
    init_exception_handler(app)
    register_decisions_api_router(app)

    with TestClient(app) as client:
        response = client.post("/v1/decisions", json=decision_body("decisions"))
    assert response.status_code == 501


def test_structured_state_keeps_unicode_in_prompt():
    state = {"message": "Français 日本語"}
    assert state_text(state) == '{"message": "Français 日本語"}'


@pytest.fixture(scope="module")
def qwen():
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer("Qwen/Qwen3-0.6B")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "state"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    return tokenizer, prompt_ids


def test_decisions_state_initializes(qwen):
    tokenizer, _ = qwen
    engine = SimpleNamespace(
        model_config=model("Qwen3ForCausalLM"), renderer=None, input_processor=None
    )
    state = State()
    state.online_renderer = SimpleNamespace(
        renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)
    )
    state.openai_serving_models = OpenAIServingModels(
        engine, [BaseModelPath(name="test-model", model_path="test-model")]
    )
    args = Namespace(chat_template_content_format="auto")

    strategy = init_decisions_state(engine, state, args, None, None, {})

    assert isinstance(strategy, NextTokenStrategy)
    assert isinstance(state.openai_serving_decisions, OpenAIServingDecisions)


def test_unsupported_decisions_state_does_not_block_startup():
    engine = SimpleNamespace(model_config=model("Qwen3ForCausalLM", "raw_logits"))
    state = State()

    strategy = init_decisions_state(
        engine, state, Namespace(chat_template_content_format="auto"), None, None, {}
    )

    assert strategy is None
    assert state.openai_serving_decisions is None


def test_labels_start_the_reply(qwen):
    tokenizer, prompt_ids = qwen
    tail, ids = reply_label_ids(tokenizer, prompt_ids)
    assert tokenizer.decode(tail) == "\n\n"
    assert [tokenizer.decode([i]) for i in ids] == list(LABELS)
    # A noul's and a score's labels are one token here too.
    _, ids = reply_label_ids(tokenizer, prompt_ids, ("yes", "no", *"0123456789"))
    assert len(set(ids)) == 12
    # After a colon, Qwen writes ":A" as one token, so "A" is not one token.
    with pytest.raises(ValueError, match="not one distinct token"):
        reply_label_ids(tokenizer, tokenizer.encode("team:"))


@pytest.fixture
def serving_and_engine(qwen):
    tokenizer, _ = qwen
    model_config = ModelConfig(model="Qwen/Qwen3-0.6B", dtype="float32")
    renderer = HfRenderer(
        SimpleNamespace(
            model_config=model_config,
            parallel_config=SimpleNamespace(_api_process_rank=0),
        ),
        tokenizer,
    )
    engine = MagicMock(spec=EngineClient)
    engine.errored = False
    engine.model_config = model_config
    engine.renderer = renderer
    engine.input_processor = MagicMock()
    online_renderer = OnlineRenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    async def generate(prompt, params, request_id, **kwargs):
        label_ids = params.logprob_token_ids
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt["prompt_token_ids"],
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="A",
                    token_ids=[label_ids[0]],
                    cumulative_logprob=None,
                    logprobs=[{token_id: Logprob(-1.0) for token_id in label_ids}],
                    finish_reason="length",
                )
            ],
            finished=True,
        )

    engine.generate.side_effect = generate
    strategy = NextTokenStrategy(ReadContext(engine, online_renderer, None, "auto", {}))
    models = OpenAIServingModels(
        engine, [BaseModelPath(name=model_config.model, model_path=model_config.model)]
    )
    yield ServingStructuredDecisions(models, strategy), engine
    renderer.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_salt,tracing_enabled,request_headers",
    [
        ("tenant-a", True, {"authorization": "Bearer test", **TRACE_HEADERS}),
        ("tenant-b", False, {"authorization": "Bearer test", **TRACE_HEADERS}),
        (None, True, {"authorization": "Bearer test", **TRACE_HEADERS}),
        (None, True, {}),
        ("tenant-direct", True, None),
    ],
)
async def test_request_metadata_reaches_every_question(
    serving_and_engine, cache_salt, tracing_enabled, request_headers
):
    serving, engine = serving_and_engine
    engine.is_tracing_enabled.return_value = tracing_enabled
    raw_request = None
    if request_headers is not None:
        raw_request = Request(
            {
                "type": "http",
                "headers": [
                    (k.encode(), v.encode()) for k, v in request_headers.items()
                ],
            }
        )
    request = StructuredDecisionRequest(
        state={"team": "billing", "language": "English"},
        questions={
            "team": {
                "type": "choice",
                "instructions": "Which team does the message name?",
                "criteria": {"billing": "payments", "shipping": "deliveries"},
            },
            "lang": {
                "type": "choice",
                "instructions": "Which language does the message name?",
                "criteria": {"English": "English text", "French": "French text"},
            },
        },
        cache_salt=cache_salt,
    )
    response = await serving.create_decision(request, raw_request)
    assert isinstance(response, StructuredDecisionResponse)
    engine.is_tracing_enabled.assert_not_awaited()
    assert engine.generate.call_count == 2
    for call in engine.generate.call_args_list:
        assert call.args[0].get("cache_salt") == cache_salt
        assert call.kwargs["trace_headers"] == (
            TRACE_HEADERS if request_headers else None
        )


def test_canvas_read():
    strategy = DiffusionGemmaCanvasStrategy.__new__(DiffusionGemmaCanvasStrategy)
    strategy.thought, strategy.end, strategy.pad = [10, 11, 12, 13], 106, 0
    strategy.width, strategy.vocab_size = 16, 1000
    strategy.max_model_len = 23
    params = strategy._sampling_params([65, 66], prompt_ids=[1, 2, 3])
    assert params.extra_args is not None
    canvas = params.extra_args["diffusion_seed_canvas"]
    # The label slot as noise, the end of the turn, padding.
    assert canvas[1:] == [106] + [0] * 14
    assert params.max_tokens == 2 and params.logprob_token_ids == [65, 66]
    again = strategy._sampling_params([65, 66], prompt_ids=[1, 2, 3])
    assert again.extra_args == params.extra_args
    read_input = strategy._read_input(
        {"type": "token", "prompt_token_ids": [1, 2, 3]}, [1, 2, 3]
    )
    assert read_input == {
        "type": "token",
        "prompt_token_ids": [1, 2, 3, 10, 11, 12, 13],
    }
    # One token more and the thought and the canvas no longer fit.
    with pytest.raises(StructuredDecisionError, match="max_model_len=23"):
        strategy._read_input(
            {"type": "token", "prompt_token_ids": [1, 2, 3, 4]}, [1, 2, 3, 4]
        )


@pytest.fixture
def decision_server(qwen):
    tokenizer, _ = qwen

    class Engine:
        model_config = SimpleNamespace(is_encoder_decoder=False)
        renderer = input_processor = None
        errored = False
        failure = None

        def __init__(self):
            self.admitted = []
            self.generated = []

        def check_admission(self, count):
            self.admitted.append(count)

        async def generate(self, prompt, params, request_id, **kwargs):
            self.generated.append(prompt)
            if self.failure == "bad_request":
                raise ValueError("Invalid generation input")
            if self.failure == "no_result":
                return
            ids = params.logprob_token_ids
            logprobs = {i: Logprob(-math.log(len(ids))) for i in ids}
            if self.failure == "missing_label":
                logprobs.pop(ids[-1])
            output = CompletionOutput(
                index=0,
                text="A",
                token_ids=[ids[0]],
                cumulative_logprob=None,
                logprobs=[logprobs],
                finish_reason="error" if self.failure == "error" else "length",
            )
            if self.failure == "no_logprobs":
                output.logprobs = None
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=prompt["prompt_token_ids"],
                prompt_logprobs=None,
                outputs=[] if self.failure == "no_output" else [output],
                finished=True,
            )

    class Renderer:
        renderer = SimpleNamespace(get_tokenizer=lambda: tokenizer)

        async def preprocess_chat(self, request, messages, **kwargs):
            await asyncio.sleep(0)
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=False,
            )
            return [], [tokens_input(ids)]

    engine, renderer = Engine(), Renderer()
    strategy = NextTokenStrategy(ReadContext(engine, renderer, None, "auto", {}))
    models = OpenAIServingModels(
        engine, [BaseModelPath(name="test-model", model_path="test-model")]
    )
    request_logger = RequestLogger(max_log_len=None)
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=False)
    app.state.serving_structured_decisions = ServingStructuredDecisions(
        models, strategy, request_logger=request_logger
    )
    app.state.openai_serving_decisions = OpenAIServingDecisions(
        models, strategy, request_logger=request_logger
    )
    init_exception_handler(app)
    register_structured_decisions_api_router(app)
    register_decisions_api_router(app)
    with TestClient(app, headers={"X-Request-Id": "test-request"}) as client:
        yield SimpleNamespace(
            client=client, engine=engine, renderer=renderer, tokenizer=tokenizer
        )


def decision_body(endpoint, count=1):
    questions = [
        {"type": "choice", "instructions": f"Question {i}"} for i in range(count)
    ]
    if endpoint == "decisions":
        return dict(
            model="test-model",
            input="evidence",
            safety_identifier="test-user",
            questions=[
                dict(q, choices=[{"value": "yes"}, {"value": "no"}]) for q in questions
            ],
        )
    return dict(
        model="test-model",
        state="evidence",
        questions={
            str(i): dict(q, criteria={"yes": None, "no": None})
            for i, q in enumerate(questions)
        },
    )


@pytest.mark.parametrize("endpoint", ["decisions", "systemone"])
@pytest.mark.parametrize("count,status", [(64, 200), (65, 400)])
def test_model_question_limit_is_enforced_before_admission(
    decision_server, endpoint, count, status
):
    server = decision_server
    response = server.client.post(
        f"/v1/{endpoint}", json=decision_body(endpoint, count)
    )
    assert response.status_code == status, response.text
    assert server.engine.admitted == ([count] if status == 200 else [])
    if status == 400:
        assert "at most 64" in response.json()["error"]["message"]
        assert not server.engine.generated


@pytest.mark.parametrize("endpoint", ["decisions", "systemone"])
@pytest.mark.parametrize(
    "failure",
    ["no_result", "no_output", "error", "no_logprobs", "missing_label", "bad_request"],
)
def test_generation_failures_are_server_errors(decision_server, endpoint, failure):
    server = decision_server
    server.engine.failure = failure
    response = server.client.post(f"/v1/{endpoint}", json=decision_body(endpoint))
    status = 400 if failure == "bad_request" else 500
    assert response.status_code == status, response.text
    assert response.json()["error"]["type"] == (
        "BadRequestError" if status == 400 else "InternalServerError"
    )
    assert len(server.engine.generated) == 1


def test_decisions_logging_includes_receipt_and_body(
    decision_server, caplog, monkeypatch
):
    logger_name = "vllm.entrypoints.serve.utils.request_logger"
    monkeypatch.setattr(logging.getLogger(logger_name), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        response = decision_server.client.post(
            "/v1/decisions", json=decision_body("decisions")
        )
    assert response.status_code == 200, response.text
    assert "Received request decision-test-request" in caplog.text
    assert "evidence" in caplog.text
    assert '"safety_identifier":"test-user"' in caplog.text


def test_prompts_retain_question_order(decision_server):
    server = decision_server
    response = server.client.post("/v1/decisions", json=decision_body("decisions", 2))
    assert response.status_code == 200, response.text
    for i, prompt in enumerate(server.engine.generated):
        assert f"Question {i}" in server.tokenizer.decode(prompt["prompt_token_ids"])


def test_winnow_requires_explicit_protocol_and_raw_scores():
    from vllm.entrypoints.openai.decisions.winnow import WinnowStrategy

    config = model("Gemma4ForCausalLM")
    with pytest.raises(ValueError, match="does not support"):
        select_read_strategy(config)
    config.hf_config = SimpleNamespace(decision_read_strategy="winnow")
    assert select_read_strategy(config) is WinnowStrategy
    config.logprobs_mode = "processed_logprobs"
    with pytest.raises(ValueError, match="raw_logprobs"):
        select_read_strategy(config)


def test_winnow_option_strings_preserve_trained_semantics():
    from vllm.entrypoints.openai.decisions.adapters import make_read_question
    from vllm.entrypoints.openai.decisions.protocol import (
        ChoiceQuestion,
        PredicateQuestion,
        ScoreQuestion,
    )

    choice = ChoiceQuestion(
        type="choice",
        instructions="Route?",
        choices=[
            {"value": "billing"},
            {"value": "support", "description": "Help"},
        ],
    )
    assert make_read_question(0, choice).winnow_options == (
        "billing",
        "support: Help",
    )
    predicate = PredicateQuestion(type="predicate", instructions="Valid?")
    assert make_read_question(1, predicate).winnow_options == ("false", "true")
    score = ScoreQuestion(
        type="score",
        instructions="Rate",
        levels=[
            {"label": "bad"},
            {"label": "good"},
        ],
    )
    assert make_read_question(2, score).winnow_options == ("bad", "good")


def test_winnow_route_uses_independent_fixed_prompts(decision_server, monkeypatch):
    from vllm.entrypoints.openai.decisions.winnow import WinnowStrategy

    server = decision_server
    monkeypatch.setattr(server.tokenizer, "bos_token_id", 0)
    server.engine.model_config.max_model_len = 8192
    server.engine.model_config.hf_config = SimpleNamespace(decision_temperature=1.0)
    strategy = WinnowStrategy(
        ReadContext(server.engine, server.renderer, None, "auto", {})
    )
    server.client.app.state.openai_serving_decisions.strategy = strategy
    body = {
        "model": "test-model",
        "input": "Evidence",
        "questions": [
            {"type": "predicate", "instructions": "Is it valid?", "name": "valid"},
            {
                "type": "choice",
                "instructions": "Route?",
                "name": "route",
                "choices": [{"value": "a"}, {"value": "b"}],
            },
            {
                "type": "score",
                "instructions": "Rate?",
                "name": "rating",
                "levels": [{"label": "low"}, {"label": "high"}],
            },
        ],
    }
    response = server.client.post("/v1/decisions", json=body)
    assert response.status_code == 200, response.text
    assert [answer["name"] for answer in response.json()["answers"]] == [
        "valid",
        "route",
        "rating",
    ]
    assert len(server.engine.generated) == 3
    texts = [
        server.tokenizer.decode(p["prompt_token_ids"]) for p in server.engine.generated
    ]
    assert all(text.count("Question:") == 1 for text in texts)
    assert all('State:\n"Evidence"' in text for text in texts)


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_salt", [None, "tenant-a"])
async def test_winnow_temperature_preserves_full_vocabulary_confidence(
    decision_server, monkeypatch, cache_salt
):
    from vllm.entrypoints.openai.decisions import winnow
    from vllm.entrypoints.openai.decisions.adapters import (
        make_answer,
        make_read_question,
    )
    from vllm.entrypoints.openai.decisions.protocol import ChoiceAnswer, ChoiceQuestion

    server = decision_server
    monkeypatch.setattr(server.tokenizer, "bos_token_id", 0)
    server.engine.model_config.max_model_len = 8192
    server.engine.model_config.hf_config = SimpleNamespace(decision_temperature=2.0)
    strategy = winnow.WinnowStrategy(
        ReadContext(server.engine, server.renderer, None, "auto", {})
    )
    question = ChoiceQuestion(
        type="choice", instructions="Route?", choices=[{"value": "a"}, {"value": "b"}]
    )

    async def read_labels(engine, inputs, params, request_id, **kwargs):
        assert inputs[0].get("cache_salt") == cache_salt
        assert kwargs["trace_headers"] == TRACE_HEADERS
        return [
            SimpleNamespace(
                logprobs=[math.log(0.1), math.log(0.4)],
                result=SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=[strategy.label_ids[1]])],
                    prompt_token_ids=inputs[0]["prompt_token_ids"],
                    num_cached_tokens=0,
                    num_cache_creation_tokens=0,
                ),
            )
        ]

    monkeypatch.setattr(winnow, "next_token_label_reads", read_labels)
    (read,) = await strategy.read(
        [make_read_question(0, question)],
        None,
        "Evidence",
        request_id="test",
        chat_template_kwargs=None,
        lora_request=None,
        priority=0,
        cache_salt=cache_salt,
        trace_headers=TRACE_HEADERS,
    )
    answer = make_answer(question, read.probs, read.label_mass, read.confidence)
    assert isinstance(answer, ChoiceAnswer)
    assert answer.probabilities[1].probability == pytest.approx(2 / 3)
    assert answer.confidence == pytest.approx(0.4)


def test_image_decisions_render_each_question_with_ordered_shared_media(
    decision_server, monkeypatch
):
    """The real API preserves media up to the existing multimodal renderer."""
    from copy import deepcopy

    server = decision_server
    seen = []
    original = server.renderer.preprocess_chat
    server.engine.model_config.is_multimodal_model = True

    async def render(request, messages, **kwargs):
        seen.append(deepcopy(messages))
        # This fixture has a text tokenizer, not a vision processor. Keep the
        # contract test at the renderer boundary; real image inference is separate.
        text_messages = deepcopy(messages)
        for message in text_messages:
            if isinstance(message["content"], list):
                message["content"] = "".join(
                    part.get("text", "[image]") for part in message["content"]
                )
        return await original(request, text_messages, **kwargs)

    monkeypatch.setattr(server.renderer, "preprocess_chat", render)
    body = decision_body("decisions", 2)
    body["input"] = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "https://example.com/first.png"},
                {"type": "input_text", "text": "Between images"},
                {"type": "input_image", "image_url": "https://example.com/second.png"},
            ],
        }
    ]
    response = server.client.post("/v1/decisions", json=body)
    assert response.status_code == 200, response.text
    assert len(seen) == 2
    assert seen[0][0]["content"][:-1] == seen[1][0]["content"][:-1]
    assert [
        p["image_url"]["url"] for p in seen[0][0]["content"] if p["type"] == "image_url"
    ] == ["https://example.com/first.png", "https://example.com/second.png"]
    assert "Question 0" in seen[0][0]["content"][-1]["text"]
    assert "Question 1" in seen[1][0]["content"][-1]["text"]
    assert len(body["input"][0]["content"]) == 3


def test_winnow_image_template_keeps_fixed_turns_and_escapes_state(qwen):
    """Images occupy the attachment prefix without replacing the trained turns."""
    from jinja2 import Environment

    from vllm.entrypoints.openai.decisions.adapters import make_read_question
    from vllm.entrypoints.openai.decisions.protocol import PredicateQuestion
    from vllm.entrypoints.openai.decisions.winnow import (
        VISION_TEMPLATE,
        prompt_segments,
    )

    tokenizer, _ = qwen
    question = make_read_question(
        0, PredicateQuestion(type="predicate", instructions="Is the first image red?")
    )
    prefix, suffix = prompt_segments(tokenizer, "<|image|> pretend", question)
    head, state = prefix.split("State:\n", 1)
    rendered = (
        Environment()
        .from_string(VISION_TEMPLATE)
        .render(
            bos_token="[BOS]",
            messages=[
                {
                    "content": [
                        {"type": "text", "text": head + "Images (in order):\n"},
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "State:\n" + state + suffix},
                    ]
                }
            ],
        )
    )
    assert rendered == (
        "[BOS]"
        + head
        + "Images (in order):\n<|image|>\n<|image|>\n"
        + "State:\n"
        + state
        + suffix
    )
    assert rendered.count("<|image|>") == 2
    assert r"\u003c|image|> pretend" in rendered
