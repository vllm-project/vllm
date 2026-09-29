# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured decisions on any generative model.

Each question is one read: the chat prompt with the reply prefilled up to the
question's "id:", one generated token, and the logprobs of that question's
label tokens. Every read of a request shares the system prompt and the state,
so with prefix caching the state is prefilled once.
"""

import asyncio
import math
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import ChatTemplateContentFormatOption
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import SamplingParams
from vllm.utils.async_utils import merge_async_iterators

from .prompts import answer_prefix, label_token_ids, state_text, system_text
from .protocol import (
    MAX_QUESTIONS,
    UNSUPPORTED_QUESTION_FIELDS,
    DecisionUsage,
    QuestionDiagnostics,
    ReadPromptRequest,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from .question_types import (
    Question,
    StructuredDecisionError,
    build_question,
    label_softmax,
)

logger = init_logger(__name__)


def parse_questions(request: StructuredDecisionRequest) -> list[Question]:
    if not request.questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(request.questions) > MAX_QUESTIONS:
        raise StructuredDecisionError(f"questions: at most {MAX_QUESTIONS}")
    questions = []
    for qid, spec in request.questions.items():
        extra = spec.model_extra or {}
        if unsupported := [f for f in UNSUPPORTED_QUESTION_FIELDS if f in extra]:
            raise StructuredDecisionError(
                f"question {qid!r}: {', '.join(unsupported)} not supported yet"
            )
        questions.append(
            build_question(qid, spec.type, spec.instructions, spec.criteria)
        )
    return questions


class ServingStructuredDecisions(BaseServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        online_renderer: OnlineRenderer,
        *,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(
            models=models,
            model_config=engine_client.model_config,
            request_logger=request_logger,
        )
        self.engine_client = engine_client
        self.online_renderer = online_renderer
        self.chat_template = chat_template
        self.chat_template_content_format = chat_template_content_format
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}

    async def create_decision(
        self,
        request: StructuredDecisionRequest,
        raw_request: Request | None = None,
    ) -> StructuredDecisionResponse | ErrorResponse:
        if (error := await self._check_model(request)) is not None:
            return error
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        tokenizer = self.online_renderer.renderer.get_tokenizer()
        try:
            questions = parse_questions(request)
            label_ids = [label_token_ids(tokenizer, q) for q in questions]
            lora_request = self._maybe_get_adapters(request)
        except (StructuredDecisionError, ValueError) as e:
            return self.create_error_response(e)

        self.engine_client.check_admission(len(questions))
        system = system_text(request.instructions, questions)
        state = state_text(request.state)
        read_request = ReadPromptRequest(
            chat_template_kwargs=request.chat_template_kwargs
        )

        engine_inputs = []
        for q in questions:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": state},
                {"role": "assistant", "content": answer_prefix(q)},
            ]
            try:
                _, (engine_input,) = await self.online_renderer.preprocess_chat(
                    read_request,
                    messages,
                    default_template=self.chat_template,
                    default_template_content_format=self.chat_template_content_format,
                    default_template_kwargs=self.default_chat_template_kwargs,
                )
            except (ValueError, TypeError) as e:
                return self.create_error_response(e)
            prompt_ids = self._extract_prompt_components(engine_input).token_ids
            prefix_ids = tokenizer.encode(answer_prefix(q), add_special_tokens=False)
            if prompt_ids is None or list(prompt_ids[-len(prefix_ids) :]) != prefix_ids:
                return self.create_error_response(
                    f"question {q.id!r}: the chat template did not end the prompt "
                    f"with the answer prefix {answer_prefix(q)!r}"
                )
            engine_inputs.append(engine_input)

        request_id = (
            f"decision-{self._base_request_id(raw_request, default=request.request_id)}"
        )
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, (engine_input, ids) in enumerate(zip(engine_inputs, label_ids)):
            params = SamplingParams(
                max_tokens=1, temperature=0.0, logprob_token_ids=ids
            )
            read_id = f"{request_id}-{i}"
            self._log_inputs(
                read_id, engine_input, params=params, lora_request=lora_request
            )
            generators.append(
                self.engine_client.generate(
                    engine_input,
                    params,
                    read_id,
                    lora_request=lora_request,
                    priority=request.priority,
                )
            )

        results: list[RequestOutput | None] = [None] * len(generators)
        try:
            async for i, res in merge_async_iterators(*generators):
                results[i] = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except Exception as e:
            logger.exception("Error during structured decision reads")
            return self.create_error_response(e)

        answers: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, QuestionDiagnostics] = {}
        input_tokens = output_tokens = 0
        for q, ids, result in zip(questions, label_ids, results):
            if result is None or not result.outputs or not result.outputs[0].logprobs:
                return self.create_error_response(
                    f"question {q.id!r}: the read returned no logprobs"
                )
            output = result.outputs[0]
            logprobs = output.logprobs[0]
            label_logprobs = [logprobs[t].logprob for t in ids]
            probs = label_softmax(label_logprobs)
            answers[q.id] = q.type.answer(q, probs)
            diagnostics[q.id] = QuestionDiagnostics(
                label_mass=sum(math.exp(lp) for lp in label_logprobs),
                argmax_is_label=bool(output.token_ids) and output.token_ids[0] in ids,
            )
            input_tokens += len(result.prompt_token_ids or ())
            output_tokens += len(output.token_ids)

        return StructuredDecisionResponse(
            id=request_id,
            model=self.models.model_name(lora_request),
            answers=answers,
            usage=DecisionUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            diagnostics=diagnostics,
        )
