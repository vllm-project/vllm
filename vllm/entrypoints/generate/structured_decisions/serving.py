# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.logger import init_logger

from .protocol import (
    DecisionUsage,
    QuestionDiagnostics,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)
from .question_types import Question, StructuredDecisionError, build_question
from .strategies import DecisionLimits, ReadStrategy

logger = init_logger(__name__)


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def parse_questions(
    request: StructuredDecisionRequest,
    limits: DecisionLimits,
) -> list[Question]:
    if not request.questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(request.questions) > limits.max_questions:
        raise StructuredDecisionError(
            f"questions: at most {limits.max_questions} for this model"
        )
    questions = []
    for qid, spec in request.questions.items():
        if spec.model_extra:
            raise StructuredDecisionError(
                f"question {qid!r}: unknown field(s) {sorted(spec.model_extra)}"
            )
        questions.append(
            build_question(
                qid,
                spec.type,
                spec.instructions,
                spec.criteria,
                limits.max_options,
            )
        )
    return questions


class ServingStructuredDecisions(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels,
        strategy: ReadStrategy,
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        model_config = strategy.context.engine_client.model_config
        super().__init__(
            models=models, model_config=model_config, request_logger=request_logger
        )
        self.strategy = strategy
        self.limits = strategy.limits()

    async def create_decision(
        self,
        request: StructuredDecisionRequest,
        raw_request: Request | None = None,
    ) -> StructuredDecisionResponse | ErrorResponse:
        if (error := await self._check_model(request)) is not None:  # type: ignore[arg-type]
            return error
        engine_client = self.strategy.context.engine_client
        if engine_client.errored:
            raise engine_client.dead_error

        base_id = self._base_request_id(raw_request, default=request.request_id)
        request_id = f"decision-{base_id}"
        try:
            questions = parse_questions(request, self.limits)
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
            engine_client.check_admission(len(questions))
            reads = await self.strategy.read(
                questions,
                request.instructions,
                state_text(request.state),
                request_id=request_id,
                chat_template_kwargs=request.chat_template_kwargs,
                lora_request=lora_request,
                priority=request.priority,
            )
        except StructuredDecisionError as e:
            return self.create_error_response(e)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        answers: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, QuestionDiagnostics] = {}
        for q, read in zip(questions, reads):
            answers[q.id] = q.type.answer(q, read.probs, read.label_mass)
            diagnostics[q.id] = QuestionDiagnostics(
                label_mass=read.label_mass, argmax_is_label=read.argmax_is_label
            )
        return StructuredDecisionResponse(
            id=request_id,
            model=self.models.model_name(lora_request),
            answers=answers,
            usage=DecisionUsage(
                input_tokens=sum(r.input_tokens for r in reads),
                output_tokens=sum(r.output_tokens for r in reads),
            ),
            diagnostics=diagnostics,
        )
