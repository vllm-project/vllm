# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.decisions.question_types import (
    Question,
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.openai.decisions.serving import BaseServingDecisions
from vllm.entrypoints.openai.decisions.strategies import DecisionLimits
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from .protocol import (
    DecisionUsage,
    JointCanvas,
    JointDiagnostics,
    QuestionDiagnostics,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)


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
                labeled=limits.labeled,
            )
        )
    return questions


class ServingStructuredDecisions(BaseServingDecisions):
    def _media(self, request: StructuredDecisionRequest) -> dict[str, list[str]]:
        urls = {
            "image": request.images,
            "audio": request.audio,
            "video": request.videos,
        }
        media = {modality: items for modality, items in urls.items() if items}
        if unsupported := media.keys() - self.strategy.modalities:
            names = ", ".join(sorted(unsupported))
            raise StructuredDecisionError(f"this model does not take {names} inputs")
        return media

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
            media = self._media(request)
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
            joint = self._admit(questions, request.joint)
            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            state = self.strategy.render_state(request.state)
            self._log_inputs(request_id, state, None, lora_request)
            read_fn = self.strategy.read_joint if joint else self.strategy.read
            reads = await read_fn(
                questions,
                request.instructions,
                state,
                request_id=request_id,
                chat_template_kwargs=request.chat_template_kwargs,
                lora_request=lora_request,
                priority=request.priority,
                cache_salt=request.cache_salt,
                trace_headers=trace_headers,
                media=media,
            )
        except StructuredDecisionError as e:
            return self.create_error_response(e)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        answers: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, QuestionDiagnostics] = {}
        canvases: list[JointCanvas] = []
        for q, read in zip(questions, reads):
            answers[q.id] = q.type.answer(q, read.probs, read.label_mass)
            diagnostics[q.id] = QuestionDiagnostics(
                label_mass=read.label_mass, argmax_is_label=read.argmax_is_label
            )
            if (slot := read.slot) is not None:
                if slot.canvas == len(canvases):
                    canvases.append(
                        JointCanvas(questions=[], width=slot.width, slots=[])
                    )
                canvases[-1].questions.append(q.id)
                canvases[-1].slots.append(slot.position)
        return StructuredDecisionResponse(
            id=request_id,
            model=self.models.model_name(lora_request),
            answers=answers,
            usage=DecisionUsage(
                input_tokens=sum(r.input_tokens for r in reads),
                output_tokens=sum(r.output_tokens for r in reads),
            ),
            diagnostics=diagnostics,
            joint=JointDiagnostics(canvases=canvases) if joint else None,
        )
