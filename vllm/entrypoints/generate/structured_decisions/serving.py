# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from http import HTTPStatus
from typing import Any, cast

from fastapi import Request

from vllm.entrypoints.openai.decisions.question_types import (
    Question,
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.openai.decisions.serving import BaseServingDecisions
from vllm.entrypoints.openai.decisions.strategies import (
    DecisionLimits,
    NextTokenStrategy,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from .box_contract import BOX_MAX_QUESTIONS, BoxOutputError, BoxQuestion
from .box_generation import generate_boxes
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
) -> list[Question | BoxQuestion]:
    if not request.questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(request.questions) > limits.max_questions:
        raise StructuredDecisionError(
            f"questions: at most {limits.max_questions} for this model"
        )
    questions: list[Question | BoxQuestion] = []
    for qid, spec in request.questions.items():
        if spec.model_extra:
            raise StructuredDecisionError(
                f"question {qid!r}: unknown field(s) {sorted(spec.model_extra)}"
            )
        if spec.type == "box":
            questions.append(BoxQuestion.parse(qid, spec.instructions, spec.criteria))
            continue
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
    def _media_modalities(self) -> frozenset[str]:
        modalities = self.strategy.modalities
        if type(self.strategy) is NextTokenStrategy:
            renderer = self.strategy.context.online_renderer.renderer
            processor = renderer.mm_processor
            if processor is not None and processor.info.allowed_mm_limits.get(
                "image", 0
            ):
                return modalities | {"image"}
        return modalities

    def _media(
        self, request: StructuredDecisionRequest, modalities: frozenset[str]
    ) -> dict[str, list[str]]:
        urls = {
            "image": request.images,
            "audio": request.audio,
            "video": request.videos,
        }
        media = {modality: items for modality, items in urls.items() if items}
        if unsupported := media.keys() - modalities:
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
        renderer = self.strategy.context.online_renderer
        if (
            error := renderer.validate_chat_template(
                request_chat_template=None,
                chat_template_kwargs=request.chat_template_kwargs,
                trust_request_chat_template=renderer.trust_request_chat_template,
            )
        ) is not None:
            return error
        engine_client = self.strategy.context.engine_client
        if engine_client.errored:
            raise engine_client.dead_error

        base_id = self._base_request_id(raw_request, default=request.request_id)
        request_id = f"decision-{base_id}"
        try:
            parsed = parse_questions(request, self.limits)
            questions = [q for q in parsed if isinstance(q, Question)]
            boxes = [q for q in parsed if isinstance(q, BoxQuestion)]
            if boxes:
                if request.joint or len(request.images or ()) != 1:
                    raise StructuredDecisionError(
                        "box needs exactly one image and joint=false"
                    )
                if len(boxes) > BOX_MAX_QUESTIONS:
                    raise StructuredDecisionError(
                        f"at most {BOX_MAX_QUESTIONS} box questions per request"
                    )
            modalities = self._media_modalities()
            if boxes and (
                type(self.strategy) is not NextTokenStrategy
                or "image" not in modalities
            ):
                return self.create_error_response(
                    "generated boxes need an autoregressive image read strategy",
                    "NotImplementedError",
                    HTTPStatus.NOT_IMPLEMENTED,
                )
            media = self._media(request, modalities)
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
            if boxes:
                engine_client.check_admission(
                    len(boxes) + self.strategy.num_requests(questions)
                )
                joint = False
            else:
                joint = self._admit(questions, request.joint)
            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            read_fn = self.strategy.read_joint if joint else self.strategy.read
            box_reads = []
            if boxes:
                box_reads = await generate_boxes(
                    cast(NextTokenStrategy, self.strategy),
                    boxes,
                    media["image"][0],
                    request.instructions,
                    self.strategy.render_state(request.state),
                    request_id=request_id,
                    chat_template_kwargs=request.chat_template_kwargs,
                    lora_request=lora_request,
                    priority=request.priority,
                    cache_salt=request.cache_salt,
                    trace_headers=trace_headers,
                )
            reads = (
                await read_fn(
                    questions,
                    request.instructions,
                    self.strategy.render_state(request.state),
                    request_id=request_id,
                    chat_template_kwargs=request.chat_template_kwargs,
                    lora_request=lora_request,
                    priority=request.priority,
                    cache_salt=request.cache_salt,
                    trace_headers=trace_headers,
                    media=media,
                )
                if questions
                else []
            )
        except BoxOutputError as e:
            return self.create_error_response(
                e, "BadGatewayError", HTTPStatus.BAD_GATEWAY
            )
        except StructuredDecisionError as e:
            return self.create_error_response(e)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        answers: dict[str, dict[str, Any]] = {
            q.id: r.answer for q, r in zip(boxes, box_reads)
        }
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
            answers={qid: answers[qid] for qid in request.questions},
            usage=DecisionUsage(
                input_tokens=(
                    sum(r.input_tokens for r in reads)
                    + sum(r.input_tokens for r in box_reads)
                ),
                output_tokens=(
                    sum(r.output_tokens for r in reads)
                    + sum(r.output_tokens for r in box_reads)
                ),
            ),
            diagnostics=diagnostics,
            joint=JointDiagnostics(canvases=canvases) if joint else None,
        )
