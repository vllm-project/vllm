# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

from fastapi import Request

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger

from .adapters import input_text, make_answer, make_read_question
from .protocol import (
    DecisionRequest,
    DecisionResponse,
    InputTokensDetails,
    OpenAIDecisionUsage,
    OutputTokensDetails,
)
from .question_types import StructuredDecisionError
from .strategies import ReadStrategy


class BaseServingDecisions(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels,
        strategy: ReadStrategy,
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(
            models=models,
            model_config=strategy.context.engine_client.model_config,
            request_logger=request_logger,
        )
        self.engine_client = strategy.context.engine_client
        self.strategy = strategy
        self.limits = strategy.limits()


class OpenAIServingDecisions(BaseServingDecisions):
    async def create_decisions(
        self, request: DecisionRequest, raw_request: Request | None = None
    ) -> DecisionResponse | ErrorResponse:
        if (error := await self._check_model(request)) is not None:  # type: ignore[arg-type]
            return error
        engine = self.strategy.context.engine_client
        if engine.errored:
            raise engine.dead_error
        request_id = f"decision-{self._base_request_id(raw_request)}"
        try:
            if len(request.questions) > self.limits.max_questions:
                raise StructuredDecisionError(
                    f"questions: at most {self.limits.max_questions} for this model"
                )
            questions = [
                make_read_question(i, question, self.limits.max_options)
                for i, question in enumerate(request.questions)
            ]
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
            engine.check_admission(len(questions))
            text = input_text(request.input)
            self._log_inputs(request_id, text, None, lora_request)
            reads = await self.strategy.read(
                questions,
                None,
                text,
                request_id=request_id,
                chat_template_kwargs=None,
                lora_request=lora_request,
                priority=0,
            )
        except StructuredDecisionError as exc:
            return self.create_error_response(exc)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        input_tokens = sum(read.input_tokens for read in reads)
        output_tokens = sum(read.output_tokens for read in reads)
        return DecisionResponse(
            model=self.models.model_name(lora_request),
            answers=[
                make_answer(question, read.probs, read.label_mass, read.confidence)
                for question, read in zip(request.questions, reads)
            ],
            usage=OpenAIDecisionUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=sum(read.cached_tokens for read in reads),
                    cache_write_tokens=sum(read.cache_write_tokens for read in reads),
                ),
                output_tokens_details=OutputTokensDetails(),
            ),
        )
