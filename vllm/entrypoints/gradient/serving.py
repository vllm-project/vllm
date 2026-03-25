# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gradient computation serving handler.

Translates HTTP GradientRequest (prompt + target text) into
engine-level GradientParams (prompt_token_ids + target_token_ids)
and formats the GradientOutput back into GradientResponse JSON.
"""

from typing import TYPE_CHECKING

from fastapi import Request

from vllm.entrypoints.gradient.protocol import (
    GradientRequest,
    GradientResponse,
    GradientTokenLogProb,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.gradient_params import GradientParams
from vllm.logger import init_logger
from vllm.utils import random_uuid

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels

logger = init_logger(__name__)


class ServingGradient:
    """Handles gradient computation requests."""

    def __init__(
        self,
        engine_client: "EngineClient",
        model_serving: "OpenAIServingModels",
        request_logger: "RequestLogger | None" = None,
    ):
        self.engine_client = engine_client
        self.model_serving = model_serving
        self.request_logger = request_logger
        self.tokenizer = engine_client.renderer.get_tokenizer()

    async def __call__(
        self,
        request: GradientRequest,
        raw_request: Request,
    ) -> GradientResponse | ErrorResponse:
        """Handle a gradient computation request."""

        request_id = f"grad-{random_uuid()}"

        # Tokenize the target to get target_token_ids.
        target_token_ids = self.tokenizer.encode(
            request.target, add_special_tokens=False
        )
        if not target_token_ids:
            return ErrorResponse(
                message="Target text produced no tokens after tokenization.",
                type="invalid_request_error",
                code=400,
            )

        # Build GradientParams.
        gradient_params = GradientParams(
            target_token_ids=target_token_ids,
            gradient_of=request.gradient_of,
            gradient_targets=request.gradient_targets,
            target_token_indices=request.target_token_indices,
            aggregation=request.aggregation,
            loss_function=request.loss_function,
            return_log_probs=request.return_log_probs,
        )

        # Tokenize prompt.
        prompt_token_ids = self.tokenizer.encode(
            request.prompt, add_special_tokens=True
        )
        if not prompt_token_ids:
            return ErrorResponse(
                message="Prompt text produced no tokens after tokenization.",
                type="invalid_request_error",
                code=400,
            )

        # Use TokensPrompt format (a PromptType) for the engine.
        prompt = {"prompt_token_ids": prompt_token_ids}

        # Call the engine.
        result = None
        async for output in self.engine_client.compute_gradients(
            prompt=prompt,
            gradient_params=gradient_params,
            request_id=request_id,
        ):
            result = output

        if result is None:
            return ErrorResponse(
                message="Gradient computation produced no output.",
                type="server_error",
                code=500,
            )

        # Build response.
        gradient_output = result.outputs

        # Format token log-probs with decoded tokens.
        token_log_probs_response = None
        if gradient_output.token_log_probs is not None:
            token_log_probs_response = []
            for i, lp in enumerate(gradient_output.token_log_probs):
                tid = target_token_ids[i]
                token_str = self.tokenizer.decode([tid])
                token_log_probs_response.append(
                    GradientTokenLogProb(
                        token=token_str,
                        token_id=tid,
                        log_prob=lp,
                    )
                )

        # Format token attributions.
        token_attributions_response = None
        if gradient_output.token_attributions is not None:
            token_attributions_response = {
                "type": request.aggregation,
                "shape": [
                    len(gradient_output.token_attributions),
                    len(gradient_output.token_attributions[0])
                    if gradient_output.token_attributions
                    else 0,
                ],
                "data": gradient_output.token_attributions,
            }

        # Format loss gradients.
        loss_gradients_response = None
        if gradient_output.loss_gradients is not None:
            loss_gradients_response = {}
            for name, grad_data in gradient_output.loss_gradients.items():
                loss_gradients_response[name] = {
                    "type": request.aggregation,
                    "shape": [
                        len(grad_data),
                        len(grad_data[0]) if grad_data else 0,
                    ],
                    "data": grad_data,
                }

        return GradientResponse(
            id=request_id,
            model=request.model,
            token_log_probs=token_log_probs_response,
            token_attributions=token_attributions_response,
            loss=gradient_output.loss,
            loss_gradients=loss_gradients_response,
        )
