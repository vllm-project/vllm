# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gradient computation API — POST /v1/gradients.

Computes gradients of loss / per-token log-probs w.r.t. model embeddings,
enabling token-level attribution and interpretability analysis.

Architecture:
  1. FastAPI router receives GradientRequest (prompt + target text).
  2. ServingGradient tokenizes both, builds GradientParams.
  3. AsyncLLM.compute_gradients() sends an EngineCoreRequest.
  4. EngineCore bypasses the scheduler and dispatches via collective_rpc
     to Worker.compute_gradients() (needs torch.enable_grad()).
  5. GradientRunner runs forward + backward, returns gradients.
  6. Results flow back through the output processor as GradientRequestOutput.
"""

from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object
    SupportedTask = object


def register_gradient_api_router(app: FastAPI):
    from vllm.entrypoints.gradient.api_router import router as gradient_router

    app.include_router(gradient_router)


def init_gradient_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    from vllm.entrypoints.gradient.serving import ServingGradient

    state.serving_gradient = (
        ServingGradient(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if "gradient" in supported_tasks
        else None
    )
