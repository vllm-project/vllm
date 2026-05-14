# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gradient request handler for EngineCore.

Encapsulates the queueing and execution of gradient computation requests,
which bypass the scheduler because they need torch.enable_grad() and
cannot be batched with normal inference.

Gradient requests are queued during add_request() and executed during
step() via collective_rpc to avoid blocking the engine's request
processing loop.
"""

from collections import deque

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class GradientHandler:
    """Manages gradient request queuing and execution in EngineCore.

    Gradient requests bypass the scheduler (they need torch.enable_grad())
    and are executed synchronously via collective_rpc during step().

    Usage in EngineCore:
        self._gradient_handler = GradientHandler(self.model_executor)

        # In add_request():
        if request.gradient_params is not None:
            self._gradient_handler.enqueue(request)
            return

        # In step():
        self._gradient_handler.process_queue()
        ...
        self._gradient_handler.flush_outputs(engine_core_outputs)
    """

    def __init__(self, model_executor):
        self._executor = model_executor
        self._queued: deque[Request] = deque()
        self._pending: list[tuple[int, EngineCoreOutputs]] = []

    def enqueue(self, request: Request) -> None:
        """Queue a gradient request for execution during step()."""
        self._queued.append(request)

    def process_queue(self) -> None:
        """Execute all queued gradient requests."""
        while self._queued:
            request = self._queued.popleft()
            self._handle(request)

    def flush_outputs(
        self,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        """Merge pending gradient outputs into the step's output dict."""
        for client_index, grad_outputs in self._pending:
            if client_index in engine_core_outputs:
                engine_core_outputs[client_index].outputs.extend(grad_outputs.outputs)
            else:
                engine_core_outputs[client_index] = grad_outputs
        self._pending.clear()

    @property
    def has_pending(self) -> bool:
        """Whether there are completed gradient outputs awaiting delivery."""
        return bool(self._pending)

    def _handle(self, request: Request) -> None:
        """Execute a single gradient request via collective_rpc."""
        gradient_params = request.gradient_params
        assert gradient_params is not None
        assert request.prompt_token_ids is not None, (
            "Gradient requests require prompt_token_ids"
        )

        gradient_result = None
        finish_reason = FinishReason.STOP
        try:
            # collective_rpc returns [result_per_worker]; take first.
            results = self._executor.collective_rpc(
                "compute_gradients",
                args=(request.prompt_token_ids, gradient_params),
            )
            gradient_result = results[0]
            gradient_result["target_token_ids"] = gradient_params.target_token_ids
        except Exception:
            logger.exception(
                "Gradient computation failed for request %s",
                request.request_id,
            )
            finish_reason = FinishReason.ERROR

        output = EngineCoreOutput(
            request_id=request.request_id,
            new_token_ids=[],
            gradient_output=gradient_result or {},
            finish_reason=finish_reason,
        )
        self._pending.append(
            (request.client_index, EngineCoreOutputs(outputs=[output]))
        )
