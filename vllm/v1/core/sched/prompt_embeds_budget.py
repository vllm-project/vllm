# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.core.sched.output import num_prompt_embeds_to_send
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig


def prompt_embeds_budget_rows(scheduler_config: "SchedulerConfig") -> int:
    """Prompt embeddings each worker holds on the GPU: one step's, which it
    stages there while the rest stay in host memory."""
    return scheduler_config.max_num_batched_tokens


class SentPromptEmbeds:
    """The prompt embeddings sent to the workers, per request: those of
    positions not computed yet when the request was sent. The workers hold them
    until its prompt is computed, preempted or freed.
    """

    def __init__(self) -> None:
        # req_id -> number of embeddings sent, for requests whose workers
        # hold them.
        self._num_sent: dict[str, int] = {}

    def add(self, request: Request, num_computed_tokens: int) -> None:
        if request.prompt_embeds is None:
            return
        if request.request_id in self._num_sent:
            raise RuntimeError(
                f"prompt embeddings of request {request.request_id} are already "
                "sent; sending them again, the workers would hold them twice."
            )
        self._num_sent[request.request_id] = num_prompt_embeds_to_send(
            request, num_computed_tokens
        )

    def release(self, req_id: str) -> None:
        self._num_sent.pop(req_id, None)

    def offset(self, request: Request) -> int:
        """Prompt position of the first embedding the workers hold: those
        below it were computed before the request was sent, or all of them once
        the workers dropped them. 0 without prompt embeds."""
        if request.prompt_embeds is None:
            return 0
        return request.num_prompt_tokens - self._num_sent.get(request.request_id, 0)
