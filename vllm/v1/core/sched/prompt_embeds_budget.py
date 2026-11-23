# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.v1.core.sched.output import num_prompt_embeds_to_send
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig, VllmConfig


def prompt_embeds_budget_rows(scheduler_config: "SchedulerConfig") -> int:
    """`prompt_embeds` rows each worker holds on the GPU: one step's rows, which
    it stages there while the rest stay in host memory."""
    return scheduler_config.max_num_batched_tokens


def prompt_embeds_row_bytes(vllm_config: "VllmConfig") -> int:
    """GPU memory of one `prompt_embeds` row, or 0 without prompt embeds.

    Leaves out the 1-byte `prompt_is_token_ids` flag the workers also hold per
    row of a mixed prompt."""
    model_config = vllm_config.model_config
    if not model_config.enable_prompt_embeds:
        return 0
    assert isinstance(model_config.dtype, torch.dtype)
    return model_config.get_inputs_embeds_size() * model_config.dtype.itemsize


def prompt_embeds_budget_bytes(vllm_config: "VllmConfig") -> int:
    """GPU memory of the `prompt_embeds` rows a worker stages per step, or 0
    without prompt embeds."""
    row_bytes = prompt_embeds_row_bytes(vllm_config)
    if not row_bytes:
        return 0
    return prompt_embeds_budget_rows(vllm_config.scheduler_config) * row_bytes


class SentPromptEmbeds:
    """The `prompt_embeds` rows sent to the workers, per request: those of
    positions not computed yet when the request was sent. The workers hold them
    until its prompt is computed, preempted or freed.
    """

    def __init__(self) -> None:
        # req_id -> rows sent, for requests whose workers hold rows.
        self._rows: dict[str, int] = {}

    def add(self, request: Request, num_computed_tokens: int) -> None:
        if request.prompt_embeds is None:
            return
        if request.request_id in self._rows:
            raise RuntimeError(
                f"prompt_embeds rows of request {request.request_id} are already "
                "sent; sending them again would leak them."
            )
        self._rows[request.request_id] = num_prompt_embeds_to_send(
            request, num_computed_tokens
        )

    def release(self, req_id: str) -> None:
        self._rows.pop(req_id, None)

    def offset(self, request: Request) -> int:
        """Prompt position of the first row the workers hold: rows below it
        were computed before the request was sent, or all of them once the
        workers dropped them. 0 without prompt embeds."""
        if request.prompt_embeds is None:
            return 0
        return request.num_prompt_tokens - self._rows.get(request.request_id, 0)
