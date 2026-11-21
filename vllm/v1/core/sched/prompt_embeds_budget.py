# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.v1.core.sched.output import num_prompt_embeds_to_send
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig, VllmConfig


def prompt_embeds_budget_rows(
    max_model_len: int, scheduler_config: "SchedulerConfig"
) -> int:
    """`prompt_embeds` rows the workers may hold at once, summed over the
    requests whose prompts are not computed yet: one max-length prompt still
    being prefilled plus a full step of new prompts."""
    return max_model_len + scheduler_config.max_num_batched_tokens


def prompt_embeds_row_bytes(vllm_config: "VllmConfig") -> int:
    """GPU memory of one `prompt_embeds` row, or 0 without prompt embeds.

    Leaves out the 1-byte `prompt_is_token_ids` flag the workers also hold per
    row of a mixed prompt."""
    model_config = vllm_config.model_config
    if not model_config.enable_prompt_embeds:
        return 0
    assert isinstance(model_config.dtype, torch.dtype)
    return model_config.get_inputs_embeds_size() * model_config.dtype.itemsize


class PromptEmbedsBudget:
    """The `prompt_embeds` rows the workers hold on the GPU, against the
    `max_num_rows` that KV cache planning sets memory aside for.

    A request reserves the rows it is sent, those of positions not computed
    yet, and releases them once its prompt is computed, preempted or freed.

    Attributes:
        max_num_rows: Rows the workers may hold at once, summed over requests.
        num_rows: Rows reserved now, summed over requests.

    """

    def __init__(self, max_num_rows: int) -> None:
        self.max_num_rows = max_num_rows
        self.num_rows = 0
        # req_id -> rows reserved by the request, for requests holding rows.
        self._rows: dict[str, int] = {}

    def fits(self, request: Request, num_computed_tokens: int) -> bool:
        num_rows = num_prompt_embeds_to_send(request, num_computed_tokens)
        return self.num_rows + num_rows <= self.max_num_rows

    def reserve(self, request: Request, num_computed_tokens: int) -> None:
        if request.prompt_embeds is None:
            return
        if request.request_id in self._rows:
            raise RuntimeError(
                f"prompt_embeds rows of request {request.request_id} are already "
                "reserved; reserving again would leak them."
            )
        num_rows = num_prompt_embeds_to_send(request, num_computed_tokens)
        self._rows[request.request_id] = num_rows
        self.num_rows += num_rows

    def release(self, req_id: str) -> None:
        self.num_rows -= self._rows.pop(req_id, 0)

    def offset(self, request: Request) -> int:
        """Prompt position of the first row the workers hold: rows below it
        were computed before the request was sent, or all of them once the
        workers dropped them. 0 without prompt embeds."""
        if request.prompt_embeds is None:
            return 0
        return request.num_prompt_tokens - self._rows.get(request.request_id, 0)
