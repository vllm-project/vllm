# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length-tail policy for Uno speculative decoding.

Uno's draft writes one seed row and ``K-1`` noisy rows beyond the target's
query rows, so every drafted step reserves ``K`` extra KV slots. On the step
where a request can reach its output limit (``max_tokens``) or the model
context limit, the draft cannot be used at all: one target sample ends the
request. Without this policy the scheduler pads that terminal step to ``K+1``
tokens, allocates KV for rows it will never consume, and under KV pressure can
preempt other requests for nothing.

The policy chooses ``K=0`` before KV admission on a step that could reach the
request's limit, and remembers that choice across preemption. It does not
persist across a new streaming-input turn or the end of the request.

When a per-request "next draft width" override is available, the plugin can set
it to zero on a possibly-terminal step and this policy disappears.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


class UnoTailPolicy:
    """Contained Uno length-tail admission policy for the scheduler."""

    def __init__(self, enabled: bool, max_model_len: int) -> None:
        self.enabled = enabled
        self.max_model_len = max_model_len
        # Preserve the tail decision across preemption, but not across request
        # lifetimes (including a new streaming-input turn).
        self._tail_requests: set[Request] = set()

    def _output_token_limit(self, request: Request) -> int:
        return min(request.max_tokens, self.max_model_len - request.num_prompt_tokens)

    def apply(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        """Choose K=0 before KV admission for a possibly terminal step."""
        if not self.enabled:
            return num_new_tokens
        target_queries = (
            request.num_tokens + request.num_output_placeholders - num_computed_tokens
        )
        if num_new_tokens < target_queries:
            # Preserve nonfinal prefill chunks without treating them as samples.
            return num_new_tokens
        max_output = num_new_tokens - target_queries + 1
        # This upper-bound policy deliberately forgoes usable drafts.
        if (
            request.num_output_tokens + request.num_output_placeholders + max_output
            >= self._output_token_limit(request)
        ):
            self._tail_requests.add(request)
        if request in self._tail_requests:
            request.spec_token_ids = []
            return target_queries
        return num_new_tokens

    def in_tail(self, request: Request) -> bool:
        return request in self._tail_requests

    def forget(self, request: Request) -> None:
        self._tail_requests.discard(request)
