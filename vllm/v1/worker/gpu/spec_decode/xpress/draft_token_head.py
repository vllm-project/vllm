# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    AcceptanceEstimationInput,
)
from vllm.v1.worker.gpu.spec_decode.draft_token_head import (
    DraftBlockInput,
    DraftBlockOutput,
)


class XPressDraftTokenHead:
    """Drafts greedily with K Jacobi refinement passes of the XPress refiner
    head over each request's block. Fixed-shape and loop-free per pass, so it
    is CUDA-graph capturable."""

    def __init__(self, num_jacobi_passes: int):
        self.num_jacobi_passes = num_jacobi_passes
        self.xpress_head: nn.Module | None = None

    def post_load(self, draft_model: nn.Module) -> None:
        self.xpress_head = draft_model.model.xpress_head

    def draft(self, input: DraftBlockInput) -> DraftBlockOutput:
        # Each block is the anchor (whose refined output is discarded) and
        # the draft slots.
        num_reqs, B = input.num_reqs, input.num_query_per_req
        h_full = input.hidden_states[: num_reqs * B].view(num_reqs, B, -1)
        # The anchor's own predecessor lives in the KV cache, not in any id
        # buffer the speculator can reach, so the anchor id stands in for it.
        assert self.xpress_head is not None, "call post_load() first"
        draft_tokens, refined_logits = self.xpress_head.jacobi_refine_greedy(
            input.logits.view(num_reqs, B - 1, -1),
            h_full,
            input.anchor_ids,
            input.anchor_ids,
            self.num_jacobi_passes,
            with_logits=input.needs_logits,
        )
        # Acceptance is estimated from the last pass's refined logits.
        acceptance = AcceptanceEstimationInput(
            refined_logits, input.idx_mapping, input.draft_step
        )
        return DraftBlockOutput(draft_tokens, acceptance)
