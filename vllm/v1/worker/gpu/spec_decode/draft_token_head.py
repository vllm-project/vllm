# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft token heads: how a block-diffusion speculator turns the draft model's
outputs into draft tokens."""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from vllm.v1.watermarking.spec_decode import DraftWatermarker
from vllm.v1.worker.gpu.sample.draft import sample_independently
from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import (
    AcceptanceEstimationInput,
)
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache


@dataclass
class DraftBlockInput:
    """A block-diffusion draft step's input to `DraftTokenHead.draft`. Draft
    rows are ordered (request, draft position)."""

    num_reqs: int
    num_query_per_req: int
    num_speculative_steps: int
    # [num_tokens, hidden_size], each request's num_query_per_req queries.
    hidden_states: torch.Tensor
    # [num_rows, vocab_size] the draft model's logits at the draft rows.
    logits: torch.Tensor
    # [num_reqs] the verified token each request's block drafts after.
    anchor_ids: torch.Tensor
    # [num_rows] per draft row: the position its sampling is keyed by, its
    # request state index and its draft position.
    sample_src_positions: torch.Tensor
    idx_mapping: torch.Tensor
    draft_step: torch.Tensor
    # [max_num_reqs]
    temperature: torch.Tensor
    seeds: torch.Tensor
    draft_logits: DraftLogitsCache | None
    watermarker: DraftWatermarker | None
    use_fp64_gumbel: bool
    # Whether the acceptance estimator reads the output's logits.
    needs_logits: bool


@dataclass
class DraftBlockOutput:
    # [num_reqs, num_speculative_steps]
    draft_tokens: torch.Tensor
    acceptance: AcceptanceEstimationInput


class DraftTokenHead(Protocol):
    def post_load(self, draft_model: nn.Module) -> None:
        """Take what it needs (e.g. its weights) from the loaded draft model."""
        ...

    def draft(self, input: DraftBlockInput) -> DraftBlockOutput: ...


class IndependentDraftTokenHead:
    """Samples every draft position independently from its logits."""

    def post_load(self, draft_model: nn.Module) -> None:
        return

    def draft(self, input: DraftBlockInput) -> DraftBlockOutput:
        draft_tokens = sample_independently(
            input.logits,
            input.sample_src_positions,
            input.idx_mapping,
            input.temperature,
            input.seeds,
            input.draft_step,
            input.draft_logits,
            input.watermarker,
            input.use_fp64_gumbel,
            num_steps=input.num_speculative_steps,
        )
        acceptance = AcceptanceEstimationInput(
            input.logits, input.idx_mapping, input.draft_step
        )
        return DraftBlockOutput(draft_tokens.view(input.num_reqs, -1), acceptance)
