# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrapper functions that delegate to the shared DFlash implementations."""

from vllm.config import ModelConfig
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
)
from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
    get_dflash_causal,
)


def get_domino_causal(draft_model_config: ModelConfig) -> bool:
    """Whether the Domino draft uses causal (vs non-causal) attention."""
    return get_dflash_causal(draft_model_config)


def prepare_domino_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: "torch.Tensor",
    context_positions: "torch.Tensor",
    context_slot_mapping: "torch.Tensor",
    sample_indices: "torch.Tensor",
    sample_pos: "torch.Tensor",
    sample_idx_mapping: "torch.Tensor",
    input_batch: InputBatch,
    num_sampled: "torch.Tensor",
    num_rejected: "torch.Tensor",
    last_sampled: "torch.Tensor",
    next_prefill_tokens: "torch.Tensor",
    block_table: "torch.Tensor",
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
) -> None:
    """Prepare Domino draft inputs (delegates to DFlash)."""
    return prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        block_table,
        block_size,
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        sample_from_anchor,
    )
