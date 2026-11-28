# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step routing for the Qwen3 RDNA dense specialized decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def layer_attn_metadata(attn_metadata: Any, layer_name: str | None) -> Any:
    if attn_metadata is None or isinstance(attn_metadata, list):
        return None
    if isinstance(attn_metadata, dict):
        if layer_name is None:
            return next(iter(attn_metadata.values()), None)
        return attn_metadata.get(layer_name)
    return attn_metadata


def metadata_num_decodes(
    attn_metadata: Any,
    layer_name: str | None = None,
) -> int | None:
    metadata = layer_attn_metadata(attn_metadata, layer_name)
    num_decodes = getattr(metadata, "num_decodes", None)
    if num_decodes is not None:
        return int(num_decodes)

    # Older ROCm attention metadata does not expose num_decodes/num_prefills.
    # For PR1 we only need to identify the narrow "single decode token for one
    # request" case that the dense backend handles.
    num_actual_tokens = getattr(metadata, "num_actual_tokens", None)
    max_query_len = getattr(metadata, "max_query_len", None)
    block_table = getattr(metadata, "block_table", None)
    if (
        num_actual_tokens is None
        or max_query_len is None
        or not isinstance(block_table, torch.Tensor)
        or block_table.dim() != 2
    ):
        return None
    if int(max_query_len) != 1:
        return 0
    if int(num_actual_tokens) == block_table.shape[0]:
        return int(block_table.shape[0])
    return None


@dataclass(frozen=True, slots=True)
class StepDecision:
    use_specialized: bool
    reason: str = ""
    block_table: torch.Tensor | None = None


def decide_step(
    *,
    input_ids: torch.Tensor | None,
    intermediate_tensors: object | None,
    inputs_embeds: torch.Tensor | None,
    metadata: object,
    layer_name: str | None,
) -> StepDecision:
    num_decodes = metadata_num_decodes(metadata, layer_name)
    if num_decodes != 1:
        return StepDecision(False, f"num_decodes={num_decodes}")
    if input_ids is None:
        return StepDecision(False, "input_ids_missing")
    if intermediate_tensors is not None:
        return StepDecision(False, "intermediate_tensors_present")
    if inputs_embeds is not None:
        return StepDecision(False, "inputs_embeds_present")
    if input_ids.shape[0] != 1:
        return StepDecision(False, f"token_count={input_ids.shape[0]}")
    layer_metadata = layer_attn_metadata(metadata, layer_name)
    if getattr(layer_metadata, "num_prefills", 0):
        return StepDecision(False, "mixed_prefill_decode_step")
    block_table = getattr(layer_metadata, "block_table", None)
    if (
        block_table is None
        or block_table.dtype != torch.int32
        or block_table.dim() != 2
        or not block_table[0].is_contiguous()
    ):
        return StepDecision(False, "block_table_invalid")
    return StepDecision(True, block_table=block_table)


def matches_hidden_states(hidden_states: Any, mk_out: Any) -> bool:
    if hidden_states is None or mk_out is None:
        return False
    if not hasattr(hidden_states, "data_ptr") or not hasattr(mk_out, "data_ptr"):
        return False
    if hidden_states.data_ptr() == mk_out.data_ptr():
        return True
    if (
        hidden_states.shape != mk_out.shape
        or hidden_states.dtype != mk_out.dtype
        or hidden_states.device != mk_out.device
    ):
        return False
    return bool(
        hidden_states.shape
        and hidden_states.shape[0] == 1
        and hidden_states.ndim == 2
        and torch.equal(hidden_states, mk_out)
    )
