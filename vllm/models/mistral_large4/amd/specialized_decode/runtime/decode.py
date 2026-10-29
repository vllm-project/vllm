# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-mode specialized decode: lifecycle, per-step plan and fail-stop.

``plan()`` runs once per forward. A single-token decode step (eager, or inside
a FULL graph capture) runs every MoE layer as one kernel launch (``run``);
every other step runs vLLM's layers. Every predicate reads host state that
every TP rank holds alike: the kernels all-reduce in-kernel, so ranks that
disagreed would wait on each other until the fail-stop bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available

from .guards import cache_refusals, model_refusals
from .spec import DecodeConfig

FAULT_MESSAGE = (
    "Mistral Large 4 specialized decode: a kernel wait timed out on some rank "
    "(fail-stop word set); the step is discarded"
)


def tp_agree(ok: bool) -> bool:
    from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.consensus import (  # noqa: E501
        tp_agree as agree,
    )

    return agree(ok, get_tp_group().cpu_group)


@dataclass(frozen=True)
class Step:
    """One planned step: ``rows`` tokens, ``q`` consecutive ones per request."""

    rows: int


def step_refusal(
    context, layer_name: str, positions: torch.Tensor, widths, q: int = 1
) -> str | None:
    """Why this forward runs vLLM's layers, or None for the kernels: a decode
    step of whole requests of ``q`` rows each (a speculative verify block, its
    causality from each row's position; padded rows have slot -1)."""
    rows = positions.shape[0]
    if rows not in widths or rows % q:
        return f"{rows} rows"
    if positions.dtype != torch.int64:
        return f"positions {positions.dtype}"
    if context.ubatch_slices is not None:
        return "microbatched"
    metadata = context.attn_metadata
    if not isinstance(metadata, dict) or not isinstance(context.slot_mapping, dict):
        return "no per-layer metadata"
    md = cast(Any, metadata.get(layer_name))
    if md is None or md.num_prefills or md.max_query_len != q:
        return "not a pure decode step"
    if getattr(md, "num_decode_tokens", 0) > rows:
        return "more decode tokens than rows"
    bt = md.block_table
    if bt.dtype != torch.int32 or bt.dim() != 2 or bt.stride(1) != 1:
        return "block table layout"
    slots = context.slot_mapping.get(layer_name)
    if slots is None or slots.dtype != torch.int64:
        return "slot mapping"
    return None


class SpecializedDecode:
    """Built after weight loading; never replaces stock modules or parameters."""

    def __init__(self, model, config: DecodeConfig, max_model_len: int) -> None:
        self.model = model
        self.config = config
        reasons = model_refusals(model, config)
        if not tp_agree(not reasons):
            raise ValueError(
                "Mistral Large 4 specialized decode model unsupported: "
                + ("; ".join(reasons) or "refused on another TP rank")
            )
        from .runner import MonoRunner

        self.runner = MonoRunner(model, config, max_model_len)
        self.layer_name = self.runner.layer_name
        self.index_name = self.runner.index_name
        self._closed = False
        self.mono_steps = 0
        self.stock_steps = 0

    @classmethod
    def build(cls, model, config: DecodeConfig, vllm_config) -> SpecializedDecode:
        return cls(model, config, vllm_config.model_config.max_model_len)

    def _bind(self, context) -> None:
        """The first planned step: check the caches, build the launchers."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mistral Large 4 specialized decode: first bind during graph capture"
            )
        reasons = cache_refusals(self.model, self.runner.max_model_len)
        slots = context.slot_mapping
        index_slots = slots.get(self.index_name)
        if index_slots is None or not torch.equal(
            index_slots[:1], slots[self.layer_name][:1]
        ):
            # the kernels insert K/V and the index key through one slot mapping
            reasons.append("the index cache does not share the KV cache's block ids")
        if not tp_agree(not reasons):
            raise ValueError(
                "Mistral Large 4 specialized decode caches unsupported: "
                + ("; ".join(reasons) or "refused on another TP rank")
            )
        md = context.attn_metadata[self.layer_name]
        page = self.model.layers[1].self_attn.attn.kv_cache.shape[1]
        self.runner.bind(page, md.block_table.stride(0))

    def plan(self, positions: torch.Tensor, intermediate_tensors) -> Step | None:
        if (
            intermediate_tensors is not None
            or self._closed
            or not is_forward_context_available()
        ):
            self.stock_steps += 1
            return None
        context = get_forward_context()
        if step_refusal(
            context, self.layer_name, positions, self.config.widths, self.config.q
        ):
            self.stock_steps += 1
            return None
        if not self.runner.bound:
            self._bind(context)
        stride = context.attn_metadata[self.layer_name].block_table.stride(0)
        if stride != self.runner.block_table_stride:
            raise RuntimeError(
                f"block table width {stride} != bound {self.runner.block_table_stride}"
            )
        self.mono_steps += 1
        return Step(positions.shape[0])

    def run(
        self,
        step: Step,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.runner.forward(input_ids, positions, inputs_embeds)

    def after_step(self) -> None:
        """After every forward (kernels or stock, eager or in a FULL graph),
        before logits: a device-side assert that the fail-stop word is clear. A
        set word traps the GPU queue, so none of the step's later work runs and
        the worker aborts: no token of a failed step reaches a client."""
        if not self._closed:
            torch._assert_async(self.runner.fault_word[0] == 0, FAULT_MESSAGE)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.runner.close()
