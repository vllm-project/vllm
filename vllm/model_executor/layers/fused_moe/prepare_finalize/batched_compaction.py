# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


def _maybe_contiguous(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    return x


@triton.jit
def _copy_expert_prefix_kernel(
    src_ptr,
    dst_ptr,
    ELEMENTS: tl.constexpr,
    SRC_EXPERT_STRIDE: tl.constexpr,
    DST_EXPERT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        src_ptr + expert * SRC_EXPERT_STRIDE + offsets,
        mask=offsets < ELEMENTS,
        other=0.0,
    )
    tl.store(
        dst_ptr + expert * DST_EXPERT_STRIDE + offsets,
        values,
        mask=offsets < ELEMENTS,
    )


def _copy_expert_prefix(src: torch.Tensor, dst: torch.Tensor, rows: int) -> None:
    elements = rows * src.shape[2]
    _copy_expert_prefix_kernel[(src.shape[0], triton.cdiv(elements, 4096))](
        src, dst, elements, src.stride(0), dst.stride(0), 4096
    )


class BatchedExpertCompaction:
    """Compact packed expert prefixes and restore the communication layout."""

    def __init__(self):
        self._compact_tokens: int | None = None
        self._restore_shapes: list[tuple[int, ...] | None] = [None, None]
        self._experts: mk.SupportsBatchedCompactionExperts | None = None

    def configure(
        self,
        expert_capacity: int | None,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        supports_token_dropping: bool,
        experts: mk.SupportsBatchedCompactionExperts,
        use_fp8_dispatch: bool,
    ) -> int | None:
        supports_compaction = experts.supports_batched_compaction(use_fp8_dispatch)
        if (
            supports_token_dropping
            and expert_capacity is not None
            and supports_compaction
        ):
            self._experts = experts
            tokens_per_dispatcher = max(1, min(max_tokens_per_rank, expert_capacity))
            self._compact_tokens = tokens_per_dispatcher * num_dispatchers
            logger.info_once(
                "Compact expert layout: at most %d rows per expert; "
                "dispatch limit remains %d tokens per rank.",
                self._compact_tokens,
                max_tokens_per_rank,
            )
            return tokens_per_dispatcher

        self._compact_tokens = None
        self._experts = None
        return None

    def compact(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ubatch_id: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._restore_shapes[ubatch_id] = None
        if self._compact_tokens is not None:
            payload = x if isinstance(x, torch.Tensor) else x[0]
            payload_rows = payload.shape[1]
            if self._compact_tokens < payload_rows:
                self._restore_shapes[ubatch_id] = tuple(payload.shape)
                if isinstance(x, torch.Tensor):
                    if payload.is_cuda and payload.is_contiguous():
                        compact = payload.new_empty(
                            (payload.shape[0], self._compact_tokens, payload.shape[2])
                        )
                        _copy_expert_prefix(payload, compact, self._compact_tokens)
                        x = compact
                    else:
                        x = _maybe_contiguous(payload[:, : self._compact_tokens])
                else:
                    values, scales = x
                    scale_slice = scales[:, : self._compact_tokens]
                    if self._experts is not None and (
                        self._experts.use_row_major_dispatch_scales
                    ):
                        compact_scales = _maybe_contiguous(scale_slice)
                    else:
                        compact_scales = torch.empty_like(
                            scale_slice, memory_format=torch.preserve_format
                        )
                        compact_scales.copy_(scale_slice, non_blocking=True)
                    if values.is_cuda and values.is_contiguous():
                        compact_values = values.new_empty(
                            (values.shape[0], self._compact_tokens, values.shape[2])
                        )
                        _copy_expert_prefix(
                            values, compact_values, self._compact_tokens
                        )
                    else:
                        compact_values = _maybe_contiguous(
                            values[:, : self._compact_tokens]
                        )
                    x = compact_values, compact_scales
        return x

    def restore(
        self, output: torch.Tensor, ubatch_id: int
    ) -> tuple[torch.Tensor, Callable[[], None]]:
        shape = self._restore_shapes[ubatch_id]
        if shape is not None:
            # Combine only reads live token positions, which all lie in the
            # compacted head, so the tail rows need no zero fill.
            restored = output.new_empty(shape)
            if output.is_cuda and output.is_contiguous():
                _copy_expert_prefix(output, restored, output.shape[1])
            else:
                restored[:, : output.shape[1]].copy_(output, non_blocking=True)
            output = restored

        def receiver():
            # Retain the restored allocation through the asynchronous recv hook.
            _ = output
            self._restore_shapes[ubatch_id] = None

        return output, receiver
