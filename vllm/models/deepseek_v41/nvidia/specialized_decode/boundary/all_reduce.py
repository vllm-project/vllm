# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's fused all-reduce + mHC kernel and its publish
bookkeeping."""

import torch
from transformers import PreTrainedConfig

from vllm.distributed import get_tp_group
from vllm.platforms import current_platform

from ...ops.cute_dsl import AllReduceMHC
from .. import FUSED_MHC_MAX_ROWS


class BoundaryAllReduce(AllReduceMHC):
    """``AllReduceMHC`` sized for every FULL decode graph of the executor, with
    the residual-only variant and, optionally, the MXFP8 layer input.

    A producer that stores its contribution into the mailbox itself (the
    publishing ``wo_b`` GEMM) calls ``mark_published``; the next ``__call__``
    with that tensor then launches only the consumer.
    """

    def __init__(self, config: PreTrainedConfig, quantize_layer_input: bool) -> None:
        super().__init__(
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            max_num_tokens=FUSED_MHC_MAX_ROWS,
            top_k=config.num_experts_per_tok,
            device=current_platform.current_device(),
            quantize_layer_input=quantize_layer_input,
            residual_only=True,
        )
        group = get_tp_group()
        self.tp = group.world_size
        self.rank = group.rank_in_group
        self._published: torch.Tensor | None = None

    def publish_target(self) -> tuple[int, int]:
        """(mailbox publish address, stage-state address) for a producer
        that stores its contribution itself, as ``_SharedOnlyPublishDeviceKernel``
        does: generation ``stage_state[0]``, slot ``[rank][token]``, BF16 with
        negative zero cleared, then ``stage_state[1] = stage_state[0]``. The
        publish address is the mailbox's multicast address, or with ``unicast``
        the address of the table of every rank's mailbox base. Both stay valid
        for this instance's lifetime."""
        return self._multicast, self._stage_state.data_ptr()

    def mark_published(self, x: torch.Tensor) -> None:
        """Record that ``x``'s kernel published it; ``__call__(x, ...)``, which
        must come next, then launches only the consumer."""
        if self._published is not None:
            raise RuntimeError("a published contribution is still unreduced")
        self._published = x

    def reset(self) -> None:
        """Drop a publish an aborted step left unreduced."""
        self._published = None

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        published, self._published = self._published, None
        if published is None:
            return super().__call__(
                x, residual, post, comb, pre, norm_weight, norm_eps, mxfp8_out
            )
        if published is not x:
            raise RuntimeError("reducing a tensor other than the published one")
        m = x.shape[0]
        self._check(m, residual, post, comb, pre)
        return self._collect(
            residual, post, comb, pre, norm_weight, norm_eps, m, mxfp8_out
        )

    def finalize(
        self,
        gemm2_permuted: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        shared_output: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor | None,
        norm_weight: torch.Tensor | None,
        norm_eps: float,
        mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._published is not None:
            raise RuntimeError("a published contribution is still unreduced")
        return super().finalize(
            gemm2_permuted,
            expert_weights,
            expanded_idx_to_permuted_idx,
            shared_output,
            residual,
            post,
            comb,
            pre,
            norm_weight,
            norm_eps,
            mxfp8_out,
        )
