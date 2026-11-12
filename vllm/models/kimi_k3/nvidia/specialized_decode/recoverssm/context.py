# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's deferred RecoverSSM commit (backend `fp32_deferred`):
the FP32-state commit of the Triton RecoverSSM (same association; FP32
checkpoints), whose commit after sampling turns each request's accepted count into
the pending replay of its record slot (its request-state index); the next verify
replays it (the specialized KDA core in the same pass, the stock verify after
`flush`)."""

from dataclasses import dataclass, field

import torch

from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
    KDARecoverSSMCommitContext,
    KDARecoverSSMRecords,
    _commit_kda_state_kernel,
    _compact_conv_state_kernel,
    _prepare_commit_plan_kernel,
)
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from .kernels import _defer_commit_kernel, _prepare_flush_plan_kernel

FP32_DEFERRED = "fp32_deferred"


def allocate_records(
    max_num_reqs: int,
    num_heads: int,
    spec_query_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> KDARecoverSSMRecords:
    """The verify records of a deferred RecoverSSM layer, indexed by record slot.

    Each tensor is a view that starts one row into its allocation: row -1 is a
    scratch row. A row that owns no record slot has record index -1; the verify
    kernels write its records there. Such rows can carry live state slots: a DCP
    FULL CUDA-graph capture gives its dummy requests KV context
    (`set_dcp_capture_context`: block ids 0, 1, ...) while their record rows are
    -1. Nothing reads row -1."""
    shape = (1 + max_num_reqs, num_heads, spec_query_len)
    return KDARecoverSSMRecords(
        torch.empty((*shape, head_dim), dtype=torch.float32, device=device)[1:],
        torch.empty((*shape, 2 * head_dim), dtype=dtype, device=device)[1:],
    )


@dataclass
class KDARecoverSSMDeferred:
    """A step's deferred RecoverSSM (the KDA metadata's `recoverssm_deferred`,
    set by the specialized decode's metadata build): after sampling, each request's
    accepted count becomes the pending replay of its record slot instead of a
    commit; the next verify replays it."""

    record_rows: torch.Tensor  # [>= spec rows] int32: record slot per row, -1: none
    pending: torch.Tensor  # [max_num_reqs] int32, per record slot
    # [spec rows] mask of rows committed after sampling (align: near a block
    # boundary), None: no row.
    eager: torch.Tensor | None
    context: "KDADeferredCommitContext" = field(repr=False, compare=False)
    state_indices: torch.Tensor = field(repr=False, compare=False)
    flushed: bool = False

    def flush(self) -> None:
        """Fold the step's pending replays into the checkpoints (every KDA layer,
        once per step) before a verify that does not replay them."""
        if self.flushed:
            return
        self.flushed = True
        rows = self.state_indices.shape[0]
        self.context.flush(self.state_indices, self.record_rows[:rows], self.pending)


@dataclass
class KDADeferredCommitContext(KDARecoverSSMCommitContext):
    """`KDARecoverSSMCommitContext` with the `fp32_deferred` backend: the
    FP32-state commit (`cutedsl.kda.commit`) and the deferred replay. Falls back
    to the Triton commit (backend `triton`) where the FP32 commit cannot run."""

    @classmethod
    def create(  # type: ignore[override]
        cls,
        layers,
        *,
        spec_query_len: int,
        max_num_reqs: int,
        backend: str = FP32_DEFERRED,
    ) -> "KDADeferredCommitContext":
        assert backend == FP32_DEFERRED
        context = super().create(
            layers,
            spec_query_len=spec_query_len,
            max_num_reqs=max_num_reqs,
            backend="triton",
        )
        assert isinstance(context, cls)
        from ..cutedsl.kda import commit as sdec_commit

        state_ref = context.checkpoints[0]
        if sdec_commit.supported(state_ref, context.lower_bound):
            for align in (False, True):
                for indexed in (False, True):
                    sdec_commit.compile_commit_f32(
                        state_ref, spec_query_len, layers[0].A_log.dtype, align, indexed
                    )
            context.backend = FP32_DEFERRED  # type: ignore[assignment]
        return context

    def commit(
        self,
        num_accepted_tokens: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_indices: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        num_computed_tokens: torch.Tensor | None = None,
        mamba_block_size: int | None = None,
        record_rows: torch.Tensor | None = None,
        pending: torch.Tensor | None = None,
        eager: torch.Tensor | None = None,
    ) -> None:
        """Fold accepted KDA and convolution inputs into every layer.

        Deferred RecoverSSM (`pending`, backend `fp32_deferred`): the records of
        request b are in slot `record_rows[b]`; its accepted count becomes that
        slot's pending replay (the next verify folds it in), except for the rows
        of `eager` (a [B] mask, or None: no row), which are committed now."""
        if self.backend != FP32_DEFERRED:
            assert pending is None
            super().commit(
                num_accepted_tokens,
                state_indices,
                query_start_loc,
                request_indices=request_indices,
                block_table=block_table,
                num_computed_tokens=num_computed_tokens,
                mamba_block_size=mamba_block_size,
            )
            return
        batch = state_indices.shape[0]
        if batch == 0:
            return
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM commit batch exceeds its plan capacity")
        if query_start_loc.shape[0] != batch + 1:
            raise ValueError("KDA RecoverSSM commit metadata is incompatible")
        if request_indices is not None and request_indices.shape[0] < batch:
            raise ValueError("KDA RecoverSSM request mapping is too short")
        align_args = (block_table, num_computed_tokens, mamba_block_size)
        if any(arg is not None for arg in align_args) and any(
            arg is None for arg in align_args
        ):
            raise ValueError("KDA RecoverSSM align metadata is incomplete")
        if mamba_block_size is not None and mamba_block_size < self.spec_query_len:
            raise ValueError(
                "KDA RecoverSSM align block size must cover one speculative window"
            )
        if block_table is not None and block_table.ndim != 2:
            raise ValueError("KDA RecoverSSM block table must be two-dimensional")
        device = self.checkpoints[0].device
        if (
            any(
                tensor.device != device
                for tensor in (
                    num_accepted_tokens,
                    state_indices,
                    query_start_loc,
                )
            )
            or (request_indices is not None and request_indices.device != device)
            or (block_table is not None and block_table.device != device)
            or (
                num_computed_tokens is not None and num_computed_tokens.device != device
            )
        ):
            raise ValueError("KDA RecoverSSM commit inputs must be on the same device")

        block_table_stride = (0, 0) if block_table is None else block_table.stride()
        num_computed_stride = (
            0 if num_computed_tokens is None else num_computed_tokens.stride(0)
        )

        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens,
            request_indices,
            state_indices,
            query_start_loc,
            block_table,
            num_computed_tokens,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            mamba_block_size or 1,
            block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0),
            query_start_loc.stride(0),
            block_table_stride[0],
            block_table_stride[1],
            num_computed_stride,
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        if pending is not None:
            assert record_rows is not None
            _defer_commit_kernel[(batch,)](
                state_indices,
                self.commit_lens,
                record_rows,
                pending,
                eager,
                NULL_BLOCK_ID,
                state_indices.stride(0),
                num_warps=1,
            )
            if eager is None:
                return
        self._fold(state_indices, batch, block_table is not None, record_rows)

    def flush(
        self,
        state_indices: torch.Tensor,
        record_rows: torch.Tensor,
        pending: torch.Tensor,
    ) -> None:
        """Deferred RecoverSSM: fold the pending replays of the step's requests
        (`record_rows[b]`: request b's record slot) into their checkpoints in
        place, every layer, so a verify that does not replay them (the stock
        verify) starts from the accepted state. `pending` is left: the commit
        after this step's sampling rewrites it."""
        batch = state_indices.shape[0]
        if batch == 0:
            return
        assert self.backend == FP32_DEFERRED
        if batch > self.commit_lens.shape[0]:
            raise ValueError("KDA RecoverSSM flush batch exceeds its plan capacity")
        _prepare_flush_plan_kernel[(batch,)](
            state_indices,
            record_rows,
            pending,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            state_indices.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len,
            num_warps=1,
        )
        self._fold(state_indices, batch, False, record_rows)

    def _fold(
        self,
        state_indices: torch.Tensor,
        batch: int,
        align: bool,
        record_rows: torch.Tensor | None = None,
    ) -> None:
        """Run a prepared plan: the conv state compaction and the state commit."""
        num_layers = len(self.checkpoints)
        conv_ref = self.conv_states[0]
        conv_dim = conv_ref.shape[1]
        block_history = triton.next_power_of_2(self.conv_history_len)
        block_table = True if align else None
        _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
            conv_ref,
            self.conv_state_base_addrs,
            self.conv_state_block_strides,
            self.conv_state_dim_strides,
            self.conv_state_token_strides,
            state_indices,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            NULL_BLOCK_ID,
            conv_dim,
            self.conv_history_len,
            state_indices.stride(0),
            BLOCK_D=256,
            BLOCK_HISTORY=block_history,
            ALIGN_MODE=block_table is not None,
            num_warps=4,
        )

        from ..cutedsl.kda import commit as sdec_commit

        # Indexed records: only the FP32 commit reads them (deferred steps).
        if self.backend == FP32_DEFERRED and (
            record_rows is not None
            or sdec_commit.use_commit_f32(
                self.checkpoints[0].shape[1], batch, num_layers
            )
        ):
            sdec_commit.commit_f32(
                self,
                state_indices,
                batch,
                align=block_table is not None,
                record_rows=record_rows,
            )
            return
        state_ref = self.checkpoints[0]
        _, num_heads, value_dim, key_dim = state_ref.shape
        block_k = triton.next_power_of_2(key_dim)
        block_v = min(triton.next_power_of_2(value_dim), 32)
        grid = (
            triton.cdiv(value_dim, block_v),
            batch,
            num_layers * num_heads,
        )
        _commit_kda_state_kernel[grid](
            state_ref,
            self.state_base_addrs,
            self.state_block_strides,
            self.correction_caches[0],
            self.correction_cache_base_addrs,
            self.correction_cache_block_strides,
            self.kg_caches[0],
            self.kg_cache_base_addrs,
            self.kg_cache_block_strides,
            self.A_log,
            self.dt_bias,
            state_indices,
            None,
            self.commit_lens,
            self.final_state_indices,
            self.boundary_state_indices,
            self.boundary_recovery_lens,
            self.lower_bound or 0.0,
            NULL_BLOCK_ID,
            state_ref.stride(1),
            state_ref.stride(2),
            state_ref.stride(3),
            self.correction_caches[0].stride(1),
            self.correction_caches[0].stride(2),
            self.correction_caches[0].stride(3),
            self.kg_caches[0].stride(1),
            self.kg_caches[0].stride(2),
            self.kg_caches[0].stride(3),
            self.A_log.stride(0),
            self.A_log.stride(1),
            self.dt_bias.stride(0),
            self.dt_bias.stride(1),
            self.dt_bias.stride(2),
            state_indices.stride(0),
            K=key_dim,
            V=value_dim,
            BK=block_k,
            BV=block_v,
            NUM_HEADS=num_heads,
            USE_LOWER_BOUND=self.lower_bound is not None,
            ALIGN_MODE=block_table is not None,
            num_warps=4,
            num_stages=2,
        )
