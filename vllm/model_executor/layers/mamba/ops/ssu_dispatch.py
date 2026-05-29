# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
either the Triton or FlashInfer backend based on the configured
`MambaBackendEnum`. Follows SGLang's dispatch pattern adapted for vLLM.
"""

from abc import ABC, abstractmethod

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


@triton.jit
def _update_checkpointing_trackers_kernel(
    cache_buf_idx,
    prev_num_accepted_tokens,
    state_batch_indices,
    cu_seqlens,
    num_accepted_tokens,
    fixed_seq_len: tl.constexpr,
    max_window: tl.constexpr,
    pad_slot_id,
    n_slots,
    HAS_CU_SEQLENS: tl.constexpr,
    HAS_NUM_ACCEPTED: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(state_batch_indices + offsets, mask=mask, other=pad_slot_id)
    valid = mask & (slots != pad_slot_id)
    if HAS_CU_SEQLENS:
        seq_lens = tl.load(cu_seqlens + offsets + 1, mask=mask, other=0) - tl.load(
            cu_seqlens + offsets, mask=mask, other=0
        )
    else:
        seq_lens = tl.full((BLOCK,), fixed_seq_len, tl.int32)
    # `seq_lens` is what the kernel wrote (NPREDICTED). Under MTP only the
    # first `accepted` tokens are kept; for window-overflow detection the
    # full NPREDICTED write counts, but `prev_num_accepted_tokens` must
    # advance only by the accepted count.
    if HAS_NUM_ACCEPTED:
        accepted = tl.load(num_accepted_tokens + offsets, mask=mask, other=0)
    else:
        accepted = seq_lens
    prev = tl.load(prev_num_accepted_tokens + slots, mask=valid, other=0)
    must_checkpoint = prev + seq_lens > max_window
    old_buf = tl.load(cache_buf_idx + slots, mask=valid, other=0)
    new_buf = tl.where(must_checkpoint, 1 - old_buf, old_buf)
    new_prev = tl.where(must_checkpoint, accepted, prev + accepted)
    tl.store(cache_buf_idx + slots, new_buf, mask=valid)
    tl.store(prev_num_accepted_tokens + slots, new_prev, mask=valid)


@triton.jit
def _reset_checkpointing_trackers_kernel(
    cache_buf_idx,
    prev_num_accepted_tokens,
    state_batch_indices,
    pad_slot_id: tl.constexpr,
    n_slots: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(state_batch_indices + offsets, mask=mask, other=pad_slot_id)
    valid = mask & (slots != pad_slot_id)
    tl.store(cache_buf_idx + slots, 0, mask=valid)
    tl.store(prev_num_accepted_tokens + slots, 0, mask=valid)


@triton.jit
def _make_old_cumAdt_cumulative_kernel(
    old_cumAdt,        # (cache, 2, nheads, max_window) f32, indexed by cache_slot
    state_batch_indices,    # (n_slots,) int32; batch_slot -> cache_slot
    cache_buf_idx,          # (cache_size,) int32; pre-kernel value, indexed by cache_slot
    prev_num_accepted_tokens,  # (cache_size,) int32; pre-kernel value, indexed by cache_slot
    cumAdt_stride_seq,
    cumAdt_stride_dbuf,
    cumAdt_stride_head,
    seq_len: tl.constexpr,
    max_window: tl.constexpr,
    nheads: tl.constexpr,
    pad_slot_id,
    n_slots,
    BLOCK_H: tl.constexpr,
) -> None:
    """Convert the per-call prefix-sum the kernel just wrote at
    `[prev_k .. prev_k + seq_len - 1]` into the running cumulative
    cumAdt-up-to-slot.

    The FlashInfer kernel writes `smem.cumAdt[lane]` which is the inclusive
    prefix sum across the NPREDICTED (= `seq_len`) tokens *in the current
    call only*. The commit step on the *next* call reads
    `total_old_cumAdt = old_cumAdt[prev_k_new - 1]` and treats it as
    cumulative over the whole accumulated window. Without this update the
    commit math inverts the sign of cross-slot exponent differences for
    some heads and produces +/-inf/NaN. We patch each just-written slot
    here by adding `old_cumAdt[prev_k - 1]` (the running cumulative from
    the prior call) so the running cumulative is correct for the next
    commit.

    Under MTP (NPREDICTED > 1) some of the written entries correspond to
    rejected speculative tokens; they will be overwritten on the next
    call (which starts at `prev_k + num_accepted_tokens`). It is harmless
    to fold them now — their wrong-but-overwritten values are never read.

    Skips slots where:
      - the cache slot is padded,
      - the kernel committed this step (`prev_k + seq_len > max_window`):
        the kernel wrote to slot 0 of the OTHER buffer with a fresh
        intra-call prefix sum that is already globally cumulative
        (prior cumulative is 0 in the new window),
      - `prev_k == 0`: slot 0..seq_len-1 of the current buffer are already
        globally cumulative for the first call into a window.
    """
    slot = tl.program_id(0)
    if slot >= n_slots:
        return
    cache_slot = tl.load(state_batch_indices + slot)
    if cache_slot == pad_slot_id:
        return
    prev_k = tl.load(prev_num_accepted_tokens + cache_slot)
    committed = (prev_k + seq_len) > max_window
    if committed or prev_k < 1:
        return
    buf = tl.load(cache_buf_idx + cache_slot)
    h_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_offsets < nheads
    base = (
        cache_slot * cumAdt_stride_seq
        + buf * cumAdt_stride_dbuf
        + h_offsets * cumAdt_stride_head
    )
    prev = tl.load(old_cumAdt + base + (prev_k - 1), mask=mask)
    # `seq_len` is the constexpr NPREDICTED; tl.static_range unrolls.
    for i in tl.static_range(seq_len):
        p = prev_k + i
        cur = tl.load(old_cumAdt + base + p, mask=mask)
        tl.store(old_cumAdt + base + p, cur + prev, mask=mask)


@triton.jit
def _copy_checkpointing_slots_kernel(
    tensor,
    src_indices,
    dst_indices,
    slot_size: tl.constexpr,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    slot = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < slot_size
    src = tl.load(src_indices + slot)
    dst = tl.load(dst_indices + slot)
    valid = (src != pad_slot_id) & (dst != pad_slot_id) & (src != dst)
    values = tl.load(tensor + src * slot_size + offsets, mask=mask & valid)
    tl.store(tensor + dst * slot_size + offsets, values, mask=mask & valid)


class MambaSSUBackend(ABC):
    """Abstract base class for Mamba SSU backends."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        is_blackwell: bool = False,
        old_x: torch.Tensor | None = None,
        old_B: torch.Tensor | None = None,
        old_dt: torch.Tensor | None = None,
        old_cumAdt: torch.Tensor | None = None,
        cache_buf_idx: torch.Tensor | None = None,
        prev_num_accepted_tokens: torch.Tensor | None = None,
        state_scales: torch.Tensor | None = None,
        spec_uniform_state_slots: bool = False,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based SSU backend (vLLM's default)."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update as _triton_selective_state_update,
        )

        self._kernel = _triton_selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        is_blackwell: bool = False,
        old_x: torch.Tensor | None = None,
        old_B: torch.Tensor | None = None,
        old_dt: torch.Tensor | None = None,
        old_cumAdt: torch.Tensor | None = None,
        cache_buf_idx: torch.Tensor | None = None,
        prev_num_accepted_tokens: torch.Tensor | None = None,
        state_scales: torch.Tensor | None = None,
        spec_uniform_state_slots: bool = False,
    ) -> None:
        # Triton backend has no checkpointing or quantized-state path;
        # state_scales and spec_uniform_state_slots are accepted for API
        # parity with FlashInferSSUBackend and ignored.
        del spec_uniform_state_slots, state_scales
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            null_block_id=null_block_id,
            out=out,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
            is_blackwell=is_blackwell,
            enable_stochastic_rounding=self._mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based SSU backend."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        try:
            from flashinfer.mamba import checkpointing_ssu as _fi_checkpointing_ssu
            from flashinfer.mamba import selective_state_update as _fi_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer Mamba SSU backend. "
                "Please install a FlashInfer build with Mamba checkpointing SSU."
            ) from e
        self._kernel = _fi_ssu
        self._checkpointing_kernel = _fi_checkpointing_ssu

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        is_blackwell: bool = False,
        old_x: torch.Tensor | None = None,
        old_B: torch.Tensor | None = None,
        old_dt: torch.Tensor | None = None,
        old_cumAdt: torch.Tensor | None = None,
        cache_buf_idx: torch.Tensor | None = None,
        prev_num_accepted_tokens: torch.Tensor | None = None,
        state_scales: torch.Tensor | None = None,
        spec_uniform_state_slots: bool = False,
    ) -> None:
        rand_seed = (
            torch.randint(0, 2**32, (1,), dtype=torch.int64, device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )
        _philox_rounds = self._mamba_config.stochastic_rounding_philox_rounds or 10

        checkpointing_args = (
            old_x,
            old_B,
            old_dt,
            old_cumAdt,
            cache_buf_idx,
            prev_num_accepted_tokens,
        )
        # Under MTP, callers that route 1+num_spec tokens through the SAME
        # cache slot per sequence (mamba_cache_mode ∈ {"align", "none"}) set
        # `spec_uniform_state_slots=True`. The state_batch_indices tensor
        # then has shape (batch, 1+num_spec) with identical columns; we
        # extract the first column as the 1D form the kernel expects.
        # For "all" mode + spec the columns differ across spec positions
        # (different blocks per token) — the helper returns None and we
        # fall through to the non-checkpointing kernel.
        ckpt_state_indices = self._checkpointing_state_indices(
            state_batch_indices, spec_uniform_state_slots=spec_uniform_state_slots
        )
        can_checkpoint = (
            all(arg is not None for arg in checkpointing_args)
            and ckpt_state_indices is not None
            and state.dtype
            in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float8_e4m3fn,
                torch.int8,
            )
        )
        if can_checkpoint:
            assert old_x is not None
            assert old_B is not None
            assert old_dt is not None
            assert old_cumAdt is not None
            assert cache_buf_idx is not None
            assert prev_num_accepted_tokens is not None
            state_indices = ckpt_state_indices
            assert state_indices is not None
            kernel_state_indices = state_indices
            # Only extract dst_indices when src and dst are DIFFERENT
            # tensor objects (the "all" cache mode case). For "none"/
            # "align" the caller passes the SAME tensor object for both,
            # so no slot relocation is needed and skipping the copy
            # kernels here avoids 7 spurious launches per dispatch call.
            # The previous version called `_copy_checkpointing_slots`
            # unconditionally and relied on the kernel's `(src != dst)`
            # mask to make the copy a no-op — that's correct but wasteful,
            # and may also be a regression source if the masked copy has
            # a subtle Triton bug under certain shapes.
            if dst_state_batch_indices is not state_batch_indices:
                dst_indices = self._checkpointing_state_indices(
                    dst_state_batch_indices,
                    spec_uniform_state_slots=spec_uniform_state_slots,
                )
                if (
                    dst_indices is not None
                    and dst_indices.numel() == state_indices.numel()
                ):
                    self._copy_checkpointing_slots(
                        (
                            state,
                            old_x,
                            old_B,
                            old_dt,
                            old_cumAdt,
                            cache_buf_idx,
                            prev_num_accepted_tokens,
                        ),
                        state_indices,
                        dst_indices,
                        null_block_id,
                    )
                    kernel_state_indices = dst_indices
            ckpt_cu_seqlens = self._checkpointing_cu_seqlens(
                cu_seqlens, x, kernel_state_indices, max_seqlen
            )
            x_ckpt, dt_ckpt, B_ckpt, C_ckpt, z_ckpt, out_ckpt, ckpt_max_seqlen = (
                self._reshape_checkpointing_inputs(
                    x,
                    dt,
                    B,
                    C,
                    z,
                    out,
                    kernel_state_indices,
                    ckpt_cu_seqlens,
                    max_seqlen,
                )
            )
            kernel_max_seqlen = ckpt_max_seqlen if ckpt_cu_seqlens is not None else None
            # When we synthesised `cu_seqlens = arange(batch+1)` for the
            # simple_decode shape, every sequence is length 1 — the
            # kernel's NPREDICTED (taken from `max_seqlen`) must reflect
            # that, regardless of what the caller passed as `max_seqlen`
            # (which is a padding budget, not an actual sequence length).
            if (
                ckpt_cu_seqlens is not None
                and x.shape[0] == kernel_state_indices.numel()
            ):
                kernel_max_seqlen = 1
            self._checkpointing_kernel(
                state,
                old_x,
                old_B,
                old_dt,
                old_cumAdt,
                cache_buf_idx,
                prev_num_accepted_tokens,
                x_ckpt,
                dt_ckpt,
                A,
                B_ckpt,
                C_ckpt,
                out_ckpt,
                D=D,
                z=z_ckpt,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=kernel_state_indices,
                pad_slot_id=null_block_id,
                rand_seed=rand_seed,
                philox_rounds=_philox_rounds,
                cu_seqlens=ckpt_cu_seqlens,
                max_seqlen=kernel_max_seqlen,
                state_scale=state_scales,
            )
            # Fold the per-step value the kernel just wrote into the running
            # cumulative cumAdt so the next commit step reads a globally-
            # cumulative value; then advance the per-slot trackers.
            self._make_old_cumAdt_cumulative(
                old_cumAdt,
                cache_buf_idx,
                prev_num_accepted_tokens,
                kernel_state_indices,
                ckpt_max_seqlen,
                old_x.size(1),
                null_block_id,
            )
            self._update_checkpointing_trackers(
                cache_buf_idx,
                prev_num_accepted_tokens,
                kernel_state_indices,
                ckpt_cu_seqlens,
                ckpt_max_seqlen,
                old_x.size(1),
                null_block_id,
                num_accepted_tokens=num_accepted_tokens,
            )
            return

        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            cache_steps=state_batch_indices.size(-1)
            if cu_seqlens is not None and state_batch_indices is not None
            else 0,
            pad_slot_id=null_block_id,
            out=out,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
        )
        if (
            num_accepted_tokens is None
            and cache_buf_idx is not None
            and prev_num_accepted_tokens is not None
            and dst_state_batch_indices is not None
            and not self._same_state_indices(
                state_batch_indices, dst_state_batch_indices
            )
        ):
            dst_indices = self._checkpointing_state_indices(dst_state_batch_indices)
            if dst_indices is not None:
                self._reset_checkpointing_trackers(
                    cache_buf_idx,
                    prev_num_accepted_tokens,
                    dst_indices,
                    null_block_id,
                )

    @staticmethod
    def _same_state_indices(
        state_batch_indices: torch.Tensor | None,
        dst_state_batch_indices: torch.Tensor | None,
    ) -> bool:
        if dst_state_batch_indices is None:
            return True
        if state_batch_indices is None:
            return False
        if state_batch_indices.shape != dst_state_batch_indices.shape:
            return False
        if (
            state_batch_indices.data_ptr() == dst_state_batch_indices.data_ptr()
            and state_batch_indices.stride() == dst_state_batch_indices.stride()
        ):
            return True
        return bool(torch.equal(state_batch_indices, dst_state_batch_indices))

    @staticmethod
    def _checkpointing_state_indices(
        state_batch_indices: torch.Tensor | None,
        *,
        spec_uniform_state_slots: bool = False,
    ) -> torch.Tensor | None:
        if state_batch_indices is None:
            return None
        if state_batch_indices.dim() == 1:
            return state_batch_indices.to(torch.int32).contiguous()
        if state_batch_indices.dim() == 2 and state_batch_indices.size(1) == 1:
            return state_batch_indices[:, 0].to(torch.int32).contiguous()
        if (
            spec_uniform_state_slots
            and state_batch_indices.dim() == 2
            and state_batch_indices.size(1) > 1
        ):
            # MTP with align/none cache mode: all 1+num_spec columns point
            # at the same in-place cache slot. Take the first column;
            # per-call NPREDICTED is conveyed via cu_seqlens/max_seqlen.
            return state_batch_indices[:, 0].to(torch.int32).contiguous()
        return None

    @staticmethod
    def _checkpointing_cu_seqlens(
        cu_seqlens: torch.Tensor | None,
        x: torch.Tensor,
        state_batch_indices: torch.Tensor,
        max_seqlen: int | None,
    ) -> torch.Tensor | None:
        del x, state_batch_indices, max_seqlen
        return cu_seqlens

    @staticmethod
    def _reshape_checkpointing_inputs(
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        z: torch.Tensor | None,
        out: torch.Tensor | None,
        state_batch_indices: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        int,
    ]:
        assert out is not None
        if cu_seqlens is not None:
            # Force x/B/C contig to remove the per-token gap stride (split
            # of hidden_states_B_C produces stride(0) > nheads*dim).
            # Diagnostic: see if the kernel's gap-strided path subtly
            # mis-behaves in production at batch>=16 even though the
            # FlashInfer test passed at all batch sizes with this layout.
            x = x.contiguous()
            B = B.contiguous()
            C = C.contiguous()
            if z is not None:
                z = z.contiguous()
            return (
                x.unsqueeze(0),
                dt.unsqueeze(0),
                B.unsqueeze(0),
                C.unsqueeze(0),
                z.unsqueeze(0) if z is not None else None,
                out.unsqueeze(0),
                max_seqlen or 1,
            )
        batch = state_batch_indices.numel()
        tokens_per_batch = x.shape[0] // batch
        z_ckpt = None
        if z is not None:
            z_ckpt = z.view(batch, tokens_per_batch, *z.shape[1:])
        return (
            x.view(batch, tokens_per_batch, *x.shape[1:]),
            dt.view(batch, tokens_per_batch, *dt.shape[1:]),
            B.view(batch, tokens_per_batch, *B.shape[1:]),
            C.view(batch, tokens_per_batch, *C.shape[1:]),
            z_ckpt,
            out.view(batch, tokens_per_batch, *out.shape[1:]),
            tokens_per_batch,
        )

    @staticmethod
    def _update_checkpointing_trackers(
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        state_batch_indices: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int,
        max_window: int,
        pad_slot_id: int,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> None:
        block = 128
        n_slots = state_batch_indices.numel()
        _update_checkpointing_trackers_kernel[(triton.cdiv(n_slots, block),)](
            cache_buf_idx,
            prev_num_accepted_tokens,
            state_batch_indices,
            cu_seqlens,
            num_accepted_tokens,
            max_seqlen,
            max_window,
            pad_slot_id,
            n_slots,
            cu_seqlens is not None,
            num_accepted_tokens is not None,
            BLOCK=block,
        )

    @staticmethod
    def _make_old_cumAdt_cumulative(
        old_cumAdt: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        state_batch_indices: torch.Tensor,
        seq_len: int,
        max_window: int,
        pad_slot_id: int,
    ) -> None:
        """Run after `_checkpointing_kernel` (and before
        `_update_checkpointing_trackers`) to fold the new per-step
        `smem.cumAdt[0]` into a running cumulative cumAdt. See the kernel
        docstring for why this is needed.

        Reads `prev_num_accepted_tokens` and `cache_buf_idx` (which still
        hold their pre-kernel values at this point) to locate the slot that
        was just written.
        """
        if state_batch_indices is None or state_batch_indices.numel() == 0:
            return
        n_slots = state_batch_indices.numel()
        nheads = old_cumAdt.size(-2)
        block_h = max(16, min(nheads, 128))
        _make_old_cumAdt_cumulative_kernel[(n_slots, triton.cdiv(nheads, block_h))](
            old_cumAdt,
            state_batch_indices,
            cache_buf_idx,
            prev_num_accepted_tokens,
            old_cumAdt.stride(0),
            old_cumAdt.stride(1),
            old_cumAdt.stride(2),
            seq_len,
            max_window,
            nheads,
            pad_slot_id,
            n_slots,
            block_h,
        )

    @staticmethod
    def _reset_checkpointing_trackers(
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        state_batch_indices: torch.Tensor,
        pad_slot_id: int,
    ) -> None:
        block = 128
        n_slots = state_batch_indices.numel()
        _reset_checkpointing_trackers_kernel[(triton.cdiv(n_slots, block),)](
            cache_buf_idx,
            prev_num_accepted_tokens,
            state_batch_indices,
            pad_slot_id,
            n_slots,
            BLOCK=block,
        )

    @staticmethod
    def _copy_checkpointing_slots(
        tensors: tuple[torch.Tensor, ...],
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        pad_slot_id: int,
    ) -> None:
        block = 256
        n_slots = src_indices.numel()
        for tensor in tensors:
            slot_size = tensor[0].numel()
            _copy_checkpointing_slots_kernel[(n_slots, triton.cdiv(slot_size, block))](
                tensor,
                src_indices,
                dst_indices,
                slot_size,
                pad_slot_id,
                BLOCK=block,
            )


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """Initialize the global Mamba SSU backend.

    No-op if `kv_cache_config` contains no specs that call
    selective_state_update.
    """
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _mamba_ssu_backend

    backend = mamba_config.backend
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_cls = _BACKEND_REGISTRY[backend]
    if isinstance(_mamba_ssu_backend, backend_cls):
        return

    _mamba_ssu_backend = backend_cls(mamba_config)
    logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)


def get_mamba_ssu_backend() -> MambaSSUBackend:
    """Get the current Mamba SSU backend. Raises if not initialized."""
    if _mamba_ssu_backend is None:
        raise RuntimeError(
            "Mamba SSU backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_backend


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    is_blackwell: bool = False,
    old_x: torch.Tensor | None = None,
    old_B: torch.Tensor | None = None,
    old_dt: torch.Tensor | None = None,
    old_cumAdt: torch.Tensor | None = None,
    cache_buf_idx: torch.Tensor | None = None,
    prev_num_accepted_tokens: torch.Tensor | None = None,
    state_scales: torch.Tensor | None = None,
    spec_uniform_state_slots: bool = False,
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer).

    `spec_uniform_state_slots`: caller-asserted flag that the 2D
    `state_batch_indices` (shape `(batch, 1+num_spec)`) under MTP has
    identical entries across the spec-token dimension — true for
    mamba_cache_mode ∈ {"align", "none"}, false for "all". Enables the
    FlashInfer backend to checkpoint the in-place same-slot history.
    """
    get_mamba_ssu_backend()(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=null_block_id,
        out=out,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        is_blackwell=is_blackwell,
        old_x=old_x,
        old_B=old_B,
        old_dt=old_dt,
        old_cumAdt=old_cumAdt,
        cache_buf_idx=cache_buf_idx,
        prev_num_accepted_tokens=prev_num_accepted_tokens,
        state_scales=state_scales,
        spec_uniform_state_slots=spec_uniform_state_slots,
    )
