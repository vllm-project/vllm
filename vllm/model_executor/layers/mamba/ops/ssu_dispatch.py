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
    fixed_seq_len: tl.constexpr,
    max_window: tl.constexpr,
    pad_slot_id: tl.constexpr,
    n_slots: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
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
    prev = tl.load(prev_num_accepted_tokens + slots, mask=valid, other=0)
    must_checkpoint = prev + seq_lens > max_window
    old_buf = tl.load(cache_buf_idx + slots, mask=valid, other=0)
    new_buf = tl.where(must_checkpoint, 1 - old_buf, old_buf)
    new_prev = tl.where(must_checkpoint, seq_lens, prev + seq_lens)
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
    ) -> None:
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
    ) -> None:
        rand_seed = (
            torch.randint(0, 2**32, (1,), dtype=torch.int64, device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )

        checkpointing_args = (
            old_x,
            old_B,
            old_dt,
            old_cumAdt,
            cache_buf_idx,
            prev_num_accepted_tokens,
        )
        can_checkpoint = (
            num_accepted_tokens is None
            and all(arg is not None for arg in checkpointing_args)
            and self._checkpointing_state_indices(state_batch_indices) is not None
            and state.dtype
            in (torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn)
        )
        import os as _os
        if _os.environ.get("SSU_DBG", "0") == "1":
            _ckpt_none = [a is None for a in checkpointing_args]
            _sbi_ok = self._checkpointing_state_indices(state_batch_indices) is not None
            _sbi_shape = (
                tuple(state_batch_indices.shape) if state_batch_indices is not None else None
            )
            print(
                f"[SSU_DBG] can_checkpoint={can_checkpoint} "
                f"num_accepted_tokens_None={num_accepted_tokens is None} "
                f"ckpt_args_None={_ckpt_none} "
                f"sbi_ok={_sbi_ok} sbi_shape={_sbi_shape} "
                f"state.dtype={state.dtype} state.shape={tuple(state.shape)} "
                f"state_scales_None={state_scales is None} "
                f"state_scales_shape={(tuple(state_scales.shape) if state_scales is not None else None)} "
                f"state_scales_dtype={(state_scales.dtype if state_scales is not None else None)}",
                flush=True,
            )
        if can_checkpoint:
            assert old_x is not None
            assert old_B is not None
            assert old_dt is not None
            assert old_cumAdt is not None
            assert cache_buf_idx is not None
            assert prev_num_accepted_tokens is not None
            state_indices = self._checkpointing_state_indices(state_batch_indices)
            assert state_indices is not None
            kernel_state_indices = state_indices
            dst_indices = self._checkpointing_state_indices(dst_state_batch_indices)
            if dst_indices is not None and dst_indices.numel() == state_indices.numel():
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
            if ckpt_cu_seqlens is None and kernel_state_indices.numel() > 1:
                for start in range(kernel_state_indices.numel()):
                    end = start + 1
                    chunk_rand_seed = (
                        torch.randint(
                            0,
                            2**32,
                            (1,),
                            dtype=torch.int64,
                            device=state.device,
                        )
                        if self._mamba_config.enable_stochastic_rounding
                        else None
                    )
                    chunk_indices = kernel_state_indices[start:end]
                    self._checkpointing_kernel(
                        state,
                        old_x,
                        old_B,
                        old_dt,
                        old_cumAdt,
                        cache_buf_idx,
                        prev_num_accepted_tokens,
                        x_ckpt[start:end],
                        dt_ckpt[start:end],
                        A,
                        B_ckpt[start:end],
                        C_ckpt[start:end],
                        out_ckpt[start:end],
                        D=D,
                        z=z_ckpt[start:end] if z_ckpt is not None else None,
                        dt_bias=dt_bias,
                        dt_softplus=dt_softplus,
                        state_batch_indices=chunk_indices,
                        pad_slot_id=null_block_id,
                        rand_seed=chunk_rand_seed,
                        philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds
                        or 10,
                        cu_seqlens=None,
                        max_seqlen=None,
                        state_scale=state_scales,
                    )
                    self._update_checkpointing_trackers(
                        cache_buf_idx,
                        prev_num_accepted_tokens,
                        chunk_indices,
                        None,
                        ckpt_max_seqlen,
                        old_x.size(1),
                        null_block_id,
                    )
                return
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
                philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds
                or 10,
                cu_seqlens=ckpt_cu_seqlens,
                max_seqlen=kernel_max_seqlen,
                state_scale=state_scales,
            )
            self._update_checkpointing_trackers(
                cache_buf_idx,
                prev_num_accepted_tokens,
                kernel_state_indices,
                ckpt_cu_seqlens,
                ckpt_max_seqlen,
                old_x.size(1),
                null_block_id,
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
    ) -> torch.Tensor | None:
        if state_batch_indices is None:
            return None
        if state_batch_indices.dim() == 1:
            return state_batch_indices.to(torch.int32).contiguous()
        if state_batch_indices.dim() == 2 and state_batch_indices.size(1) == 1:
            return state_batch_indices[:, 0].to(torch.int32).contiguous()
        return None

    @staticmethod
    def _checkpointing_cu_seqlens(
        cu_seqlens: torch.Tensor | None,
        x: torch.Tensor,
        state_batch_indices: torch.Tensor,
        max_seqlen: int | None,
    ) -> torch.Tensor | None:
        del max_seqlen
        if cu_seqlens is not None and x.shape[0] == state_batch_indices.numel():
            return None
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
    ) -> None:
        block = 128
        n_slots = state_batch_indices.numel()
        _update_checkpointing_trackers_kernel[(triton.cdiv(n_slots, block),)](
            cache_buf_idx,
            prev_num_accepted_tokens,
            state_batch_indices,
            cu_seqlens,
            max_seqlen,
            max_window,
            pad_slot_id,
            n_slots,
            cu_seqlens is not None,
            BLOCK=block,
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
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer).
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
    )
