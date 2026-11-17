# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Mamba2 SSD prefill scan's kernel interface."""

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

import torch

from vllm.model_executor.layers.linear_attention.kernel import LinearAttnKernel
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

# Max length of the SSD autotune warmup, in chunks.
_WARMUP_MAX_CHUNKS = 32


class SSDLayerParams(Protocol):
    """The layer parameters an SSD kernel reads."""

    A: torch.Tensor
    D: torch.Tensor
    dt_bias: torch.Tensor


@dataclass(frozen=True, slots=True)
class SSDWarmupKey:
    """What a layer's SSD kernels compile for."""

    num_heads: int
    head_dim: int
    state_size: int
    num_groups: int
    chunk_size: int
    dtype: torch.dtype
    state_dtype: torch.dtype
    state_indices_stride: int
    max_num_seqs: int
    max_num_batched_tokens: int
    exports_checkpoints: bool


class MambaSSDKernel(LinearAttnKernel):
    """The Mamba2 SSD prefill scan, including its state cache reads and writes."""

    @classmethod
    def register_warmup(
        cls,
        *,
        num_heads: int,
        head_dim: int,
        state_size: int,
        num_groups: int,
        chunk_size: int,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        state_indices_stride: int,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        exports_checkpoints: bool,
    ) -> None:
        """Register the JIT warmup of this kernel's launches for a layer of this
        geometry (per TP rank)."""
        cls._register_warmups(
            SSDWarmupKey(
                num_heads=num_heads,
                head_dim=head_dim,
                state_size=state_size,
                num_groups=num_groups,
                chunk_size=chunk_size,
                dtype=dtype,
                state_dtype=state_dtype,
                state_indices_stride=state_indices_stride,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                exports_checkpoints=exports_checkpoints,
            )
        )

    @classmethod
    def _register_warmups(cls, key: SSDWarmupKey) -> None:
        """Register the state gather and scatter, which every GPU kernel uses."""
        _GATHER_WARMUP.register_warmup(key)
        scatter_states.register_warmup(
            state_shape=(key.num_heads, key.head_dim, key.state_size),
            state_dtype=key.state_dtype,
            indices_dtype=torch.int32,
            max_num_tokens=key.max_num_seqs,
            indices_stride=key.state_indices_stride,
        )

    @classmethod
    def prepare_metadata(
        cls,
        metadata: "Mamba2AttentionMetadata",
        kv_cache_spec: MambaSpec,
        num_computed_tokens_p_cpu: torch.Tensor,
        query_start_loc_p_cpu: torch.Tensor,
    ) -> "Mamba2AttentionMetadata":
        """Add the prefill tensors this kernel reads. The default is the chunk
        layout the Triton and CPU kernels read."""
        from vllm.v1.attention.backends.mamba_attn import (
            BaseMambaAttentionMetadataBuilder,
        )

        assert metadata.query_start_loc_p is not None
        checkpoint = metadata.checkpoint_meta
        # The host offsets place a chunk boundary on each checkpoint.
        cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p, checkpoint_chunk_idx = (
            BaseMambaAttentionMetadataBuilder._build_chunk_metadata_tensors(
                metadata.chunk_size,
                num_computed_tokens_p_cpu,
                query_start_loc_p_cpu,
                metadata.query_start_loc_p.device,
                checkpoint.offsets if checkpoint is not None else None,
            )
        )
        return replace(
            metadata,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            seq_idx_p=seq_idx_p,
            last_chunk_indices_p=last_chunk_indices_p,
            checkpoint_chunk_idx=checkpoint_chunk_idx,
        )

    @classmethod
    def read_initial_states(
        cls, ssm_state: torch.Tensor, metadata: "Mamba2AttentionMetadata"
    ) -> torch.Tensor | None:
        """Each prefill's cached state, zeros where it has none."""
        if metadata.has_initial_states_p is None or not metadata.prep_initial_states:
            return metadata.zero_initial_states_p
        assert metadata.state_indices_tensor_p is not None
        return cls.gather_states(
            ssm_state, metadata.state_indices_tensor_p, metadata.has_initial_states_p
        )

    @classmethod
    def gather_states(
        cls,
        ssm_state: torch.Tensor,
        indices: torch.Tensor,
        has_initial_states: torch.Tensor,
    ) -> torch.Tensor:
        return gather_initial_states(ssm_state, indices, has_initial_states)

    @classmethod
    def write_final_states(
        cls,
        ssm_state: torch.Tensor,
        final_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        scatter_states(ssm_state, final_states, indices)

    @classmethod
    def forward(
        cls,
        layer: SSDLayerParams,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
    ) -> torch.Tensor | None:
        """Scan the prefills from their cached states into `out`, and cache
        their final states.

        Returns:
            The checkpoint states, indexed by `metadata.checkpoint_chunk_idx`,
            or None when no checkpoints are exported.

        """
        assert metadata.state_indices_tensor_p is not None
        final_states, checkpoint_states = cls._scan(
            layer,
            x,
            dt,
            B,
            C,
            out,
            ssm_state,
            metadata,
            cls.read_initial_states(ssm_state, metadata),
        )
        if final_states is not None:
            cls.write_final_states(
                ssm_state, final_states, metadata.state_indices_tensor_p
            )
        return checkpoint_states

    @classmethod
    def _scan(
        cls,
        layer: SSDLayerParams,
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: "Mamba2AttentionMetadata",
        initial_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the final states, or None if already written into
        `ssm_state`, and the checkpoint states, or None when none are
        exported."""
        raise NotImplementedError


@dataclass(slots=True)
class _WarmupParams:
    A: torch.Tensor
    D: torch.Tensor
    dt_bias: torch.Tensor


class _GatherWarmup(VllmJitKernel[SSDWarmupKey]):
    """Compiles the initial-state gather that the GPU kernels read through."""

    def get_warmup_keys(self, key: SSDWarmupKey) -> list[SSDWarmupKey]:
        return [key]

    @torch.inference_mode()
    def compile(self, compile_key: SSDWarmupKey) -> None:
        device = current_platform.device_type
        stride = compile_key.state_indices_stride
        state = torch.zeros(
            1,
            compile_key.num_heads,
            compile_key.head_dim,
            compile_key.state_size,
            device=device,
            dtype=compile_key.state_dtype,
        )
        # One state index, strided like a block-table column.
        gather_initial_states(
            state,
            torch.zeros(stride, device=device, dtype=torch.int32)[::stride],
            torch.ones(1, device=device, dtype=torch.bool),
        )


class SSDScanWarmup(VllmJitKernel[SSDWarmupKey]):
    """Compiles (and for Triton, autotunes) one kernel's scan variants for each
    geometry; the state gather and scatter are warmed on their own."""

    def __init__(self, kernel: type[MambaSSDKernel]) -> None:
        super().__init__()
        self.kernel = kernel

    def get_warmup_keys(self, key: SSDWarmupKey) -> list[SSDWarmupKey]:
        return [key]

    @torch.inference_mode()
    def compile(self, compile_key: SSDWarmupKey) -> None:
        from vllm.model_executor.layers.mamba.checkpoint import (
            MambaPrefillCheckpointMetadata,
        )
        from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

        device = current_platform.device_type
        dtype, state_dtype = compile_key.dtype, compile_key.state_dtype
        heads, dim = compile_key.num_heads, compile_key.head_dim
        groups, dstate = compile_key.num_groups, compile_key.state_size
        # Autotune keys have no sequence-length term, so the tile picked here
        # serves every prefill: pick it on a prefill-sized sequence.
        chunk_size = compile_key.chunk_size
        tokens = chunk_size * max(
            1, min(_WARMUP_MAX_CHUNKS, compile_key.max_num_batched_tokens // chunk_size)
        )
        params = _WarmupParams(
            A=-torch.rand(heads, device=device),
            D=torch.randn(heads, device=device, dtype=dtype),
            dt_bias=torch.randn(heads, device=device, dtype=dtype),
        )
        x = torch.randn(tokens, heads, dim, device=device, dtype=dtype)
        dt = torch.randn(tokens, heads, device=device, dtype=dtype)
        B = torch.randn(tokens, groups, dstate, device=device, dtype=dtype)
        C = torch.randn_like(B)
        out = torch.empty_like(x)
        state = torch.zeros(1, heads, dim, dstate, device=device, dtype=state_dtype)
        # Triton writes final states through a block-table column, as in serving.
        stride = compile_key.state_indices_stride
        slots = torch.zeros(stride, device=device, dtype=torch.int32)[::stride]
        spec = MambaSpec(
            block_size=chunk_size,
            shapes=((1, 1), (heads, dim, dstate)),
            dtypes=(dtype, state_dtype),
        )
        checkpoint = MambaPrefillCheckpointMetadata(
            torch.tensor([tokens // 2], dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            offsets=[tokens // 2],
        )
        initial_states = torch.randn(
            1, heads, dim, dstate, device=device, dtype=state_dtype
        )
        # Initial states and checkpoint export each select compiled variants.
        for has_initial_state, exports_checkpoint in itertools.product(
            (False, True),
            (False, True) if compile_key.exports_checkpoints else (False,),
        ):
            metadata = self.kernel.prepare_metadata(
                Mamba2AttentionMetadata(
                    num_prefills=1,
                    num_prefill_tokens=tokens,
                    num_decodes=0,
                    num_decode_tokens=0,
                    num_reqs=1,
                    has_initial_states_p=None,
                    query_start_loc_p=torch.tensor(
                        [0, tokens], device=device, dtype=torch.int32
                    ),
                    state_indices_tensor_p=slots,
                    state_indices_tensor_d=None,
                    query_start_loc_d=None,
                    num_accepted_tokens=None,
                    seq_lens=torch.tensor([tokens], device=device, dtype=torch.int32),
                    chunk_size=chunk_size,
                    checkpoint_meta=checkpoint if exports_checkpoint else None,
                ),
                spec,
                torch.zeros(1, dtype=torch.int32),
                torch.tensor([0, tokens], dtype=torch.int32),
            )
            self.kernel._scan(
                params,
                x,
                dt,
                B,
                C,
                out,
                state,
                metadata,
                initial_states if has_initial_state else metadata.zero_initial_states_p,
            )


_GATHER_WARMUP = _GatherWarmup()
