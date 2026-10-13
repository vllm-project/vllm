# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba conv-state sub-projection decomposition for NIXL transfer.

With DS conv state layout (dim, state_len), sub-projections are
contiguous in memory.  Each D rank reads its slices via separate
RDMA transfers — no P-side permutation needed.

Supported model types:
  - Mamba1: conv = [x], temporal = (intermediate_size, state_size)
  - Mamba2: conv = [x, B, C], temporal = (num_heads, head_dim)
  - GDN (Gated Delta Net): conv = [Q, K, V] (dim(Q)==dim(K)),
    temporal = (num_v_heads, v_dim, k_dim)
"""

import math
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of the conv sub-projections.

    Used by both P and D sides for NIXL descriptor registration.
    Sizes are LOCAL to this engine's TP (already divided by TP size).

    DS memory layout within one page (contiguous):
      Mamba1: |---- x ----|  (single sub-projection, no decomposition)
      Mamba2: |-- x --|- B -|- C -|  (B == C)
      GDN:    |- Q -|- K -|-- V --|  (dim(Q)==dim(K), V may differ)
    """

    conv_rows: int  # conv_kernel - 1 (typically 3)
    # Per-rank column counts per sub-projection:
    # 1 entry for Mamba1, 3 for Mamba2/GDN.
    local_proj_dims: tuple[int, ...]
    conv_dtype_size: int  # bytes per element (e.g. 2 for float16)
    ssm_sizes: tuple[int, int]  # (conv_state_bytes, ssm_state_bytes)
    local_tp: int = 1
    # (intermediate_size, n_groups, num_heads, state_size) of a Mamba2 layer
    # that declares its TP split; gives its layout for any TP size.
    mamba2_dims: tuple[int, int, int, int] | None = None

    @property
    def local_conv_dim(self) -> int:
        """Total conv columns per rank."""
        return sum(self.local_proj_dims)

    @property
    def proj_bytes(self) -> tuple[int, ...]:
        """Byte sizes of the sub-projections for one rank."""
        row_bytes = self.conv_rows * self.conv_dtype_size
        return tuple(d * row_bytes for d in self.local_proj_dims)

    @property
    def local_conv_offsets(self) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of each sub-projection within this
        engine's page.

        Used by both P and D for local descriptor registration.
        """
        offsets: list[tuple[int, int]] = []
        offset = 0
        for size in self.proj_bytes:
            offsets.append((offset, size))
            offset += size
        return offsets

    def proj_layout(self, tp_size: int) -> tuple[tuple[int, int], ...]:
        """(columns, replicas) of each sub-projection on a rank of an engine
        with TP size ``tp_size`` (see ``MambaStateSplit``)."""
        if self.mamba2_dims is not None:
            intermediate_size, n_groups, num_heads, state_size = self.mamba2_dims
            splits = MambaStateShapeCalculator.mamba2_state_splits(
                tp_world_size=tp_size,
                intermediate_size=intermediate_size,
                n_groups=n_groups,
                num_heads=num_heads,
                state_size=state_size,
            )
            if splits is None:
                raise NotImplementedError(
                    f"Mamba2 conv state with n_groups={n_groups} and "
                    f"num_heads={num_heads} cannot be split across "
                    f"TP={tp_size}."
                )
            return splits[0].segments
        return tuple(
            (dim * self.local_tp // tp_size, 1) for dim in self.local_proj_dims
        )

    def _remote_tp_size(self, tp_ratio: int) -> int:
        if tp_ratio >= 1:
            return self.local_tp // tp_ratio
        return self.local_tp * -tp_ratio

    def remote_conv_offsets(
        self, tp_rank: int, tp_ratio: int, remote_tp_rank: int
    ) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of this D rank's sub-projection slices
        within one P page.

        Used by D side only, during remote descriptor registration.

        Args:
            tp_rank: this D rank.
            tp_ratio: signed TP ratio.
                >= 1:  D_TP >= P_TP — D reads its slice of one P page.
                < 0:   P_TP > D_TP — D reads one chunk of its page from
                       each of |tp_ratio| P ranks.
            remote_tp_rank: the P rank read.

        """
        if tp_ratio >= 1:
            num_chunks, chunk_idx = 1, 0
        else:
            num_chunks = -tp_ratio
            chunk_idx = remote_tp_rank - tp_rank * num_chunks
            assert 0 <= chunk_idx < num_chunks
        row_bytes = self.conv_rows * self.conv_dtype_size

        offsets: list[tuple[int, int]] = []
        remote_base = 0
        for (local_dim, local_reps), (remote_dim, remote_reps) in zip(
            self.proj_layout(self.local_tp),
            self.proj_layout(self._remote_tp_size(tp_ratio)),
            strict=True,
        ):
            assert local_dim % num_chunks == 0, (
                f"{local_dim} columns don't split into {num_chunks} chunks."
            )
            chunk = local_dim // num_chunks
            # Global column of the chunk minus that of the remote shard.
            start = (tp_rank // local_reps) * local_dim + chunk_idx * chunk
            start -= (remote_tp_rank // remote_reps) * remote_dim
            assert start >= 0 and start + chunk <= remote_dim, (
                f"D rank {tp_rank} slice [{start}, {start + chunk}) is not "
                f"within P rank {remote_tp_rank}'s {remote_dim} columns."
            )
            offsets.append(((remote_base + start) * row_bytes, chunk * row_bytes))
            remote_base += remote_dim
        return offsets

    def remote_conv_bytes(self, tp_ratio: int) -> int:
        """Conv state bytes of one remote page."""
        layout = self.proj_layout(self._remote_tp_size(tp_ratio))
        return sum(dim for dim, _ in layout) * self.conv_rows * self.conv_dtype_size


def derive_mamba_conv_split(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> MambaConvSplitInfo:
    """Derive per-rank sub-projection byte sizes from a MambaSpec.

    Called once at init on both P and D.  Decomposes the conv dimension
    into its sub-projection parts based on the model type.

    Args:
        mamba_spec: MambaSpec whose shapes are:
            shapes[0] = conv state: (conv_dim_local, conv_rows) in DS layout.
            shapes[1] = temporal state (model-specific shape).
        local_tp: this engine's tensor-parallel size.

    Returns:
        MambaConvSplitInfo with per-rank sub-projection dims, conv_rows,
        conv_dtype_size, and ssm_sizes (conv_state_bytes, ssm_state_bytes).

    """
    _supported = (
        MambaAttentionBackendEnum.MAMBA1,
        MambaAttentionBackendEnum.MAMBA2,
        MambaAttentionBackendEnum.GDN_ATTN,
    )
    if mamba_spec.mamba_type not in _supported:
        raise NotImplementedError(
            f"Conv transfer only supports Mamba1, Mamba2 and GDN models, "
            f"got mamba_type={mamba_spec.mamba_type!r}."
        )

    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, f"Expected 2D conv state shape, got {conv_shape}"

    # NOTE (ZhanqiuHu): 3-read requires DS layout, which is already asserted
    # in nixl worker __init__.  Use it directly instead of heuristic detection.
    assert is_conv_state_dim_first(), "3-read requires DS conv state layout"
    local_conv_dim = conv_shape[0]  # DS: (conv_dim_local, conv_rows)
    conv_rows = conv_shape[1]

    conv_dtype_size = torch.tensor(
        [],
        dtype=mamba_spec.dtypes[0],  # type: ignore[misc]
    ).element_size()
    ssm_dtype_size = torch.tensor(
        [],
        dtype=mamba_spec.dtypes[1],  # type: ignore[misc]
    ).element_size()
    conv_state_bytes = torch.Size(mamba_spec.shapes[0]).numel() * conv_dtype_size
    ssm_state_bytes = torch.Size(mamba_spec.shapes[1]).numel() * ssm_dtype_size

    local_proj_dims: tuple[int, ...]
    mamba2_dims = None
    if mamba_spec.mamba_type == MambaAttentionBackendEnum.MAMBA1:
        # Mamba1 conv state holds only x (no B/C), so it's a single
        # contiguous TP shard with no sub-projection decomposition.
        temporal_shape = mamba_spec.shapes[1]
        assert temporal_shape[0] == local_conv_dim, (
            f"Mamba1 temporal state dim ({temporal_shape[0]}) doesn't match "
            f"conv dim ({local_conv_dim}); both should be "
            f"intermediate_size/TP."
        )
        local_proj_dims = (local_conv_dim,)
    elif mamba_spec.mamba_type == MambaAttentionBackendEnum.MAMBA2:
        # NOTE (ZhanqiuHu): intermediate_size (= global x dim) is not stored
        # in MambaSpec, so we reconstruct it from the SSM temporal state shape:
        #   shapes[1] = (local_num_heads, head_dim), already divided by TP.
        head_dim = mamba_spec.shapes[1][1]
        local_num_heads = mamba_spec.shapes[1][0]
        intermediate_size = local_num_heads * local_tp * head_dim

        # NOTE (ZhanqiuHu): global conv dim = intermediate_size + 2 * groups_ss,
        # where groups_ss is the B (= C) dimension.  B and C are always the same
        # size, so we recover groups_ss from the remainder after subtracting x.
        remainder = local_conv_dim * local_tp - intermediate_size
        assert remainder > 0 and remainder % 2 == 0, (
            f"Conv dim ({local_conv_dim}*tp={local_tp}) doesn't decompose "
            f"into intermediate_size={intermediate_size} + 2*groups_ss. "
            f"remainder={remainder}"
        )
        groups_ss = remainder // 2

        # Divide by TP to get per-rank column counts.
        x_local = intermediate_size // local_tp
        b_local = groups_ss // local_tp
        local_proj_dims = (x_local, b_local, b_local)

        if mamba_spec.state_splits is not None:
            conv_split = mamba_spec.state_splits[0]
            assert tuple(d for d, _ in conv_split.segments) == local_proj_dims, (
                f"Declared conv split {conv_split.segments} doesn't match "
                f"the conv state shape dims {local_proj_dims}."
            )
            b_replicas = conv_split.segments[1][1]
            state_size = mamba_spec.shapes[1][2]
            mamba2_dims = (
                intermediate_size,
                b_local * local_tp // b_replicas // state_size,
                local_num_heads * local_tp,
                state_size,
            )
    elif mamba_spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN:
        # GDN: conv = [Q, K, V] where dim(Q) == dim(K) == key_dim.
        # conv_dim = key_dim*2 + value_dim (all global, divided by TP).
        # Temporal state shape is (num_v_heads/TP, head_v_dim, head_k_dim).
        temporal_shape = mamba_spec.shapes[1]
        num_v_heads_local = temporal_shape[0]
        head_v_dim = temporal_shape[1]
        value_dim_local = num_v_heads_local * head_v_dim

        remainder = local_conv_dim - value_dim_local
        assert remainder > 0 and remainder % 2 == 0, (
            f"GDN conv dim ({local_conv_dim}) doesn't decompose into "
            f"2*key_dim_local + value_dim_local={value_dim_local}. "
            f"remainder={remainder}"
        )
        key_dim_local = remainder // 2
        local_proj_dims = (key_dim_local, key_dim_local, value_dim_local)
    else:
        raise NotImplementedError(
            f"Conv split not supported for mamba_type={mamba_spec.mamba_type!r}"
        )

    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        local_proj_dims=local_proj_dims,
        conv_dtype_size=conv_dtype_size,
        ssm_sizes=(conv_state_bytes, ssm_state_bytes),
        local_tp=local_tp,
        mamba2_dims=mamba2_dims,
    )


def compute_physical_blocks_per_logical(
    ssm_sizes: tuple[int, ...], block_len: int
) -> int:
    """Derive _physical_blocks_per_logical_kv_block from remote metadata.

    The remote engine's ratio is not sent directly in the handshake, so we
    reconstruct it: total mamba state per logical block / block_len.

    Args:
        ssm_sizes: (conv_state_bytes, ssm_state_bytes) from NixlAgentMetadata.
        block_len: the engine's block_len in bytes (from block_lens[0]).

    """
    return math.ceil((ssm_sizes[0] + ssm_sizes[1]) / block_len)
