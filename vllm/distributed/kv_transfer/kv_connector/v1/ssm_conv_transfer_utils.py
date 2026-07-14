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

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of the conv sub-projections.

    Used by both P and D sides for NIXL descriptor registration.
    All fields are LOCAL to this engine's TP (already divided by TP size).

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

    def remote_conv_offsets(
        self, local_rank_offset: int, tp_ratio: int
    ) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of this D rank's sub-projection slices
        within one P page.

        Used by D side only, during remote descriptor registration.

        Args:
            local_rank_offset: which slice this D rank reads.
            tp_ratio: signed TP ratio.
                >= 1:  D_TP >= P_TP — P page is larger, D reads its slice.
                < 0:   P_TP > D_TP — P pages are smaller, D reads entire
                       P page.  Local dims are scaled down by |tp_ratio|
                       to get P-sized offsets.
        """
        offsets: list[tuple[int, int]] = []
        if tp_ratio >= 1:
            remote_base = 0
            for size in self.proj_bytes:
                offsets.append((remote_base + local_rank_offset * size, size))
                remote_base += size * tp_ratio
        else:
            # NOTE (ZhanqiuHu): tp_ratio < 0 means P_TP > D_TP, so P pages
            # are smaller than D's. Local dims are D-sized, but we need
            # P-sized offsets. Scale down by |tp_ratio|.
            abs_ratio = -tp_ratio
            remote_base = 0
            for size in self.proj_bytes:
                remote_size = size // abs_ratio
                offsets.append((remote_base, remote_size))
                remote_base += remote_size
        return offsets


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
