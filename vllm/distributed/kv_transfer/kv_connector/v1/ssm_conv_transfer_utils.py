# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba conv-state sub-projection decomposition for KV transfer.

With DS conv state layout (dim, state_len), sub-projections are
contiguous in memory.  Each D rank reads its slices via separate
RDMA transfers — no P-side permutation needed.

Supported model types:
  - Mamba2: conv = [x, B, C], temporal = (num_heads, head_dim)
  - GDN (Gated Delta Net): conv = [Q, K, V] (dim(Q)==dim(K)),
    temporal = (num_v_heads, v_dim, k_dim)
"""

import math
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import MambaSpec, MultiComponentSpec


@dataclass(frozen=True, slots=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of the conv sub-projections.

    Used to build TP-aware transfer descriptors.
    All fields are LOCAL to this engine's TP (already divided by TP size).

    DS memory layout within one page (contiguous):
      Mamba2: |-- x --|- B -|- C -|  (B == C)
      GDN:    |- Q -|- K -|-- V --|  (dim(Q)==dim(K), V may differ)
    """

    conv_rows: int  # conv_kernel - 1 (typically 3)
    # Per-rank column counts per sub-projection (x, B, C or Q, K, V).
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
    *,
    require_dim_first: bool = True,
) -> MambaConvSplitInfo:
    """Derive per-rank sub-projection byte sizes from a MambaSpec.

    Called once at init on both P and D.  Decomposes the conv dimension
    into its sub-projection parts based on the model type.

    Args:
        mamba_spec: MambaSpec whose shapes are:
            shapes[0] = conv state in the configured DS or SD layout.
            shapes[1] = temporal state (model-specific shape).
        require_dim_first: Require the DS layout used by NIXL 3-read transfer.

    Returns:
        MambaConvSplitInfo with per-rank sub-projection dims, conv_rows,
        conv_dtype_size, and ssm_sizes (conv_state_bytes, ssm_state_bytes).

    """
    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, f"Expected 2D conv state shape, got {conv_shape}"

    dim_first = is_conv_state_dim_first()
    if require_dim_first:
        assert dim_first, "3-read requires DS conv state layout"
    if dim_first:
        local_conv_dim, conv_rows = conv_shape
    else:
        conv_rows, local_conv_dim = conv_shape

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

    conv = next(iter(mamba_spec.components.values()))
    if not isinstance(conv, MultiComponentSpec):
        raise NotImplementedError(
            f"Conv split not supported for mamba_type={mamba_spec.mamba_type!r}"
        )
    local_proj_dims = tuple(part.num_heads for part in conv.components.values())

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
