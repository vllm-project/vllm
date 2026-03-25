# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba conv-state sub-projection decomposition for the 3-read transfer.

With DS conv state layout (dim, state_len), x/B/C sub-projections are
contiguous in memory.  Each D rank reads its x, B, C slices via 3
separate RDMA transfers — no P-side permutation needed.
"""

import math
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of x, B, C sub-projections in the Mamba conv state.

    Used by both P and D sides for NIXL descriptor registration.
    All fields are LOCAL to this engine's TP (already divided by TP size).

    DS memory layout within one page (contiguous in memory):
        |--- x (x_local * conv_rows) ---|- B (b_local * conv_rows) -|- C -|
    """

    conv_rows: int  # conv_kernel - 1 (typically 3)
    x_local: int  # intermediate_size / TP  (columns for x)
    b_local: int  # groups_ss / TP  (columns for B; C is same size)
    conv_dtype_size: int  # bytes per element (e.g. 2 for float16)

    @property
    def conv_dim_local(self) -> int:
        """Total conv columns per rank: x + B + C."""
        return self.x_local + 2 * self.b_local

    def x_bytes(self) -> int:
        """Byte size of the x sub-projection for one rank."""
        return self.x_local * self.conv_rows * self.conv_dtype_size

    def b_bytes(self) -> int:
        """Byte size of the B (or C) sub-projection for one rank."""
        return self.b_local * self.conv_rows * self.conv_dtype_size

    def local_conv_offsets(self) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of x, B, C within this engine's page.

        Used by both P and D for local descriptor registration.
        """
        xb = self.x_bytes()
        bb = self.b_bytes()
        return [(0, xb), (xb, bb), (xb + bb, bb)]

    def remote_conv_offsets(
        self, local_rank_offset: int, tp_ratio: int
    ) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of this D rank's x, B, C slice within
        one P page.

        Used by D side only, during remote descriptor registration.

        Args:
            local_rank_offset: which slice this D rank reads.
                tp_ratio > 0: tp_rank % tp_ratio (selects slice of P's page).
                tp_ratio < 0: always 0 (read P's full page).
            tp_ratio: effective ratio (>= 1 when D_TP > P_TP, 1 when
                P_TP > D_TP since each P rank is read in full).
        """
        xb = self.x_bytes()
        bb = self.b_bytes()
        xr = xb * tp_ratio  # full remote x section in bytes
        br = bb * tp_ratio  # full remote B section in bytes
        return [
            (local_rank_offset * xb, xb),
            (xr + local_rank_offset * bb, bb),
            (xr + br + local_rank_offset * bb, bb),
        ]


def derive_mamba_conv_split(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> MambaConvSplitInfo:
    """Derive per-rank x/B/C byte sizes from a MambaSpec.

    Called once at init on both P and D.  Decomposes the conv dimension
    (= intermediate_size + 2 * groups_ss) into its x, B, C parts.

    Args:
        mamba_spec: MambaSpec whose shapes are:
            shapes[0] = conv state: (conv_dim_local, conv_rows) in DS layout.
            shapes[1] = SSM temporal: (local_num_heads, head_dim).
        local_tp: this engine's tensor-parallel size.

    Returns:
        MambaConvSplitInfo with per-rank x_local, b_local, conv_rows, and
        conv_dtype_size.
    """
    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, f"Expected 2D conv state shape, got {conv_shape}"

    # NOTE (ZhanqiuHu): 3-read requires DS layout, which is already asserted
    # in nixl_connector __init__.  Use it directly instead of heuristic detection.
    assert is_conv_state_dim_first(), "3-read requires DS conv state layout"
    local_conv_dim = conv_shape[0]  # DS: (conv_dim_local, conv_rows)
    conv_rows = conv_shape[1]

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
        f"Conv dim ({local_conv_dim}*tp={local_tp}) doesn't decompose into "
        f"intermediate_size={intermediate_size} + 2*groups_ss. "
        f"remainder={remainder}"
    )
    groups_ss = remainder // 2

    conv_dtype_size = torch.tensor(
        [],
        dtype=mamba_spec.dtypes[0],  # type: ignore[misc]
    ).element_size()

    # Divide by TP to get per-rank column counts.
    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        x_local=intermediate_size // local_tp,
        b_local=groups_ss // local_tp,
        conv_dtype_size=conv_dtype_size,
    )


def compute_mamba_phys_ratio(ssm_sizes: tuple[int, ...], block_len: int) -> int:
    """Physical kernel blocks per logical mamba block.

    Called during P/D handshake to learn the remote engine's ratio.
    Under HMA, P and D may pad differently, so this can differ per engine.

    Args:
        ssm_sizes: (conv_state_bytes, ssm_state_bytes) from NixlAgentMetadata.
        block_len: the engine's block_len in bytes.
    """
    return math.ceil((ssm_sizes[0] + ssm_sizes[1]) / block_len)
