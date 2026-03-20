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

from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class ConvDecomp:
    """Per-rank sizes of the x, B, C sub-projections in the Mamba conv state.

    Fields are LOCAL to this engine's TP (already divided by TP size).
    """

    conv_rows: int  # conv_kernel - 1 (e.g. 3)
    x_local: int  # intermediate_size / TP  (columns for x projection)
    b_local: int  # groups_ss / TP  (columns for B; C is identical)
    conv_dtype_size: int  # torch element_size() of conv state dtype

    @property
    def conv_dim_local(self) -> int:
        return self.x_local + 2 * self.b_local

    def x_bytes(self) -> int:
        return self.x_local * self.conv_rows * self.conv_dtype_size

    def b_bytes(self) -> int:
        return self.b_local * self.conv_rows * self.conv_dtype_size

    def local_conv_offsets(self) -> list[tuple[int, int]]:
        """Return (byte_offset, byte_size) of x, B, C within one local page."""
        xb = self.x_bytes()
        bb = self.b_bytes()
        return [(0, xb), (xb, bb), (xb + bb, bb)]

    def remote_conv_offsets(
        self, local_rank_offset: int, tp_ratio: int
    ) -> list[tuple[int, int]]:
        """Return (byte_offset, byte_size) of this rank's x, B, C slice
        within one remote page whose conv dim is tp_ratio times larger."""
        xb = self.x_bytes()
        bb = self.b_bytes()
        xr = xb * tp_ratio  # full remote x section
        br = bb * tp_ratio  # full remote B section
        return [
            (local_rank_offset * xb, xb),
            (xr + local_rank_offset * bb, bb),
            (xr + br + local_rank_offset * bb, bb),
        ]


def derive_conv_decomp(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> ConvDecomp:
    """Extract per-rank x/B/C column counts from a MambaSpec.

    Args:
        mamba_spec: contains conv shape ``(conv_dim_local, conv_rows)`` or
            ``(conv_rows, conv_dim_local)`` and temporal shape
            ``(local_num_heads, head_dim)``.
        local_tp: this engine's tensor-parallel size.

    Returns:
        ConvDecomp with x_local, b_local, conv_rows, and conv_dtype_size.
    """
    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, f"Expected 2D conv state shape, got {conv_shape}"

    # DS layout: (conv_dim_local, conv_rows), SD layout: (conv_rows, conv_dim_local).
    # We need to identify which axis is conv_rows (small, e.g. 3) and which is dim.
    # SSM temporal shape[1] gives head_dim, shape[0] gives local_num_heads.
    head_dim = mamba_spec.shapes[1][1]
    local_num_heads = mamba_spec.shapes[1][0]
    intermediate_size = local_num_heads * local_tp * head_dim

    # Determine conv_dim_local and conv_rows from the 2D shape.
    # conv_dim_local * local_tp must be >= intermediate_size (because it's
    # intermediate_size + 2*groups_ss).
    if conv_shape[0] * local_tp >= intermediate_size:
        # DS layout: (conv_dim_local, conv_rows)
        local_conv_dim = conv_shape[0]
        conv_rows = conv_shape[1]
    else:
        # SD layout: (conv_rows, conv_dim_local)
        conv_rows = conv_shape[0]
        local_conv_dim = conv_shape[1]

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

    return ConvDecomp(
        conv_rows=conv_rows,
        x_local=intermediate_size // local_tp,
        b_local=groups_ss // local_tp,
        conv_dtype_size=conv_dtype_size,
    )


def compute_mamba_phys_ratio(ssm_sizes: tuple[int, ...], block_len: int) -> int:
    """Return ceil((conv_bytes + ssm_bytes) / block_len).

    This is how many physical kernel blocks one logical mamba block spans,
    which can differ between P and D engines under HMA padding.
    """
    return math.ceil((ssm_sizes[0] + ssm_sizes[1]) / block_len)
