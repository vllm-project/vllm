# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba conv-state chunk-interleave utilities for heterogeneous TP.

When prefill (P) and decode (D) run at different TP sizes, P's conv cache
layout has sub-projections (x, B, C) interleaved in a way that does NOT
align with D's TP sharding boundaries.  A chunk-based permutation on P
makes each D rank's shard contiguous so it can be read in a single RDMA
transfer via rank_offset.

The permutation groups columns by GCD-reduced x:B:C ratios and transposes
within each group (rows → columns) so that row-major memory order matches
the contiguous byte range each D rank reads.  This is NOT a simple
reshape-transpose-reshape; the three sub-projections have different widths.
"""

import math
from dataclasses import dataclass

import torch

from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class ConvPermParams:
    """Parameters that describe how the Mamba conv dim decomposes into x/B/C.

    Derived once from MambaSpec at init time and reused for building
    permutation indices and in-place reordering.
    """

    conv_rows: int  # conv_kernel - 1 (small, e.g. 3)
    intermediate_size: int  # full (unsharded) intermediate dim
    groups_ss: int  # number of B/C group columns (full, unsharded)
    local_tp: int  # this engine's TP degree
    conv_dtype_size: int  # element_size() of conv state dtype

    @property
    def x_local(self) -> int:
        """Per-rank x sub-projection width."""
        return self.intermediate_size // self.local_tp

    @property
    def b_local(self) -> int:
        """Per-rank B (or C) sub-projection width."""
        return self.groups_ss // self.local_tp


def derive_conv_perm_params(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> ConvPermParams:
    """Derive conv permutation parameters from a MambaSpec.

    Raises AssertionError if the conv dim doesn't decompose into
    intermediate_size + 2 * groups_ss (n_groups * d_state).
    """
    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, (
        f"Expected 2D conv state shape (state_len, dim), got {conv_shape}. "
        f"If using DS conv state layout (dim, state_len), the chunk "
        f"permutation logic needs updating."
    )
    conv_rows = conv_shape[0]
    local_conv_dim = conv_shape[1]

    head_dim = mamba_spec.shapes[1][1]
    local_num_heads = mamba_spec.shapes[1][0]
    intermediate_size = local_num_heads * local_tp * head_dim
    remainder = local_conv_dim * local_tp - intermediate_size
    assert remainder > 0 and remainder % 2 == 0, (
        f"Conv state dim ({local_conv_dim}*tp={local_tp}) doesn't decompose "
        f"into intermediate_size={intermediate_size} + 2*groups_ss. "
        f"remainder={remainder} (expected positive and even). "
        f"Check that conv shape {conv_shape} uses SD layout (state_len, dim)."
    )
    groups_ss = remainder // 2

    conv_dtype_size = torch.tensor(
        [],
        dtype=mamba_spec.dtypes[0],  # type: ignore[misc]
    ).element_size()

    return ConvPermParams(
        conv_rows=conv_rows,
        intermediate_size=intermediate_size,
        groups_ss=groups_ss,
        local_tp=local_tp,
        conv_dtype_size=conv_dtype_size,
    )


def build_chunk_perm_forward(
    conv_rows: int,
    x_local: int,
    b_local: int,
) -> list[int]:
    """Build gather indices: original flat layout -> chunk-interleaved transposed.

    Called once at init time. The returned indices are stored as a tensor
    and used at transfer time: ``permuted = original_flat[perm]``.

    Args:
        conv_rows: Conv state rows (d_conv - 1, e.g. 3).
        x_local: Per-rank x sub-projection width (intermediate_size // tp).
        b_local: Per-rank B (or C) sub-projection width (n_groups * d_state // tp).

    Returns:
        Gather-index list of length ``conv_rows * (x_local + 2 * b_local)``.
        ``perm[i] = j`` means output position *i* reads from input position *j*.

    The layout is split into ``g = gcd(x_local, b_local)`` chunks, each
    containing ``x_local/g`` x-cols, ``b_local/g`` B-cols, and ``b_local/g``
    C-cols, column-major (transposed) within each chunk. This makes each
    D rank's shard a contiguous byte range for a single RDMA read.
    """
    g = math.gcd(x_local, b_local)
    x_r, b_r = x_local // g, b_local // g
    conv_dim = x_local + 2 * b_local

    perm: list[int] = []
    for k in range(g):
        for j in range(x_r):
            col = k * x_r + j
            for r in range(conv_rows):
                perm.append(r * conv_dim + col)
        for j in range(b_r):
            col = x_local + k * b_r + j
            for r in range(conv_rows):
                perm.append(r * conv_dim + col)
        for j in range(b_r):
            col = x_local + b_local + k * b_r + j
            for r in range(conv_rows):
                perm.append(r * conv_dim + col)
    return perm


def build_chunk_perm_inverse(
    conv_rows: int,
    x_shard: int,
    b_shard: int,
) -> list[int]:
    """Build gather indices: chunk-interleaved transposed -> original row-major.

    Inverse of :func:`build_chunk_perm_forward`, applied on D-side after
    RDMA read to restore the conv state to ``(conv_rows, [x|B|C])`` layout.

    Args:
        conv_rows: Conv state rows (d_conv - 1, e.g. 3).
        x_shard: Per-rank x sub-projection width (intermediate_size // tp).
        b_shard: Per-rank B (or C) sub-projection width (n_groups * d_state // tp).

    Returns:
        Gather-index list of length ``conv_rows * (x_shard + 2 * b_shard)``.
        ``inv[i] = j`` means restored position *i* reads from received
        position *j*.
    """
    g = math.gcd(x_shard, b_shard)
    x_r, b_r = x_shard // g, b_shard // g
    shard_dim = x_shard + 2 * b_shard
    inv_perm = [0] * (conv_rows * shard_dim)

    in_pos = 0
    for k in range(g):
        for j in range(x_r):
            out_col = k * x_r + j
            for r in range(conv_rows):
                inv_perm[r * shard_dim + out_col] = in_pos
                in_pos += 1
        for j in range(b_r):
            out_col = x_shard + k * b_r + j
            for r in range(conv_rows):
                inv_perm[r * shard_dim + out_col] = in_pos
                in_pos += 1
        for j in range(b_r):
            out_col = x_shard + b_shard + k * b_r + j
            for r in range(conv_rows):
                inv_perm[r * shard_dim + out_col] = in_pos
                in_pos += 1
    return inv_perm


def compute_mamba_phys_ratio(ssm_sizes: tuple[int, ...], block_len: int) -> int:
    """Physical kernel blocks per logical mamba block."""
    return math.ceil((ssm_sizes[0] + ssm_sizes[1]) / block_len)
