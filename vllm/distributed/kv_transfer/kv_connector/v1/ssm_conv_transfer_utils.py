# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba/GDN conv-state sub-projection decomposition for the 3-read transfer.

With DS conv state layout (dim, state_len), the three conv sub-projections are
contiguous in memory. Each D rank reads its slices via 3
separate RDMA transfers — no P-side permutation needed.
"""

import math
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of 3 sub-projections in Mamba/GDN conv state.

    Used by both P and D sides for NIXL descriptor registration.
    All fields are LOCAL to this engine's TP (already divided by TP size).

    DS memory layout within one page (contiguous in memory):
        |--- p0 (p0_local * conv_rows) ---|--- p1 ---|--- p2 ---|
    """

    conv_rows: int  # conv_kernel - 1 (typically 3)
    p0_local: int  # first projection columns per TP rank
    p1_local: int  # second projection columns per TP rank
    p2_local: int  # third projection columns per TP rank
    conv_dtype_size: int  # bytes per element (e.g. 2 for float16)
    ssm_sizes: tuple[int, int]  # (conv_state_bytes, ssm_state_bytes)

    @property
    def conv_dim_local(self) -> int:
        """Total conv columns per rank: p0 + p1 + p2."""
        return self.p0_local + self.p1_local + self.p2_local

    @property
    def p0_bytes(self) -> int:
        """Byte size of the first sub-projection for one rank."""
        return self.p0_local * self.conv_rows * self.conv_dtype_size

    @property
    def p1_bytes(self) -> int:
        """Byte size of the second sub-projection for one rank."""
        return self.p1_local * self.conv_rows * self.conv_dtype_size

    @property
    def p2_bytes(self) -> int:
        """Byte size of the third sub-projection for one rank."""
        return self.p2_local * self.conv_rows * self.conv_dtype_size

    @property
    def local_conv_offsets(self) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of p0, p1, p2 within this engine's page.

        Used by both P and D for local descriptor registration.
        """
        p0b = self.p0_bytes
        p1b = self.p1_bytes
        p2b = self.p2_bytes
        return [(0, p0b), (p0b, p1b), (p0b + p1b, p2b)]

    def remote_conv_offsets(
        self, local_rank_offset: int, tp_ratio: int
    ) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of this D rank's p0, p1, p2 slice within
        one P page.

        Used by D side only, during remote descriptor registration.

        Args:
            local_rank_offset: which slice this D rank reads.
                tp_ratio > 0: tp_rank % tp_ratio (selects slice of P's page).
                tp_ratio < 0: always 0 (read P's full page).
            tp_ratio: effective ratio (>= 1 when D_TP > P_TP, 1 when
                P_TP > D_TP since each P rank is read in full).
        """
        p0b = self.p0_bytes
        p1b = self.p1_bytes
        p2b = self.p2_bytes
        p0r = p0b * tp_ratio  # full remote p0 section in bytes
        p1r = p1b * tp_ratio  # full remote p1 section in bytes
        return [
            (local_rank_offset * p0b, p0b),
            (p0r + local_rank_offset * p1b, p1b),
            (p0r + p1r + local_rank_offset * p2b, p2b),
        ]


def derive_mamba_conv_split(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> MambaConvSplitInfo:
    """Derive per-rank 3-read split byte sizes from a MambaSpec.

    Called once at init on both P and D. Decomposes conv state into three
    contiguous projection slices read independently during transfer.

    Args:
        mamba_spec: MambaSpec whose shapes are:
            shapes[0] = conv state: (conv_dim_local, conv_rows) in DS layout.
            shapes[1] = SSM temporal:
                - Mamba2: (local_num_heads, head_dim)
                - GDN: (local_num_v_heads, head_v_dim, head_k_dim)
        local_tp: this engine's tensor-parallel size.

    Returns:
        MambaConvSplitInfo with per-rank p0/p1/p2 locals, conv_rows,
        conv_dtype_size, and ssm_sizes (conv_state_bytes, ssm_state_bytes).
    """
    if mamba_spec.mamba_type not in ("mamba2", "gdn_attention"):
        raise NotImplementedError(
            f"3-read conv transfer only supports Mamba2/GDN models, "
            f"got mamba_type={mamba_spec.mamba_type!r}.  "
            f"Mamba1 SSM temporal shape is (intermediate_size // tp, state_size), "
            f"which cannot reconstruct the 3-read split."
        )

    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, f"Expected 2D conv state shape, got {conv_shape}"

    # NOTE (ZhanqiuHu): 3-read requires DS layout, which is already asserted
    # in nixl worker __init__.  Use it directly instead of heuristic detection.
    assert is_conv_state_dim_first(), "3-read requires DS conv state layout"
    local_conv_dim = conv_shape[0]  # DS: (conv_dim_local, conv_rows)
    conv_rows = conv_shape[1]

    if mamba_spec.mamba_type == "mamba2":
        # shapes[1] = (local_num_heads, head_dim), already TP-sharded.
        head_dim = mamba_spec.shapes[1][1]
        local_num_heads = mamba_spec.shapes[1][0]
        intermediate_size = local_num_heads * local_tp * head_dim

        # global conv dim = intermediate_size + 2 * groups_ss.
        remainder = local_conv_dim * local_tp - intermediate_size
        assert remainder > 0 and remainder % 2 == 0, (
            f"Conv dim ({local_conv_dim}*tp={local_tp}) doesn't decompose into "
            f"intermediate_size={intermediate_size} + 2*groups_ss. "
            f"remainder={remainder}"
        )
        groups_ss = remainder // 2
        p0_local = intermediate_size // local_tp
        p1_local = groups_ss // local_tp
        p2_local = groups_ss // local_tp
    else:
        # GDN conv layout is [K, K, V].
        assert len(mamba_spec.shapes[1]) == 3, (
            "Expected 3D GDN temporal state shape "
            f"(num_v_heads/tp, head_v_dim, head_k_dim), got {mamba_spec.shapes[1]}"
        )
        local_num_v_heads, head_v_dim, head_k_dim = mamba_spec.shapes[1]
        local_v_dim = local_num_v_heads * head_v_dim
        assert local_conv_dim >= local_v_dim, (
            f"GDN conv dim ({local_conv_dim}) must be >= local V dim ({local_v_dim})."
        )
        remaining_for_k = local_conv_dim - local_v_dim
        assert remaining_for_k % 2 == 0, (
            f"GDN conv dim ({local_conv_dim}) must decompose as K + K + V; "
            f"remaining_for_k={remaining_for_k} is not divisible by 2."
        )
        local_k_dim = remaining_for_k // 2
        assert local_k_dim % head_k_dim == 0, (
            f"Derived local K dim ({local_k_dim}) must align with head_k_dim "
            f"({head_k_dim})."
        )
        p0_local = local_k_dim
        p1_local = local_k_dim
        p2_local = local_v_dim

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

    # Divide by TP to get per-rank column counts.
    return MambaConvSplitInfo(
        conv_rows=conv_rows,
        p0_local=p0_local,
        p1_local=p1_local,
        p2_local=p2_local,
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
