# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba/SSM state sub-projection decomposition for NIXL transfer.

With DS conv state layout (dim, state_len), sub-projections are contiguous
in memory.  Each D rank reads its slices via separate RDMA transfers.

Supported model types:
  - Mamba2: conv = [x, B, C], temporal = (num_heads, head_dim)
  - GDN (Gated Delta Net): conv = [Q, K, V] (dim(Q)==dim(K)),
    temporal = (num_v_heads, v_dim, k_dim)
  - KDA (Kimi Delta Attention): conv = [Q, K, V], temporal = recurrent
    — mapped into MambaConvSplitInfo via 4-shape dispatch.
"""

import math
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of the 3 conv sub-projections.

    Used by both P and D sides for NIXL descriptor registration.
    All fields are LOCAL to this engine's TP (already divided by TP size).

    DS memory layout within one page (contiguous):
      Mamba2: |-- x --|- B -|- C -|  (B == C)
      GDN:    |- Q -|- K -|-- V --|  (dim(Q)==dim(K), V may differ)
      KDA:    |- Q -|- K -|-- V --|  (same layout, Q/K/V are independent conv
               states, followed by SSM temporal = recurrent state)
    """

    conv_rows: int  # conv_kernel - 1 (typically 3)
    local_proj_dims: tuple[int, int, int]  # per-rank column counts per sub-proj
    conv_dtype_size: int  # bytes per element (e.g. 2 for float16)
    ssm_sizes: tuple[int, int]  # (conv_state_bytes, ssm_state_bytes)

    @property
    def state_sizes(self) -> tuple[int, ...]:
        return self.ssm_sizes

    @property
    def local_conv_dim(self) -> int:
        """Total conv columns per rank."""
        return sum(self.local_proj_dims)

    @property
    def proj_bytes(self) -> tuple[int, int, int]:
        """Byte sizes of the 3 sub-projections for one rank."""
        row_bytes = self.conv_rows * self.conv_dtype_size
        return tuple(d * row_bytes for d in self.local_proj_dims)  # type: ignore[return-value]

    @property
    def local_conv_offsets(self) -> list[tuple[int, int]]:
        """(byte_offset, byte_size) of each sub-projection within this
        engine's page.

        Used by both P and D for local descriptor registration.
        """
        conv0, conv1, conv2 = self.proj_bytes
        return [(0, conv0), (conv0, conv1), (conv0 + conv1, conv2)]

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
        conv0, conv1, conv2 = self.proj_bytes
        if tp_ratio >= 1:
            remote_conv0 = conv0 * tp_ratio
            remote_conv1 = conv1 * tp_ratio
            return [
                (local_rank_offset * conv0, conv0),
                (remote_conv0 + local_rank_offset * conv1, conv1),
                (remote_conv0 + remote_conv1 + local_rank_offset * conv2, conv2),
            ]
        else:
            # NOTE (ZhanqiuHu): tp_ratio < 0 means P_TP > D_TP, so P pages
            # are smaller than D's. Local dims are D-sized, but we need
            # P-sized offsets. Scale down by |tp_ratio|.
            abs_ratio = -tp_ratio
            remote_conv0 = conv0 // abs_ratio
            remote_conv1 = conv1 // abs_ratio
            remote_conv2 = conv2 // abs_ratio
            return [
                (0, remote_conv0),
                (remote_conv0, remote_conv1),
                (remote_conv0 + remote_conv1, remote_conv2),
            ]


def derive_mamba_conv_split(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> MambaConvSplitInfo:
    """Derive per-rank byte sizes from a MambaSpec.

    Called once at init on both P and D.  Dispatches by mamba_type:

    - GDN_ATTN with 4 shapes: KDA variant (conv_q, conv_k, conv_v, recurrent)
      mapped to MambaConvSplitInfo.
    - GDN_ATTN with 2 shapes: standard GDN (conv_state, ssm_state).
    - MAMBA2: conv_state with sub-projection decomposition.
    """
    _supported = (
        MambaAttentionBackendEnum.MAMBA2,
        MambaAttentionBackendEnum.GDN_ATTN,
    )
    if mamba_spec.mamba_type not in _supported:
        raise NotImplementedError(
            f"3-read conv transfer only supports Mamba2 and GDN models, "
            f"got mamba_type={mamba_spec.mamba_type!r}."
        )

    assert is_conv_state_dim_first(), "3-read requires DS conv state layout"

    # KDA variant of GDN: 4 separate states (conv_q, conv_k, conv_v, recurrent)
    # mapped to MambaConvSplitInfo (first 3 → conv sub-projections, 4th → SSM).
    if (
        mamba_spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
        and len(mamba_spec.shapes) == 4
    ):
        conv_rows = mamba_spec.shapes[0][1]
        assert all(s[1] == conv_rows for s in mamba_spec.shapes[:3]), (
            "KDA conv states must all have the same conv_rows"
        )
        assert all(d == mamba_spec.dtypes[0] for d in mamba_spec.dtypes[:3]), (
            "KDA conv states must all have the same dtype"
        )
        local_proj_dims = (
            mamba_spec.shapes[0][0],  # q_dim_local
            mamba_spec.shapes[1][0],  # k_dim_local
            mamba_spec.shapes[2][0],  # v_dim_local
        )
        conv_dtype_size = torch.tensor(
            [],
            dtype=mamba_spec.dtypes[0],  # type: ignore[misc]
        ).element_size()
        conv_state_bytes = sum(
            torch.Size(s).numel() * torch.tensor([], dtype=d).element_size()  # type: ignore[misc]
            for s, d in zip(mamba_spec.shapes[:3], mamba_spec.dtypes[:3])
        )
        ssm_state_bytes = (
            torch.Size(mamba_spec.shapes[3]).numel()
            * torch.tensor(
                [],
                dtype=mamba_spec.dtypes[3],  # type: ignore[misc]
            ).element_size()
        )
        return MambaConvSplitInfo(
            conv_rows=conv_rows,
            local_proj_dims=local_proj_dims,
            conv_dtype_size=conv_dtype_size,
            ssm_sizes=(conv_state_bytes, ssm_state_bytes),
        )

    # --- Standard 2-shape path (Mamba2 / GDN) ---
    conv_shape = mamba_spec.shapes[0]
    assert len(conv_shape) == 2, f"Expected 2D conv state shape, got {conv_shape}"

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

    if mamba_spec.mamba_type == MambaAttentionBackendEnum.MAMBA2:
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
    """
    return math.ceil(sum(ssm_sizes) / block_len)
