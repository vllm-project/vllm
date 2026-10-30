# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba conv-state sub-projection decomposition for the Mooncake Store.

P/D connectors take the same split from the backend's ``MambaSpec``
subclass via ``slice_for_tp_transfer``; this also handles the SD layout.

Supported model types:
  - Mamba1: conv = [x], temporal = (intermediate_size, state_size)
  - Mamba2: conv = [x, B, C], temporal = (num_heads, head_dim)
  - GDN (Gated Delta Net): conv = [Q, K, V] (dim(Q)==dim(K)),
    temporal = (num_v_heads, v_dim, k_dim)
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class MambaConvSplitInfo:
    """Per-rank byte sizes of the conv sub-projections.

    Used to build TP-aware transfer descriptors.
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


def derive_mamba_conv_split(
    mamba_spec: MambaSpec,
    local_tp: int,
) -> MambaConvSplitInfo:
    """Derive per-rank sub-projection byte sizes from a MambaSpec.

    Called once at init on both P and D.  Decomposes the conv dimension
    into its sub-projection parts based on the model type.

    Args:
        mamba_spec: MambaSpec whose shapes are:
            shapes[0] = conv state in the configured DS or SD layout.
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

    if is_conv_state_dim_first():
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
