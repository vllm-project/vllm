# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the 3-read NIXL Mamba conv-state transfer derivation.

Covers both Mamba2 and Gated-Delta-Net (GDN) layouts.  These tests do not
allocate GPU memory or open any RDMA connections; they only exercise the
shape arithmetic in `derive_mamba_conv_split`.
"""

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.v1.kv_cache_interface import MambaSpec


@pytest.fixture(autouse=True)
def _force_ds_conv_state_layout(monkeypatch):
    """The 3-read transfer is only valid for DS conv-state layout, which is
    asserted at runtime in the NIXL worker.  Make the assertion succeed in
    tests by setting the corresponding env var.
    """
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")


def _make_spec(
    *,
    mamba_type: str,
    conv_dim_local: int,
    conv_rows: int,
    temporal_shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> MambaSpec:
    """Build a minimal MambaSpec usable by the 3-read derivation."""
    return MambaSpec(
        block_size=64,
        # shapes[0] = conv state in DS layout: (conv_dim_local, conv_rows)
        # shapes[1] = SSM temporal state (Mamba2: 3D; GDN: 3D; Mamba1: 2D)
        shapes=((conv_dim_local, conv_rows), temporal_shape),
        dtypes=(dtype, dtype),
        mamba_type=mamba_type,
    )


# ---------------------------------------------------------------------------
# Mamba2 (regression) -- the pre-existing path must keep working unchanged.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("local_tp", [1, 2, 4, 8])
def test_mamba2_split_matches_known_values(local_tp: int) -> None:
    # Toy mamba2: intermediate=64, groups_ss=8, head_dim=8 -> 8 heads.
    intermediate = 64
    groups_ss = 8
    head_dim = 8
    num_heads = intermediate // head_dim  # 8
    state_size = 16
    conv_kernel = 4
    conv_dim = intermediate + 2 * groups_ss

    spec = _make_spec(
        mamba_type="mamba2",
        conv_dim_local=conv_dim // local_tp,
        conv_rows=conv_kernel - 1,
        temporal_shape=(num_heads // local_tp, head_dim, state_size),
    )

    out = derive_mamba_conv_split(spec, local_tp=local_tp)

    assert isinstance(out, MambaConvSplitInfo)
    assert out.conv_rows == conv_kernel - 1
    assert out.x_local == intermediate // local_tp
    assert out.b_local == groups_ss // local_tp
    assert out.conv_dtype_size == 2  # bfloat16


# ---------------------------------------------------------------------------
# GDN (Gated Delta Net), used by Qwen3-Next / Qwen3.5.
#
# For GDN the conv layout is identical in form to Mamba2:
#     conv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads
#              = 2*key_dim + value_dim
# and the SSM temporal state shape is (num_v_heads/tp, head_v_dim, head_k_dim).
# The math of derive_mamba_conv_split therefore applies as-is, with
#     intermediate_size_equiv = value_dim,
#     groups_ss_equiv         = key_dim.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("local_tp", [1, 2, 4, 8])
def test_gdn_split_matches_known_values_qwen35(local_tp: int) -> None:
    # Real Qwen3.5 GDN config (from Qwen3.5-397B-A17B-NVFP4 config.json).
    num_v_heads = 64
    num_k_heads = 16
    head_v_dim = 128
    head_k_dim = 128
    conv_kernel = 4

    value_dim = num_v_heads * head_v_dim  # 8192 == intermediate_size_equiv
    key_dim = num_k_heads * head_k_dim  # 2048 == groups_ss_equiv
    conv_dim = 2 * key_dim + value_dim  # 12288

    # Sanity: the derivation requires conv_dim divisible by local_tp.
    assert conv_dim % local_tp == 0
    assert num_v_heads % local_tp == 0

    spec = _make_spec(
        mamba_type="gdn_attention",
        conv_dim_local=conv_dim // local_tp,
        conv_rows=conv_kernel - 1,
        # GDN temporal shape: 3D, third axis is head_k_dim and is irrelevant
        # for the conv decomposition (only first two axes are read).
        temporal_shape=(num_v_heads // local_tp, head_v_dim, head_k_dim),
    )

    out = derive_mamba_conv_split(spec, local_tp=local_tp)

    assert out.conv_rows == conv_kernel - 1
    assert out.x_local == value_dim // local_tp
    assert out.b_local == key_dim // local_tp
    # conv layout: |x|B|C|, so total per rank = x + B + C = x + 2*B
    assert out.conv_dim_local == out.x_local + 2 * out.b_local


def test_gdn_with_uneven_value_key_heads() -> None:
    # GDN with very different K vs V head counts (smaller toy case).
    num_v_heads = 32
    num_k_heads = 4  # GQA-like ratio 8:1
    head_v_dim = 64
    head_k_dim = 64
    conv_kernel = 4
    local_tp = 2

    value_dim = num_v_heads * head_v_dim  # 2048
    key_dim = num_k_heads * head_k_dim  # 256
    conv_dim = 2 * key_dim + value_dim  # 2560

    spec = _make_spec(
        mamba_type="gdn_attention",
        conv_dim_local=conv_dim // local_tp,
        conv_rows=conv_kernel - 1,
        temporal_shape=(num_v_heads // local_tp, head_v_dim, head_k_dim),
    )

    out = derive_mamba_conv_split(spec, local_tp=local_tp)
    assert out.x_local == value_dim // local_tp  # 1024
    assert out.b_local == key_dim // local_tp  # 128


# ---------------------------------------------------------------------------
# Mamba1 (classic, e.g. mamba_ssm v1 layouts) is genuinely unsupported by the
# 3-read transfer because its SSM temporal state is 2-D and lacks the head
# factorization needed to recover intermediate_size on the decode side.
# Make sure we still NotImplementedError for it.
# ---------------------------------------------------------------------------
def test_mamba1_still_raises_not_implemented() -> None:
    intermediate = 64
    state_size = 16
    conv_kernel = 4

    spec = _make_spec(
        mamba_type="mamba1",
        conv_dim_local=intermediate,
        conv_rows=conv_kernel - 1,
        # Mamba1 temporal shape is (intermediate_size//tp, state_size).
        temporal_shape=(intermediate, state_size),
    )

    with pytest.raises(NotImplementedError, match="3-read conv transfer"):
        derive_mamba_conv_split(spec, local_tp=1)


# ---------------------------------------------------------------------------
# remote_conv_offsets / local_conv_offsets sanity (independent of mamba_type).
# ---------------------------------------------------------------------------
def test_local_offsets_are_contiguous_x_then_b_then_c() -> None:
    info = MambaConvSplitInfo(
        conv_rows=3, x_local=8192, b_local=2048, conv_dtype_size=2
    )
    offsets = info.local_conv_offsets

    # Three regions: x at offset 0, B right after, C right after B.
    assert len(offsets) == 3
    (x_off, x_sz), (b_off, b_sz), (c_off, c_sz) = offsets
    assert x_off == 0
    assert x_sz == info.x_bytes
    assert b_off == x_sz
    assert b_sz == info.b_bytes
    assert c_off == x_sz + b_sz
    assert c_sz == info.b_bytes  # B and C are the same size by construction


def test_remote_offsets_homo_tp_zero_offset() -> None:
    """When P_TP == D_TP, each D rank reads its own slice of P (offset 0)."""
    info = MambaConvSplitInfo(conv_rows=3, x_local=2048, b_local=512, conv_dtype_size=2)
    # tp_ratio=1 = homogeneous TP; local_rank_offset=0 -> own slice only.
    offsets = info.remote_conv_offsets(local_rank_offset=0, tp_ratio=1)
    assert offsets == info.local_conv_offsets
