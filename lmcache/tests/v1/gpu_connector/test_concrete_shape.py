# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :func:`get_concrete_engine_kv_shape_from_shape_desc`.

These run without a CUDA build: ``lmcache.c_ops`` resolves to the
pure-Python fallback, which provides both ``PageBufferShapeDesc`` and
``EngineKVFormat``.
"""

# First Party
from lmcache.v1.gpu_connector.utils import (
    get_concrete_engine_kv_shape_from_shape_desc,
)
import lmcache.c_ops as lmc_ops


def _make_shape_desc(
    *, kv_size: int, nl: int, nb: int, bs: int, nh: int, hs: int
) -> "lmc_ops.PageBufferShapeDesc":
    """Build a ``PageBufferShapeDesc`` with the given geometry."""
    sd = lmc_ops.PageBufferShapeDesc()
    sd.kv_size = kv_size
    sd.nl = nl
    sd.nb = nb
    sd.bs = bs
    sd.nh = nh
    sd.hs = hs
    sd.element_size = 2
    sd.block_stride_elems = 0
    return sd


def test_concrete_shape_vllm_flash_attn():
    sd = _make_shape_desc(kv_size=2, nl=32, nb=2048, bs=16, nh=8, hs=128)
    out = get_concrete_engine_kv_shape_from_shape_desc(
        sd, lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    )
    assert out == "32 x [2, 2048, 16, 8, 128]"


def test_concrete_shape_vllm_mla():
    sd = _make_shape_desc(kv_size=1, nl=61, nb=1024, bs=64, nh=1, hs=512)
    out = get_concrete_engine_kv_shape_from_shape_desc(
        sd, lmc_ops.EngineKVFormat.NL_X_NB_BS_HS
    )
    assert out == "61 x [1024, 64, 512]"


def test_concrete_shape_uses_pbs_for_folded_formats():
    # NL_X_NBBS_ONE_HS folds num_blocks * block_size into one PBS dim.
    sd = _make_shape_desc(kv_size=1, nl=2, nb=32, bs=16, nh=1, hs=128)
    out = get_concrete_engine_kv_shape_from_shape_desc(
        sd, lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS
    )
    assert out == "2 x [512, 1, 128]"  # 512 == 32 * 16


def test_concrete_shape_is_group_accurate():
    # Two groups with different layer counts produce different shapes for
    # the same format — the whole-context helper could not do this.
    fmt = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    g0 = _make_shape_desc(kv_size=2, nl=4, nb=128, bs=16, nh=8, hs=64)
    g1 = _make_shape_desc(kv_size=2, nl=2, nb=128, bs=16, nh=16, hs=64)
    assert get_concrete_engine_kv_shape_from_shape_desc(g0, fmt) == (
        "4 x [2, 128, 16, 8, 64]"
    )
    assert get_concrete_engine_kv_shape_from_shape_desc(g1, fmt) == (
        "2 x [2, 128, 16, 16, 64]"
    )
