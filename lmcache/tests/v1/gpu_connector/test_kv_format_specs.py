# SPDX-License-Identifier: Apache-2.0
"""Golden tests for the kv_format spec layer.

Each :class:`KVFormatSpec` is pinned to independently-computed expected
geometry for a synthetic ``kv_caches`` built from distinct prime-ish
dims. The values were validated to match the pre-refactor ``utils.py``
switch-on-enum accessors exactly; this file freezes them so future
edits to a spec cannot silently drift.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format import (
    describe_shape,
    get_spec,
    get_spec_class,
)
import lmcache.c_ops as lmc_ops

# Distinct dims so a wrong axis surfaces as a wrong number.
NB, NL, BS, NH, HS = 7, 5, 3, 2, 4
PBS = NB * BS  # 21
DT = torch.float16


def _t(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=DT)


def _build(name: str):
    """Build a synthetic normalized kv_caches for the named format."""
    builders = {
        "NB_NL_TWO_BS_NH_HS": lambda: _t(NB, NL, 2, BS, NH, HS),
        "NB_NL_TWO_NH_BS_HS": lambda: _t(NB, NL, 2, NH, BS, HS),
        "NL_X_TWO_NB_BS_NH_HS": lambda: [_t(2, NB, BS, NH, HS) for _ in range(NL)],
        "NL_X_NB_TWO_BS_NH_HS": lambda: [_t(NB, 2, BS, NH, HS) for _ in range(NL)],
        "NL_X_TWO_NB_NH_BS_HS": lambda: [_t(2, NB, NH, BS, HS) for _ in range(NL)],
        "NL_X_NB_TWO_NH_BS_HS": lambda: [_t(NB, 2, NH, BS, HS) for _ in range(NL)],
        "NL_X_NB_BS_HS": lambda: [_t(NB, BS, HS) for _ in range(NL)],
        "TWO_X_NL_X_NBBS_NH_HS": lambda: [
            [_t(PBS, NH, HS) for _ in range(NL)] for _ in range(2)
        ],
        "TWO_X_NL_X_NB_BS_NH_HS": lambda: [
            [_t(NB, BS, NH, HS) for _ in range(NL)] for _ in range(2)
        ],
        "NL_X_NBBS_ONE_HS": lambda: [_t(PBS, 1, HS) for _ in range(NL)],
        "NL_X_NB_NH_BS_TWO_HS": lambda: [_t(NB, NH, BS, 2, HS) for _ in range(NL)],
        "NL_X_NB_BS_NH_TWO_HS": lambda: [_t(NB, BS, NH, 2, HS) for _ in range(NL)],
        "NL_X_NB_NH_BS_CS": lambda: [_t(NB, NH, BS, 2 * HS) for _ in range(NL)],
        "NL_X_NB_BS_NH_CS": lambda: [_t(NB, BS, NH, 2 * HS) for _ in range(NL)],
    }
    return builders[name]()


# Independently-computed expected geometry per format. ``num_blocks`` /
# ``block_size`` are ``None`` for NBBS-fused formats, which must raise.
_RAISE = object()
GOLDEN = {
    "NB_NL_TWO_BS_NH_HS": dict(
        shape_desc="[NB, NL, 2, BS, NH, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * 2 * BS * NH * HS,
        concrete="[7, 5, 2, 3, 2, 4]",
    ),
    "NB_NL_TWO_NH_BS_HS": dict(
        shape_desc="[NB, NL, 2, NH, BS, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * 2 * NH * BS * HS,
        concrete="[7, 5, 2, 2, 3, 4]",
    ),
    "NL_X_TWO_NB_BS_NH_HS": dict(
        shape_desc="NL x [2, NB, BS, NH, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * BS * NH * HS * 2,
        concrete="5 x [2, 7, 3, 2, 4]",
    ),
    "NL_X_NB_TWO_BS_NH_HS": dict(
        shape_desc="NL x [NB, 2, BS, NH, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * BS * NH * HS * 2,
        concrete="5 x [7, 2, 3, 2, 4]",
    ),
    "NL_X_TWO_NB_NH_BS_HS": dict(
        shape_desc="NL x [2, NB, NH, BS, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * NH * BS * HS * 2,
        concrete="5 x [2, 7, 2, 3, 4]",
    ),
    "NL_X_NB_TWO_NH_BS_HS": dict(
        shape_desc="NL x [NB, 2, NH, BS, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * NH * BS * HS * 2,
        concrete="5 x [7, 2, 2, 3, 4]",
    ),
    "NL_X_NB_BS_HS": dict(
        shape_desc="NL x [NB, BS, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=1,
        hidden_dim=HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * BS * HS,
        concrete="5 x [7, 3, 4]",
    ),
    "TWO_X_NL_X_NBBS_NH_HS": dict(
        shape_desc="2 x NL x [PBS, NH, HS]",
        num_layers=NL,
        num_blocks=_RAISE,
        block_size=_RAISE,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=PBS * NH * HS * 2,
        concrete="2 x 5 x [21, 2, 4]",
    ),
    "TWO_X_NL_X_NB_BS_NH_HS": dict(
        shape_desc="2 x NL x [NB, BS, NH, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=NB * BS * NH * HS * 2,
        concrete="2 x 5 x [7, 3, 2, 4]",
    ),
    "NL_X_NBBS_ONE_HS": dict(
        shape_desc="NL x [PBS, 1, HS]",
        num_layers=NL,
        num_blocks=_RAISE,
        block_size=_RAISE,
        page_buffer_size=PBS,
        num_heads=1,
        hidden_dim=HS,
        head_size=HS,
        tokens_per_layer=PBS,
        elements_per_layer=PBS * HS,
        concrete="5 x [21, 1, 4]",
    ),
    "NL_X_NB_NH_BS_TWO_HS": dict(
        shape_desc="NL x [NB, NH, BS, 2, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS * 2,
        tokens_per_layer=PBS,
        elements_per_layer=NB * NH * BS * HS * 2,
        concrete="5 x [7, 2, 3, 2, 4]",
    ),
    "NL_X_NB_BS_NH_TWO_HS": dict(
        shape_desc="NL x [NB, BS, NH, 2, HS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS,
        head_size=HS * 2,
        tokens_per_layer=PBS,
        elements_per_layer=NB * BS * NH * HS * 2,
        concrete="5 x [7, 3, 2, 2, 4]",
    ),
    # CS formats: head_size is the raw trailing content dim (CS == 2 * HS),
    # so hidden_dim == num_heads * head_size holds.
    "NL_X_NB_NH_BS_CS": dict(
        shape_desc="NL x [NB, NH, BS, CS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS * 2,
        head_size=HS * 2,
        tokens_per_layer=PBS,
        elements_per_layer=NB * NH * BS * HS * 2,
        concrete="5 x [7, 2, 3, 8]",
    ),
    "NL_X_NB_BS_NH_CS": dict(
        shape_desc="NL x [NB, BS, NH, CS]",
        num_layers=NL,
        num_blocks=NB,
        block_size=BS,
        page_buffer_size=PBS,
        num_heads=NH,
        hidden_dim=NH * HS * 2,
        head_size=HS * 2,
        tokens_per_layer=PBS,
        elements_per_layer=NB * BS * NH * HS * 2,
        concrete="5 x [7, 3, 2, 8]",
    ),
}

# Only formats the installed extension actually exposes.
FORMAT_NAMES = [n for n in GOLDEN if hasattr(lmc_ops.EngineKVFormat, n)]


@pytest.fixture(params=FORMAT_NAMES)
def case(request):
    name = request.param
    return name, getattr(lmc_ops.EngineKVFormat, name), GOLDEN[name]


# Must run before any other test constructs the deprecated specs:
# lmcache_deprecate warns only on the first call per process.
def test_deprecated_two_hs_specs_warn():
    for name in ("NL_X_NB_NH_BS_TWO_HS", "NL_X_NB_BS_NH_TWO_HS"):
        with pytest.warns(DeprecationWarning, match=name):
            get_spec(_build(name), getattr(lmc_ops.EngineKVFormat, name))


def test_static_metadata(case):
    # The format's static layout flags are pinned in test_kv_format_classification
    # (read via lmc_ops); here we only freeze the symbolic shape.
    name, fmt, gold = case
    assert describe_shape(fmt) == gold["shape_desc"], name


def test_scalar_geometry(case):
    name, fmt, gold = case
    spec = get_spec(_build(name), fmt)
    for attr in (
        "num_layers",
        "page_buffer_size",
        "num_heads",
        "hidden_dim",
        "head_size",
        "tokens_per_layer",
        "elements_per_layer",
    ):
        assert getattr(spec, attr)() == gold[attr], f"{name}.{attr}"
    assert spec.dtype() == DT, name
    assert spec.concrete_shape_str() == gold["concrete"], name


def test_num_blocks_and_block_size(case):
    name, fmt, gold = case
    spec = get_spec(_build(name), fmt)
    if gold["num_blocks"] is _RAISE:
        with pytest.raises(ValueError):
            spec.num_blocks()
        with pytest.raises(ValueError):
            spec.block_size()
    else:
        assert spec.num_blocks() == gold["num_blocks"], name
        assert spec.block_size() == gold["block_size"], name


def test_data_ptrs_shape(case):
    name, fmt, gold = case
    kv = _build(name)
    spec = get_spec(kv, fmt)
    ptrs = spec.data_ptrs(list(range(NL)))
    if lmc_ops.is_cross_layer(fmt):
        assert len(ptrs) == 1, name  # single base pointer
    elif lmc_ops.is_kv_list(fmt):
        assert len(ptrs) == 2 * NL, name  # K's then V's
    else:
        assert len(ptrs) == NL, name  # one per layer
    assert all(isinstance(p, int) for p in ptrs), name


def test_all_extension_formats_registered():
    # No silent gaps: every format the extension exposes has a spec.
    for name in FORMAT_NAMES:
        assert get_spec_class(getattr(lmc_ops.EngineKVFormat, name)) is not None, name


def test_spec_carries_attention_backends(case):
    # First Party
    from lmcache.v1.gpu_connector import utils

    name, fmt, _ = case
    cls = get_spec_class(fmt)
    # Backend labels are colocated on the spec (one format -> many backends).
    assert isinstance(cls.attention_backends, tuple) and cls.attention_backends, name
    # The facade returns the first (canonical/representative) one.
    assert utils.get_attention_backend(fmt) == cls.attention_backends[0], name


def test_facade_diagnostic_labels(case):
    # First Party
    from lmcache.v1.gpu_connector import utils

    name, fmt, gold = case
    # get_engine_kv_shape_description is the geometry legend (delegates to spec).
    assert utils.get_engine_kv_shape_description(fmt) == gold["shape_desc"], name
    # get_attention_backend is a diagnostic-only representative label.
    label = utils.get_attention_backend(fmt)
    assert isinstance(label, str) and not label.startswith("Unknown"), name
