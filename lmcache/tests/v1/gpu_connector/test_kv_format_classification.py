# SPDX-License-Identifier: Apache-2.0
"""Golden tests for the ``EngineKVFormat`` classification predicates.

The structural shape (``is_cross_layer`` / ``is_kv_list`` / ``is_layer_list``)
and the ``is_mla`` modifier are defined once in ``csrc/engine_kv_format.h`` and
read via ``lmc_ops``, shared with the device kernels. This file pins each
format's classification and enforces that the three structural flags partition
every format (exactly one is true), so a new format or an edit cannot silently
break the contract the per-layer detection relies on.
"""

# First Party
import lmcache.c_ops as lmc_ops

F = lmc_ops.EngineKVFormat

# (is_cross_layer, is_kv_list, is_layer_list, is_mla) per format.
EXPECTED = {
    F.NB_NL_TWO_BS_NH_HS: (True, False, False, False),
    F.NB_NL_TWO_NH_BS_HS: (True, False, False, False),
    F.TWO_X_NL_X_NBBS_NH_HS: (False, True, False, False),
    F.TWO_X_NL_X_NB_BS_NH_HS: (False, True, False, False),
    F.NL_X_TWO_NB_BS_NH_HS: (False, False, True, False),
    F.NL_X_NB_TWO_BS_NH_HS: (False, False, True, False),
    F.NL_X_TWO_NB_NH_BS_HS: (False, False, True, False),
    F.NL_X_NB_TWO_NH_BS_HS: (False, False, True, False),
    F.NL_X_NB_NH_BS_TWO_HS: (False, False, True, False),
    F.NL_X_NB_BS_NH_TWO_HS: (False, False, True, False),
    F.NL_X_NB_NH_BS_CS: (False, False, True, False),
    F.NL_X_NB_BS_NH_CS: (False, False, True, False),
    F.NL_X_NB_BS_HS: (False, False, True, True),
    F.NL_X_NBBS_ONE_HS: (False, False, True, True),
    F.NL_X_NB_BSV_BSS: (False, False, True, True),
}


def _all_formats():
    return [v for v in vars(F).values() if isinstance(v, F)]


def test_classification_matches_golden():
    for fmt, expected in EXPECTED.items():
        got = (
            lmc_ops.is_cross_layer(fmt),
            lmc_ops.is_kv_list(fmt),
            lmc_ops.is_layer_list(fmt),
            lmc_ops.is_mla(fmt),
        )
        assert got == expected, f"{fmt}: got {got}, expected {expected}"


def test_every_format_is_pinned():
    # A new EngineKVFormat must be added to EXPECTED (and classified) deliberately.
    assert set(_all_formats()) == set(EXPECTED)


def test_structural_flags_partition_every_format():
    # Exactly one structural shape is true for every format.
    for fmt in _all_formats():
        structural = (
            lmc_ops.is_cross_layer(fmt),
            lmc_ops.is_kv_list(fmt),
            lmc_ops.is_layer_list(fmt),
        )
        assert sum(structural) == 1, f"{fmt}: structural flags {structural}"
