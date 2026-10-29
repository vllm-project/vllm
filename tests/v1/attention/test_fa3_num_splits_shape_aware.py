# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the shape-aware FA3 split-KV cap.

The FA3 heuristic sizes ``num_splits`` from a double ceiling (see
``_shape_aware_num_splits``). Both ceilings round up, so
``num_kv_heads * num_reqs * num_splits`` can exceed the SM count and a second,
mostly-idle wave is issued. These tests pin the resulting cap so the policy
cannot silently regress.
"""

import pytest

from vllm.v1.attention.backends.flash_attn import (
    _resolve_num_splits,
    _shape_aware_num_splits,
)

# H20 / H100 SM counts, 4 KV heads (GQA), config default of 32.
H20_SMS = 78
H100_SMS = 132
KV_HEADS = 4
CONFIG_CAP = 32


@pytest.mark.parametrize(
    "num_reqs,expected",
    [
        # 78 // (4 * num_reqs), floored, but never below 2.
        (1, 19),
        (2, 9),
        (3, 6),
        (4, 4),  # FA3 heuristic gives 5 here; capped to 4
        (5, 3),
        (6, 3),
        (7, 2),  # FA3 heuristic gives 3 here; capped to 2
        (8, 2),  # FA3 heuristic gives 3 here; capped to 2
        (9, 2),
        (16, 2),
        (64, 2),
        (512, 2),
    ],
)
def test_cap_on_h20(num_reqs: int, expected: int) -> None:
    assert _shape_aware_num_splits(H20_SMS, KV_HEADS, num_reqs, CONFIG_CAP) == expected


def test_cap_never_exceeds_one_wave() -> None:
    """Work items (kv_heads * num_reqs * num_splits) stay within one wave."""
    for num_reqs in range(1, 129):
        cap = _shape_aware_num_splits(H20_SMS, KV_HEADS, num_reqs, CONFIG_CAP)
        if H20_SMS // (KV_HEADS * num_reqs) >= 2:
            assert KV_HEADS * num_reqs * cap <= H20_SMS, num_reqs


def test_cap_is_monotonic_in_batch() -> None:
    caps = [
        _shape_aware_num_splits(H20_SMS, KV_HEADS, n, CONFIG_CAP) for n in range(1, 65)
    ]
    assert caps == sorted(caps, reverse=True)


@pytest.mark.parametrize("config_cap", [1, 2, 4, 8, 16, 32, 64])
def test_respects_config_as_upper_bound(config_cap: int) -> None:
    for num_reqs in (1, 2, 4, 7, 8, 32, 512):
        cap = _shape_aware_num_splits(H20_SMS, KV_HEADS, num_reqs, config_cap)
        assert cap <= config_cap


def test_explicit_disable_is_not_overridden() -> None:
    """num_splits=1 must stay 1: it is the documented workaround for the
    combine-kernel grid bug in #62626, and it selects the Split=false kernel."""
    for num_reqs in (1, 4, 8, 64):
        assert _shape_aware_num_splits(H20_SMS, KV_HEADS, num_reqs, 1) == 1


def test_never_selects_split_false_kernel() -> None:
    """With 2+ allowed by config we must not return 1: num_splits == 1 selects a
    different kernel instantiation that is slower at long context."""
    for num_reqs in range(1, 129):
        assert _shape_aware_num_splits(H20_SMS, KV_HEADS, num_reqs, CONFIG_CAP) >= 2


def test_zero_requests_does_not_divide_by_zero() -> None:
    assert _shape_aware_num_splits(H20_SMS, KV_HEADS, 0, CONFIG_CAP) == 19


def test_mha_small_batch_is_unaffected() -> None:
    """kv_heads == 1 (MHA) on H100: below ~4 requests the cap still allows 32."""
    for num_reqs in (1, 2, 4):
        assert _shape_aware_num_splits(H100_SMS, 1, num_reqs, CONFIG_CAP) == CONFIG_CAP


def test_mha_large_batch_is_capped() -> None:
    """MHA overshoots too once kv_heads * num_reqs * 32 > num_sms: at batch 8 on
    H100 that is 256 work items for 132 SMs, so the cap must come down."""
    assert _shape_aware_num_splits(H100_SMS, 1, 8, CONFIG_CAP) == H100_SMS // 8
    assert 1 * 8 * _shape_aware_num_splits(H100_SMS, 1, 8, CONFIG_CAP) <= H100_SMS


def test_no_global_constant_can_work() -> None:
    """Regression guard for the design: a single constant is wrong in both
    directions -- batch 1 needs a large cap, batch 14 must not drop to 1."""
    small = _shape_aware_num_splits(H20_SMS, KV_HEADS, 1, CONFIG_CAP)
    large = _shape_aware_num_splits(H20_SMS, KV_HEADS, 14, CONFIG_CAP)
    assert small > CONFIG_CAP // 2
    assert large == 2
    assert small != large


class TestResolveNumSplits:
    """Precedence of the budget decision.

    The switch is deliberately static (a config flag, not a property of the
    current batch): every input must be constant for a given CUDA graph, since
    num_splits sizes the accumulators, the combine grid and the scheduler
    metadata layout, which the graph freezes at capture.
    """

    @staticmethod
    def _resolve(config_cap: int, enabled: bool) -> int:
        return _resolve_num_splits(config_cap, 78, 4, 8, enabled)

    def test_disabled_is_the_status_quo(self) -> None:
        # Default off: the configured budget is used verbatim, for any shape.
        assert self._resolve(32, False) == 32
        assert self._resolve(8, False) == 8

    def test_enabled_caps(self) -> None:
        assert self._resolve(32, True) == 2

    def test_zero_means_fa3_heuristic_and_is_untouched(self) -> None:
        # 0 is the "no full CUDA graph / FA2" value; it must never become a cap.
        assert self._resolve(0, True) == 0
        assert self._resolve(0, False) == 0

    def test_explicit_one_is_preserved(self) -> None:
        # The #62626 escape hatch: num_splits=1 must stay 1, on both branches.
        assert self._resolve(1, True) == 1
        assert self._resolve(1, False) == 1

    def test_config_below_cap_is_respected(self) -> None:
        assert self._resolve(4, True) == min(4, 2)

    def test_never_exceeds_config(self) -> None:
        for cfg in (2, 3, 4, 8, 16, 32):
            assert self._resolve(cfg, True) <= cfg

    def test_decision_does_not_depend_on_batch_lengths(self) -> None:
        """Regression guard: an earlier revision gated on request lengths, which
        made the budget vary between capture and replay of the same graph."""
        import inspect

        params = inspect.signature(_resolve_num_splits).parameters
        assert list(params) == [
            "config_cap",
            "num_sms",
            "num_kv_heads",
            "num_reqs",
            "enabled",
        ]
