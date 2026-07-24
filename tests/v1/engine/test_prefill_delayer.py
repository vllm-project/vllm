# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the PrefillDelayer state machine.

The cross-DP all_reduce is mocked to simulate specific reduced (cross-rank)
states so the decision logic can be tested single-process and deterministically.
"""

from unittest.mock import patch

from vllm.v1.engine.prefill_delayer import PrefillDelayer


def _make(**kwargs) -> PrefillDelayer:
    delayer = PrefillDelayer(dp_size=8, cpu_group=None, **kwargs)
    # Consume the skip-first grace so tests exercise the steady-state logic
    # unless they explicitly want to test skip-first.
    return delayer


def _patched_reduce(any_prefillable: bool, any_not_prefillable: bool, force: bool):
    """Return an all_reduce stand-in writing a fixed MAX-reduced buffer.

    Mirrors the 3-slot encoding in PrefillDelayer: [prefillable, force,
    not_prefillable].
    """

    def _fake(buf, *args, **kwargs):
        buf[0] = 1 if any_prefillable else 0
        buf[1] = 1 if force else 0
        buf[2] = 1 if any_not_prefillable else 0

    return _fake


def _call(delayer, fake, local_prefillable=True, token_usage=1.0):
    with patch("torch.distributed.all_reduce", side_effect=fake):
        return delayer.should_allow_prefill(
            local_prefillable=local_prefillable, token_usage=token_usage
        )


def test_skip_first_allows():
    d = _make()
    fake = _patched_reduce(any_prefillable=True, any_not_prefillable=True, force=False)
    # Even in a mixed state, the very first call is allowed (batch build-up).
    assert _call(d, fake) is True


def test_all_ranks_prefillable_allows():
    d = _make()
    d._skip_first = False
    fake = _patched_reduce(any_prefillable=True, any_not_prefillable=False, force=False)
    assert _call(d, fake) is True


def test_no_rank_prefillable_allows():
    d = _make()
    d._skip_first = False
    fake = _patched_reduce(any_prefillable=False, any_not_prefillable=True, force=False)
    assert _call(d, fake, local_prefillable=False) is True


def test_mixed_state_delays():
    d = _make()
    d._skip_first = False
    fake = _patched_reduce(any_prefillable=True, any_not_prefillable=True, force=False)
    assert _call(d, fake) is False


def test_mixed_state_times_out_by_passes():
    d = _make(max_delay_passes=3, max_delay_ms=1e9)
    d._skip_first = False
    fake = _patched_reduce(any_prefillable=True, any_not_prefillable=True, force=False)
    # First max_delay_passes calls delay, then force-allow.
    assert _call(d, fake) is False
    assert _call(d, fake) is False
    assert _call(d, fake) is False
    assert _call(d, fake) is True


def test_watermark_force_allows_even_when_mixed():
    d = _make(token_usage_low_watermark=0.2)
    d._skip_first = False
    # force is computed locally then MAX-reduced; simulate a rank forcing.
    fake = _patched_reduce(any_prefillable=True, any_not_prefillable=True, force=True)
    assert _call(d, fake, token_usage=0.05) is True


def test_wave_boundary_rearms_skip_first():
    d = _make()
    d._skip_first = False
    d._delayed_count = 5
    d.on_wave_boundary()
    assert d._skip_first is True
    assert d._delayed_count == 0


def test_reduce_buffer_encoding():
    """The local contribution must encode prefillable/force/not-prefillable."""
    d = _make(token_usage_low_watermark=0.5)
    d._skip_first = False
    captured = {}

    def _fake(buf, *args, **kwargs):
        captured["buf"] = buf.clone()
        # Simulate no other rank contributing.

    with patch("torch.distributed.all_reduce", side_effect=_fake):
        d.should_allow_prefill(local_prefillable=True, token_usage=0.1)

    buf = captured["buf"]
    assert buf[0].item() == 1  # prefillable
    assert buf[1].item() == 1  # force (usage 0.1 < watermark 0.5)
    assert buf[2].item() == 0  # not-prefillable is False


def test_not_prefillable_encoding():
    d = _make()
    d._skip_first = False
    captured = {}

    def _fake(buf, *args, **kwargs):
        captured["buf"] = buf.clone()

    with patch("torch.distributed.all_reduce", side_effect=_fake):
        d.should_allow_prefill(local_prefillable=False, token_usage=1.0)

    buf = captured["buf"]
    assert buf[0].item() == 0
    assert buf[2].item() == 1


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
