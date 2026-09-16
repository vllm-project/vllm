# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DFlash speculator must align ranks before capturing under DCP.

Under decode context parallelism the draft is sharded, so the graph captured by
``DFlashSpeculator.capture`` contains a context-parallel collective. Graph
capture is only safe when every rank enters it aligned; a rank still finishing
earlier work can have its collective recorded into a peer's graph, which faults
at capture time.

This is a multi-rank boot-time race, so these tests do not reproduce the fault —
they pin the two properties that prevent it:

  * the barrier is issued, and issued *before* capture begins;
  * it is skipped entirely when DCP is off, so the single-GPU path pays nothing
    and does not require an initialised process group.

The barrier goes through the TP group's ``GroupCoordinator``, not
``torch.distributed.barrier``: the latter is an NCCL barrier that allocates
secret GPU tensors and can move the current device out from under capture, which
its own docstring in ``parallel_state.py`` warns about.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.worker.gpu.spec_decode.dflash import speculator as speculator_mod
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def _make_speculator(dcp_size: int, calls: list[str]) -> DFlashSpeculator:
    """A speculator with only what ``capture`` touches.

    Constructing the real thing would load a draft model; ``capture`` reads the
    parallel config, three buffers and the cudagraph manager, so those are all
    that need to exist.
    """
    spec = object.__new__(DFlashSpeculator)
    spec.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size)
    )
    for name in ("sample_indices", "sample_pos"):
        buf = MagicMock()
        buf.zero_ = MagicMock()
        setattr(spec, name, buf)
    spec.sample_idx_mapping = MagicMock()
    spec.query_cudagraph_manager = MagicMock()
    spec.query_cudagraph_manager.capture = MagicMock(
        side_effect=lambda *a, **k: calls.append("capture")
    )
    # Everything else capture() dereferences on the way to the manager.
    for name in (
        "input_buffers",
        "block_tables",
        "attn_groups",
        "kv_cache_config",
        "max_model_len",
        "_group_causal",
        "_generate_draft",
    ):
        setattr(spec, name, MagicMock())
    return spec


@pytest.fixture
def patched(monkeypatch):
    """Record barrier/synchronize/capture in the order they are called."""
    calls: list[str] = []

    group = MagicMock()
    group.barrier = MagicMock(side_effect=lambda: calls.append("barrier"))
    monkeypatch.setattr(speculator_mod, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        speculator_mod, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )
    monkeypatch.setattr(
        speculator_mod.torch.accelerator,
        "synchronize",
        lambda *a, **k: calls.append("synchronize"),
    )
    return calls, group


def test_the_barrier_is_on_tp_not_dcp(patched):
    """TP, not DCP: __init__ forces the draft to pcp == 1, where config
    validation requires tp % dcp == 0, so DCP is a subset of TP. The captured
    graph also carries the draft's TP all-reduces, so at dcp < tp a DCP barrier
    would align only some participants."""
    calls, group = patched
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as mod

    assert hasattr(mod, "get_tp_group"), "barrier must use the TP group"
    assert not hasattr(mod, "get_dcp_group"), (
        "DCP is a subset of TP for the draft; a DCP barrier misses participants "
        "when dcp < tp"
    )


def test_ranks_are_aligned_before_capture_under_dcp(patched):
    """With DCP on, the device is quiesced and the ranks aligned, and both
    happen before capture starts."""
    calls, group = patched
    _make_speculator(dcp_size=8, calls=calls).capture()

    assert calls == ["synchronize", "barrier", "capture"], calls
    group.barrier.assert_called_once()


def test_no_barrier_when_dcp_is_off(patched):
    """Without DCP the draft is not sharded, the captured graph holds no
    collective, and a barrier would be both pointless and a hard requirement
    for an initialised process group on the single-GPU path."""
    calls, group = patched
    _make_speculator(dcp_size=1, calls=calls).capture()

    assert calls == ["capture"], calls
    group.barrier.assert_not_called()


def test_barrier_uses_the_tp_group_not_torch_distributed(monkeypatch, patched):
    """An NCCL barrier allocates secret GPU tensors and can move the current
    device; GroupCoordinator.barrier() uses the CPU group instead. Guard against
    a well-meaning simplification back to torch.distributed.barrier()."""
    calls, _ = patched
    import torch.distributed as dist

    raw = MagicMock()
    monkeypatch.setattr(dist, "barrier", raw)
    _make_speculator(dcp_size=8, calls=calls).capture()

    raw.assert_not_called()


def test_no_barrier_off_rocm(patched, monkeypatch):
    """The fault was measured on gfx950 and the fix is untested on CUDA, so
    other platforms keep their current behaviour. Patched rather than detected,
    so both branches are exercised on any runner."""
    calls, group = patched
    monkeypatch.setattr(
        speculator_mod, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )

    _make_speculator(dcp_size=8, calls=calls).capture()

    assert calls == ["capture"], calls
    group.barrier.assert_not_called()
