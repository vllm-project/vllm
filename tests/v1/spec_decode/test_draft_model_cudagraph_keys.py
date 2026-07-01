# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for the drafter cudagraph-keys initialization gate.

The native ``draft_model`` proposer must have its cudagraph dispatcher keys
initialized during engine setup, exactly like the EAGLE / extract-hidden-states
proposers. Otherwise the draft model's captured PIECEWISE graphs are never
dispatched to and every draft step runs eager (a launch-bound regression).

These tests exercise the real
``GPUModelRunner._check_and_update_cudagraph_mode`` with mocked dependencies so
they run on CPU without loading any model. Passing empty attention/kv-cache
lists skips the attention-backend resolution loop, leaving only the drafter
keys-init gate under test.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _make_runner(spec_flags, drafter):
    """Build a minimal mock self for _check_and_update_cudagraph_mode.

    ``spec_flags`` maps the SpeculativeConfig predicate names
    (use_eagle / uses_draft_model / uses_extract_hidden_states) to booleans;
    ``drafter`` is the (mock) proposer instance the gate should dispatch to.
    """
    resolved_mode = object()  # opaque sentinel passed through to the drafter
    speculative_config = SimpleNamespace(
        use_eagle=lambda: spec_flags.get("use_eagle", False),
        uses_draft_model=lambda: spec_flags.get("uses_draft_model", False),
        uses_extract_hidden_states=lambda: spec_flags.get(
            "uses_extract_hidden_states", False
        ),
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=lambda *a, **k: resolved_mode
        ),
        uniform_decode_query_len=1,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        kv_cache_config=None,
        max_num_reqs=8,
        cudagraph_dispatcher=mock.MagicMock(),
        speculative_config=speculative_config,
        drafter=drafter,
    )
    return runner, resolved_mode


def test_draft_model_initializes_drafter_cudagraph_keys():
    """draft_model must initialize the drafter's cudagraph keys (the fix)."""
    drafter = mock.MagicMock(spec=DraftModelProposer)
    runner, resolved_mode = _make_runner({"uses_draft_model": True}, drafter)

    GPUModelRunner._check_and_update_cudagraph_mode(runner, [], [])

    drafter.initialize_cudagraph_keys.assert_called_once_with(resolved_mode)


def test_eagle_still_initializes_drafter_cudagraph_keys():
    """Guard: the EAGLE path keeps initializing the drafter's keys."""
    drafter = mock.MagicMock(spec=EagleProposer)
    runner, resolved_mode = _make_runner({"use_eagle": True}, drafter)

    GPUModelRunner._check_and_update_cudagraph_mode(runner, [], [])

    drafter.initialize_cudagraph_keys.assert_called_once_with(resolved_mode)


@pytest.mark.parametrize("predicate", ["use_eagle", "uses_draft_model"])
def test_gate_matches_isinstance_assert(predicate):
    """A drafter selected by the gate must satisfy the isinstance assert."""
    drafter_cls = {
        "use_eagle": EagleProposer,
        "uses_draft_model": DraftModelProposer,
    }[predicate]
    drafter = mock.MagicMock(spec=drafter_cls)
    runner, _ = _make_runner({predicate: True}, drafter)

    # Would raise AssertionError if the drafter type were excluded from the
    # isinstance union while the gate predicate is enabled.
    GPUModelRunner._check_and_update_cudagraph_mode(runner, [], [])
    drafter.initialize_cudagraph_keys.assert_called_once()
