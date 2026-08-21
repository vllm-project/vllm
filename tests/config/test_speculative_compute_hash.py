# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests that SpeculativeConfig.compute_hash covers num_speculative_tokens for
parallel-drafting methods.

Parallel drafting (DFlash, DSpark, P-EAGLE, PARD) emits the whole speculative
block in one forward pass, so ``num_speculative_tokens`` fixes tensor shapes in
the draft model's compiled graph -- DFlash2 derives
``block_size = 1 + num_speculative_tokens``. If the hash ignores it, a cached
compilation artifact built for one k is silently reused for another and the
engine dies at cudagraph capture with a shape assertion such as
``expected size 9==7, stride 5120==5120 at dim=1``.

Sequential methods (MTP, EAGLE, n-gram) re-run a fixed q=1 draft graph k times,
so their compiled graph does not depend on k; they must keep a stable hash so
that changing k does not force a needless recompilation.
"""

from typing import Any

import pytest

from vllm.config.speculative import SpeculativeConfig


def _config_with(
    method: str, num_speculative_tokens: int, parallel_drafting: bool
) -> Any:
    """Build a bare SpeculativeConfig exposing only what compute_hash reads.

    ``__post_init__`` resolves a real draft checkpoint, which a CPU-only unit
    test must not do, so the attributes are set directly.
    """
    config = object.__new__(SpeculativeConfig)
    config.__dict__.update(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        draft_model_config=None,
        parallel_drafting=parallel_drafting,
    )
    return config


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_parallel_drafting_hash_depends_on_num_speculative_tokens(method: str):
    """k changes the draft graph's shapes, so it must change the hash."""
    a = _config_with(method, 7, parallel_drafting=True)
    b = _config_with(method, 9, parallel_drafting=True)
    assert a.compute_hash() != b.compute_hash()


@pytest.mark.cpu_test
@pytest.mark.parametrize("method", ["mtp", "eagle", "eagle3"])
def test_sequential_drafting_hash_ignores_num_speculative_tokens(method: str):
    """k is only a loop count here, so the hash must stay stable."""
    a = _config_with(method, 7, parallel_drafting=False)
    b = _config_with(method, 9, parallel_drafting=False)
    assert a.compute_hash() == b.compute_hash()


@pytest.mark.cpu_test
def test_same_num_speculative_tokens_hashes_equal():
    """Identical configs must still hash identically (cache stays usable)."""
    a = _config_with("dflash", 7, parallel_drafting=True)
    b = _config_with("dflash", 7, parallel_drafting=True)
    assert a.compute_hash() == b.compute_hash()
