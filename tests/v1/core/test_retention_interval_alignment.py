# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The retention interval is validated against the *effective* hit granularity.

A retention checkpoint is only useful where a prefix-cache hit can actually
land, so the interval must be a multiple of the alignment a hit is reported at.
That alignment is ``scheduler_block_size`` only when fine-grained partial hash
hits are off; when they are on it is ``hash_block_size``, which under decode
context parallelism is ``scheduler_block_size // dcp_world_size``.

Validating against ``scheduler_block_size`` unconditionally rejects intervals
that are legal, and does so exactly where the feature is most useful: at DCP=8
a group whose hits land every 1536 tokens is forced to an 8x coarser 12288.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.core import kv_cache_coordinator as coord
from vllm.v1.core.kv_cache_coordinator import (
    _validate_prefix_cache_retention_interval,
    _validate_retention_alignment,
)


@pytest.fixture
def on_rocm(monkeypatch):
    """Run as if on ROCm, whatever the CI runner actually is."""
    monkeypatch.setattr(
        coord, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )


@pytest.fixture
def off_rocm(monkeypatch):
    monkeypatch.setattr(
        coord, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )


# --------------------------------------------------------------------------
# alignment check
# --------------------------------------------------------------------------


def test_interval_matching_the_hit_alignment_is_accepted(on_rocm):
    """The DCP case that used to be rejected: hits land every 1536 tokens, so
    1536 is exactly the right interval."""
    _validate_retention_alignment(1536, alignment_tokens=1536)


def test_interval_that_cannot_land_on_a_hit_boundary_is_rejected(on_rocm):
    with pytest.raises(ValueError, match="multiple of the cache-hit alignment"):
        _validate_retention_alignment(1000, alignment_tokens=1536)


@pytest.mark.parametrize("interval", [None, 0])
def test_dense_and_latest_only_skip_the_check(on_rocm, interval):
    """None means dense and 0 means keep only the latest replay boundary;
    neither describes a spacing, so alignment does not apply."""
    _validate_retention_alignment(interval, alignment_tokens=1536)


def test_a_multiple_of_the_alignment_is_accepted(on_rocm):
    _validate_retention_alignment(4608, alignment_tokens=1536)


def test_coarser_scheduler_block_no_longer_forces_a_coarser_interval(on_rocm):
    """The regression this fixes.

    Under DCP=8 the full-attention group's scheduler_block_size is scaled to
    12288 while hits are still reported every hash_block_size = 1536. Checking
    against 12288 rejected 1536; checking against the real alignment accepts it.
    """
    dcp_world_size = 8
    hash_block_size = 1536
    scheduler_block_size = hash_block_size * dcp_world_size

    _validate_retention_alignment(1536, alignment_tokens=hash_block_size)

    with pytest.raises(ValueError):
        _validate_retention_alignment(1536, alignment_tokens=scheduler_block_size)


# --------------------------------------------------------------------------
# the base validator keeps its alignment-independent checks
# --------------------------------------------------------------------------


def _kv_cache_config():
    """A config with one sliding-window group, which is what makes a retention
    interval meaningful at all (the base validator rejects it outright for
    models with no sliding-window or Mamba group)."""
    from types import SimpleNamespace

    import torch

    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    spec = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=128,
    )
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)])


def test_negative_interval_is_still_rejected(on_rocm):
    with pytest.raises(ValueError, match="must be non-negative"):
        _validate_prefix_cache_retention_interval(-1, 1536, _kv_cache_config())


def test_the_base_validator_no_longer_checks_alignment_on_rocm(on_rocm):
    """Alignment moved to _validate_retention_alignment, which runs once the
    concrete coordinator has settled enable_partial_hash_hits. If the base
    validator still rejected here, the DCP case could never be accepted no
    matter what the real granularity turned out to be."""
    _validate_prefix_cache_retention_interval(1536, 12288, _kv_cache_config())


def test_the_two_coordinators_source_their_alignment_correctly():
    """A regression guard with a scar behind it.

    ``_cache_hit_alignment_tokens`` is a HybridKVCacheCoordinator property.
    Calling it on the unitary coordinator raises AttributeError at construction,
    which is a boot failure for every single-group model -- exactly what an
    earlier draft of this change did. The unitary path has no partial hash hits,
    so scheduler_block_size is both correct and available on the base class.
    """
    from vllm.v1.core.kv_cache_coordinator import (
        HybridKVCacheCoordinator,
        UnitaryKVCacheCoordinator,
    )

    assert "_cache_hit_alignment_tokens" in vars(HybridKVCacheCoordinator)
    assert "_cache_hit_alignment_tokens" not in vars(UnitaryKVCacheCoordinator)


# --------------------------------------------------------------------------
# off ROCm: behaviour is unchanged from before this PR
# --------------------------------------------------------------------------


def test_off_rocm_the_base_validator_still_checks_scheduler_block_size(off_rocm):
    """Gating the new alignment check must not leave other platforms with no
    alignment validation at all -- that would be weaker than the status quo,
    not safer. Off ROCm the original check still applies."""
    with pytest.raises(ValueError, match="multiple of scheduler_block_size"):
        _validate_prefix_cache_retention_interval(1536, 12288, _kv_cache_config())


def test_off_rocm_a_scheduler_aligned_interval_is_still_accepted(off_rocm):
    _validate_prefix_cache_retention_interval(12288, 12288, _kv_cache_config())


def test_off_rocm_the_deferred_check_is_inert(off_rocm):
    """The base validator has already decided; a second, different check here
    would double-validate with the wrong value."""
    _validate_retention_alignment(1536, alignment_tokens=12288)
