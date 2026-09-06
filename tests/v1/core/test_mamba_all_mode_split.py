# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""_mamba_block_aligned_split: all-mode kernel-chunk clipping tests.

mamba_cache_mode='all' with a GDN/FLA-style prefill kernel (fixed chunk grid
relative to each scheduled chunk's start, no SSD-style short-first-chunk
realignment) requires prefill chunk STARTS to stay kernel-chunk (64) aligned so
every block-boundary SSM checkpoint is exactly materializable from the
kernel's per-chunk state export. The scheduler clips chunk ends to kernel-chunk
multiples — a much weaker constraint than align mode's block-size split.
"""

from types import SimpleNamespace

from vllm.config import CacheConfig
from vllm.v1.core.sched.scheduler import Scheduler


def _sched(mode="all", align_size=64, block_size=576, hash_block_size=576,
           use_eagle=False, partial_hit=False):
    """Minimal stand-in exposing the fields _mamba_block_aligned_split reads."""
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode=mode,
            mamba_all_mode_prefill_align_size=align_size,
            block_size=block_size,
        ),
        hash_block_size=hash_block_size,
        use_eagle=use_eagle,
        mamba_partial_cache_hit=partial_hit,
    )


def _req(num_computed=0, num_prompt=100_000, num_tokens=None,
         shared_prefix_boundary=0):
    return SimpleNamespace(
        num_computed_tokens=num_computed,
        num_prompt_tokens=num_prompt,
        num_tokens=num_tokens if num_tokens is not None else num_prompt,
        shared_prefix_boundary=shared_prefix_boundary,
    )


def _split(sched, req, num_new_tokens):
    return Scheduler._mamba_block_aligned_split(sched, req, num_new_tokens)


# ------------------------------ all-mode (NEW) ------------------------------


def test_all_mode_clips_end_to_chunk_multiple():
    """Mid-prompt bite ends get clipped down to a 64-multiple."""
    # 32,010 is not a 64-multiple; the nearest below is 32,000 (= 500*64).
    assert _split(_sched(), _req(num_computed=0), 32_010) == 32_000


def test_all_mode_aligned_budget_unchanged():
    """A 64-multiple budget passes through untouched."""
    assert _split(_sched(), _req(num_computed=0), 32_768) == 32_768


def test_all_mode_chunk_starts_stay_aligned_inductively():
    """After a clipped bite, the next start is 64-aligned again."""
    first = _split(_sched(), _req(num_computed=0), 1_000)
    assert first % 64 == 0
    second = _split(_sched(), _req(num_computed=first), 1_000)
    assert (first + second) % 64 == 0


def test_all_mode_small_budget_yields_empty_chunk():
    """Budget below one kernel chunk -> 0 (caller skips the request)."""
    assert _split(_sched(), _req(num_computed=64), 63) == 0


def test_all_mode_misaligned_start_realigns_at_next_boundary():
    """Externally misaligned start (e.g. loaded KV): the catch-up bite stops
    at the next 64-boundary so subsequent bites start aligned."""
    got = _split(_sched(), _req(num_computed=100), 32_000)
    assert 100 + got == 128  # next multiple of 64


def test_all_mode_final_chunk_unclipped():
    """The bite that finishes the prompt may end anywhere (tail is handled by
    kernel masking + the final-state write)."""
    assert _split(_sched(), _req(num_computed=99_968, num_prompt=100_000),
                  5_000) == 5_000


# --------------------------- align-mode regression ---------------------------


def test_align_mode_block_clip_unchanged():
    """REGRESSION: align mode still clips to block_size multiples."""
    got = _split(_sched(mode="align"), _req(num_computed=0), 32_000)
    assert got == (32_000 // 576) * 576


# ------------------------------- config field --------------------------------


def test_cache_config_field_default_none_and_not_init():
    cfg = CacheConfig()
    assert cfg.mamba_all_mode_prefill_align_size is None
    import dataclasses
    f = {f.name: f for f in dataclasses.fields(CacheConfig)}[
        "mamba_all_mode_prefill_align_size"]
    assert f.init is False
