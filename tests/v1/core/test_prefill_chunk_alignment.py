# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.core.sched.prefill_chunk_alignment import (
    DefaultPrefillChunkAlignmentPolicy,
    MambaPrefillChunkAlignmentPolicy,
    create_prefill_chunk_alignment_policy,
)


class _FakeRequest:
    def __init__(
        self,
        num_prompt_tokens: int,
        num_tokens: int,
        num_computed_tokens: int = 0,
    ) -> None:
        self.num_prompt_tokens = num_prompt_tokens
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed_tokens


@pytest.mark.parametrize("use_eagle", [False, True])
def test_mamba_align_scheduled_tokens_block_aligns(use_eagle: bool):
    policy = MambaPrefillChunkAlignmentPolicy(block_size=16, use_eagle=use_eagle)
    req = _FakeRequest(num_prompt_tokens=100, num_tokens=100)
    # Far from the tail: must be a multiple of block_size.
    assert policy.align_scheduled_tokens(req, 30) == 16


def test_mamba_align_scheduled_tokens_forces_last_chunk_no_eagle():
    policy = MambaPrefillChunkAlignmentPolicy(block_size=16, use_eagle=False)
    req = _FakeRequest(num_prompt_tokens=100, num_tokens=100, num_computed_tokens=80)
    # last_cache_position == 96; chunk crosses it -> snap to 96 - 80 == 16.
    assert policy.align_scheduled_tokens(req, 20) == 16


def test_mamba_align_scheduled_tokens_eagle_pulls_back_one_block():
    policy = MambaPrefillChunkAlignmentPolicy(block_size=16, use_eagle=True)
    req = _FakeRequest(num_prompt_tokens=100, num_tokens=100, num_computed_tokens=64)
    # Eagle: last_cache_position drops from 96 to 80; snap to 80 - 64 == 16.
    assert policy.align_scheduled_tokens(req, 20) == 16


def test_mamba_align_scheduled_tokens_passes_through_tail():
    policy = MambaPrefillChunkAlignmentPolicy(block_size=16, use_eagle=False)
    req = _FakeRequest(num_prompt_tokens=100, num_tokens=100, num_computed_tokens=96)
    # Past last_cache_position == 96: prefill the last few tokens unchanged.
    assert policy.align_scheduled_tokens(req, 4) == 4


def test_mamba_align_external_cached_tokens_rejects_nonzero():
    policy = MambaPrefillChunkAlignmentPolicy(block_size=16, use_eagle=False)
    req = _FakeRequest(num_prompt_tokens=100, num_tokens=100)
    with pytest.raises(AssertionError, match="External KV connector"):
        policy.align_external_cached_tokens(
            req, num_local_cached_tokens=0, num_external_cached_tokens=16
        )


def test_default_policy_is_noop():
    policy = DefaultPrefillChunkAlignmentPolicy()
    req = _FakeRequest(num_prompt_tokens=100, num_tokens=100)
    assert policy.align_scheduled_tokens(req, 30) == 30
    assert (
        policy.align_external_cached_tokens(
            req, num_local_cached_tokens=8, num_external_cached_tokens=24
        )
        == 24
    )


def test_factory_returns_mamba_policy_only_when_align_mode():
    mamba_policy = create_prefill_chunk_alignment_policy(
        has_mamba_layers=True,
        mamba_cache_mode="align",
        block_size=16,
        use_eagle=False,
    )
    assert isinstance(mamba_policy, MambaPrefillChunkAlignmentPolicy)

    for has_mamba, mode in [
        (False, "align"),
        (True, "all"),
        (True, "none"),
        (False, "none"),
    ]:
        default_policy = create_prefill_chunk_alignment_policy(
            has_mamba_layers=has_mamba,
            mamba_cache_mode=mode,
            block_size=16,
            use_eagle=False,
        )
        assert isinstance(default_policy, DefaultPrefillChunkAlignmentPolicy)
