# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Protocol

from vllm.config.cache import MambaCacheMode
from vllm.v1.request import Request


class PrefillChunkAlignmentPolicy(Protocol):
    def align_external_cached_tokens(
        self,
        request: Request,
        *,
        num_local_cached_tokens: int,
        num_external_cached_tokens: int,
    ) -> int:
        """Return externally cached tokens usable after policy alignment.

        The scheduler treats this value as authoritative for connector
        prefix-cache hit metrics and async-load gating.
        """
        ...

    def align_scheduled_tokens(
        self,
        request: Request,
        num_scheduled_tokens: int,
        *,
        num_local_cached_tokens: int = 0,
        num_external_cached_tokens: int = 0,
    ) -> int:
        """Return the number of tokens to schedule after policy alignment.

        `num_local_cached_tokens` and `num_external_cached_tokens` are 0 on
        the running-queue path (already reflected in
        `request.num_computed_tokens`) and non-zero on the waiting-queue path.
        """
        ...


class DefaultPrefillChunkAlignmentPolicy:
    """Default policy for models without prefill chunk alignment constraints."""

    def align_external_cached_tokens(
        self,
        request: Request,
        *,
        num_local_cached_tokens: int,
        num_external_cached_tokens: int,
    ) -> int:
        return num_external_cached_tokens

    def align_scheduled_tokens(
        self,
        request: Request,
        num_scheduled_tokens: int,
        *,
        num_local_cached_tokens: int = 0,
        num_external_cached_tokens: int = 0,
    ) -> int:
        return num_scheduled_tokens


class MambaPrefillChunkAlignmentPolicy:
    """Align Mamba align-mode prefill chunks to cache block boundaries.

    In EAGLE mode, the final cacheable chunk is backed up by one block because
    FullAttn prunes the last matching block.
    """

    def __init__(self, *, block_size: int, use_eagle: bool) -> None:
        self.block_size = block_size
        self.use_eagle = use_eagle

    def align_external_cached_tokens(
        self,
        request: Request,
        *,
        num_local_cached_tokens: int,
        num_external_cached_tokens: int,
    ) -> int:
        assert num_external_cached_tokens == 0, (
            "External KV connector is not verified yet"
        )
        return num_external_cached_tokens

    def align_scheduled_tokens(
        self,
        request: Request,
        num_scheduled_tokens: int,
        *,
        num_local_cached_tokens: int = 0,
        num_external_cached_tokens: int = 0,
    ) -> int:
        num_computed_tokens = (
            request.num_computed_tokens
            + num_local_cached_tokens
            + num_external_cached_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, scheduled tokens
            # must be a multiple of `block_size`.
            # As an exception, if the scheduled token count is less than
            # `block_size`, the state is simply not cached, requiring no special
            # handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            last_cache_position = (
                request.num_tokens - request.num_tokens % self.block_size
            )
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - self.block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_scheduled_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_scheduled_tokens = (
                    num_scheduled_tokens // self.block_size * self.block_size
                )
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_scheduled_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_scheduled_tokens


def create_prefill_chunk_alignment_policy(
    *,
    has_mamba_layers: bool,
    mamba_cache_mode: MambaCacheMode,
    block_size: int,
    use_eagle: bool,
) -> PrefillChunkAlignmentPolicy:
    if has_mamba_layers and mamba_cache_mode == "align":
        return MambaPrefillChunkAlignmentPolicy(
            block_size=block_size,
            use_eagle=use_eagle,
        )
    return DefaultPrefillChunkAlignmentPolicy()
