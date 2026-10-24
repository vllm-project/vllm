# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix caching for PrefixLM requests, which send their prefix length in
`extra_args["prefix_lm_len"]`: salts each prefix's blocks by its digest, as the KV
of every prefix token depends on the whole prefix, caps hits at the prefix and
prefills the rest of the prompt in one step."""

import copy
import hashlib
from collections.abc import Sequence

import numpy as np

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)

PREFIX_LM_LEN = "prefix_lm_len"
"""`SamplingParams.extra_args` key of the length of a prompt's PrefixLM prefix."""

PREFIX_LM_SALT = "prefix-lm:"
"""Start of the `cache_salt` of every PrefixLM request, and of no other."""


def prefix_lm_len(
    params: SamplingParams | None, prompt_token_ids: Sequence[int] | None
) -> int | None:
    """The valid PrefixLM prefix length a request sends, else None."""
    extra_args = params.extra_args if params is not None else None
    prefix_len = extra_args.get(PREFIX_LM_LEN) if extra_args else None
    if (
        type(prefix_len) is int
        and prompt_token_ids is not None
        and 0 < prefix_len <= len(prompt_token_ids)
    ):
        return prefix_len
    return None


def prefix_lm_salt(prefix_token_ids: Sequence[int], cache_salt: str | None) -> str:
    """PrefixLM `cache_salt`: namespace, prefix digest, then the caller's salt."""
    digest = hashlib.sha256(
        np.asarray(prefix_token_ids, dtype="<i8").tobytes()
    ).hexdigest()
    salt = PREFIX_LM_SALT + digest
    return f"{salt}:{cache_salt}" if cache_salt else salt


def _request_prefix_lm_len(request: Request) -> int | None:
    return prefix_lm_len(request.sampling_params, request.prompt_token_ids)


class PrefixLMScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Scheduler calls _mamba_block_aligned_split only in this mode.
        if not self.need_mamba_block_aligned_split:
            raise ValueError(
                f"{type(self).__name__} needs a hybrid model in Mamba 'align' "
                "cache mode"
            )

    def add_request(self, request: Request) -> None:
        params = request.sampling_params
        if (
            (request.resumable or request.mm_features)
            and params is not None
            and PREFIX_LM_LEN in (params.extra_args or {})
        ):
            # Later inputs and encoder budgets can cut such a prompt short.
            logger.warning_once(
                "Requests with streamed or multimodal inputs ignore %s",
                PREFIX_LM_LEN,
            )
            request.sampling_params = params = copy.copy(params)
            params.extra_args = dict(params.extra_args or {})
            del params.extra_args[PREFIX_LM_LEN]
        # A streaming update re-adds an existing request.
        if request.request_id not in self.requests:
            self._salt_by_kind(request)
        super().add_request(request)

    def _salt_by_kind(self, request: Request) -> None:
        """Salt PrefixLM requests by prefix; keep causal requests out."""
        prefix_len = _request_prefix_lm_len(request)
        salt = request.cache_salt
        if prefix_len is None:
            if not (salt and salt.startswith(PREFIX_LM_SALT)):
                return
            salt = f"causal:{salt}"
        else:
            assert request.prompt_token_ids is not None
            own = prefix_lm_salt(request.prompt_token_ids[:prefix_len], None)
            if salt == own or (salt and salt.startswith(f"{own}:")):
                return
            salt = f"{own}:{salt}" if salt else own
        request.cache_salt = salt
        request.block_hashes = []
        request.update_block_hashes()

    def _get_local_prefix_cache_hit(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int, bool]:
        prefix_len = _request_prefix_lm_len(request)
        if prefix_len is None:
            return super()._get_local_prefix_cache_hit(request)
        # KV past the prefix depends on the rest of the prompt.
        return (*self.kv_cache_manager.get_computed_blocks(request, prefix_len), False)

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        prefix_lm = (
            start < request.num_prompt_tokens
            and _request_prefix_lm_len(request) is not None
        )
        if (
            prefix_lm
            and start % self.cache_config.block_size == 0
            and start + num_new_tokens == request.num_tokens
        ):
            # A whole prompt from a block boundary needs no Mamba split, which
            # would stop at a shared-prefix junction.
            return num_new_tokens
        num_new_tokens = super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if prefix_lm and start + num_new_tokens < request.num_prompt_tokens:
            # Zero keeps the request waiting at the head of the queue.
            return 0
        return num_new_tokens


class PrefixLMAsyncScheduler(PrefixLMScheduler, AsyncScheduler): ...
