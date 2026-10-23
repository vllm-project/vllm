# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix caching for PrefixLM requests.

A PrefixLM request sends the length P of its prefix in
`extra_args["prefix_lm_len"]`. Its first P tokens attend to each other, as do
the rest of its prompt's tokens. So the KV and Mamba state of the prefix depend
on every prefix token, and those of the rest of the prompt on every prompt
token. PrefixLMScheduler therefore:

- salts the first block with the digest of the prefix, so a hit covers only the
  same prefix;
- caps prefix-cache hits at P;
- prefills the rest of the prompt after its hit in one step, or waits at the
  head of the queue.
"""

import copy
import hashlib
import sys
from array import array
from collections.abc import Sequence

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
    """The PrefixLM prefix length a request sends, or None if it sends no valid
    one, in which case the request is read causally."""
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
    """The `cache_salt` of a PrefixLM request with this prefix: the reserved
    namespace, the prefix digest, then the caller's own salt, if any."""
    ids = array("q", prefix_token_ids)
    if sys.byteorder != "little":
        ids.byteswap()
    salt = PREFIX_LM_SALT + hashlib.sha256(ids.tobytes()).hexdigest()
    return f"{salt}:{cache_salt}" if cache_salt else salt


def _request_prefix_lm_len(request: Request) -> int | None:
    return prefix_lm_len(request.sampling_params, request.prompt_token_ids)


class PrefixLMScheduler(Scheduler):
    """Scheduler of PrefixLM models served with prefix caching. Requests without
    a PrefixLM prefix length are scheduled as by `Scheduler`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The whole-prompt rule below rides on the Mamba alignment split.
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
            # Inputs that arrive later and encoder budgets can cut a prompt
            # short, so such requests are read causally.
            logger.warning_once(
                "Requests with streamed or multimodal inputs ignore %s",
                PREFIX_LM_LEN,
            )
            params = copy.copy(params)
            assert params.extra_args is not None
            params.extra_args = {
                k: v for k, v in params.extra_args.items() if k != PREFIX_LM_LEN
            }
            request.sampling_params = params
        if request.request_id not in self.requests:
            self._salt_by_kind(request)
        super().add_request(request)

    def _salt_by_kind(self, request: Request) -> None:
        """Give a PrefixLM request the `cache_salt` of its own prefix, and keep
        causal requests out of that namespace. The prefix's KV and Mamba state
        depend on all its tokens, and a causal request's do not."""
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
        blocks, num_local, shared_prefix_boundary = (
            self.kv_cache_manager.get_computed_blocks(request, prefix_len)
        )
        return blocks, num_local, shared_prefix_boundary, False

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
        if (
            start >= request.num_prompt_tokens
            or _request_prefix_lm_len(request) is None
        ):
            return super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        if (
            start % self.cache_config.block_size == 0
            and start + num_new_tokens == request.num_tokens
        ):
            # The whole prompt from a block boundary needs no split: the
            # prompt's last chunk may end inside a block, and a Mamba stop such
            # as a shared-prefix junction would cut the prompt.
            return num_new_tokens
        num_new_tokens = super()._mamba_block_aligned_split(
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if start + num_new_tokens < request.num_prompt_tokens:
            # The chunk would end inside a PrefixLM block. Zero makes the
            # request wait at the head of the queue.
            return 0
        return num_new_tokens


class PrefixLMAsyncScheduler(PrefixLMScheduler, AsyncScheduler):
    pass
