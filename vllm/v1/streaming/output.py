# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-turn output bookkeeping for bounded streaming sessions."""

from typing import Any, cast

from msgspec.structs import replace

from vllm.sampling_params import RequestOutputKind
from vllm.tokenizers import TokenizerLike
from vllm.utils.gc_utils import gc_untrack
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor
from vllm.v1.engine.output_processor import RequestState, StreamingUpdate


class BoundedRequestState(RequestState):
    _tokenizer: TokenizerLike | None
    _template: EngineCoreRequest

    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        *args: Any,
        **kwargs: Any,
    ) -> "BoundedRequestState":
        params = request.sampling_params
        if (
            params is None
            or params.n != 1
            or params.output_kind != RequestOutputKind.DELTA
        ):
            raise ValueError("Bounded streaming requires n=1 and DELTA outputs")
        state = cast(
            BoundedRequestState,
            super().from_new_request(tokenizer, request, *args, **kwargs),
        )
        state._tokenizer = tokenizer if params.detokenize else None
        state._template = replace(
            request,
            prompt_token_ids=None,
            prompt_embeds=None,
            prompt_is_token_ids=None,
            mm_features=None,
            block_hashes=None,
            sampling_params=params.clone(),
        )
        return state

    def apply_streaming_update(self, update: StreamingUpdate) -> None:
        # Each reply has its own decoder, including its UTF-8 and stop state.
        token_ids = update.prompt_token_ids
        self.prompt_token_ids = gc_untrack(
            list(token_ids) if token_ids is not None else []
        )
        self.prompt = update.prompt
        self.prompt_len = len(self.prompt_token_ids)
        request = replace(
            self._template,
            prompt_token_ids=self.prompt_token_ids,
            arrival_time=update.arrival_time,
        )
        cumulative_logprob = (
            self.logprobs_processor.cumulative_logprob
            if self.logprobs_processor is not None
            else None
        )
        self.detokenizer = IncrementalDetokenizer.from_new_request(
            tokenizer=self._tokenizer, request=request
        )
        self.logprobs_processor = LogprobsProcessor.from_new_request(
            tokenizer=self._tokenizer, request=request
        )
        if cumulative_logprob is not None:
            self.logprobs_processor.cumulative_logprob = cumulative_logprob
        self.sent_tokens_offset = 0
        self.routed_experts_chunks.clear()
        self.sampling_mask_chunks.clear()
        self.weight_version_starts.clear()
        self.num_cached_tokens = 0
        self.num_cache_creation_tokens = 0
        self.streaming_input = not update.final
        self.is_prefilling = True
        if self.stats is not None:
            self.stats.arrival_time = update.arrival_time
