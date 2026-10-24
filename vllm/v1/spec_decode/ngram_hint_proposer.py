# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def find_hint_match(
    hints: list[list[int]],
    context: np.ndarray,
    max_n: int,
    k: int,
) -> list[int]:
    """Find the longest suffix of the context in one of the hints and return
    the tokens that follow it.

    Args:
        hints: Token sequences supplied with the request.
        context: The most recent token ids of the request.
        max_n: Maximum length of the suffix to look for.
        k: Maximum number of tokens to propose.

    Returns:
        Up to k tokens following the first match, or an empty list.

    """
    if not hints or k <= 0 or context.size == 0:
        return []

    n_key = min(max_n, context.size)
    key = context[-n_key:]

    for n in range(n_key, 0, -1):
        sub = list(key[-n:])
        for hint in hints:
            if len(hint) <= n:
                continue
            # A single token only matches at the start of a hint. Otherwise a
            # common token like a newline would enter a hint in the middle and
            # waste a verification step.
            # A match at the very end of the hint has nothing to propose.
            end = 1 if n == 1 else len(hint) - n
            for i in range(end):
                if hint[i : i + n] == sub:
                    return hint[i + n : i + n + k]
    return []


def _get_hints(req_state: CachedRequestState | None) -> list[list[int]]:
    if req_state is None or req_state.sampling_params is None:
        return []
    extra_args = req_state.sampling_params.extra_args
    if not extra_args:
        return []
    hints = extra_args.get("spec_hints")
    if not hints:
        return []
    # vllm_xargs only allows a flat list, so accept a single sequence too.
    if isinstance(hints[0], int):
        hints = [hints]
    return [list(hint) for hint in hints if hint]


class NgramHintProposer:
    """Speculative decoding proposer that drafts from token sequences supplied
    with each request on SamplingParams.extra_args["spec_hints"], for example
    the chat template's rendering of each tool call. The n-gram proposers only
    find repeats of the context in the context itself, which the first tool
    call of a conversation does not have.
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        assert config.prompt_lookup_max is not None
        self.num_speculative_tokens = config.num_speculative_tokens
        # Maximum length of the context suffix to match against the hints.
        self.max_n = config.prompt_lookup_max
        self.max_model_len = vllm_config.model_config.max_model_len

    def propose(
        self,
        num_speculative_tokens: int,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        requests: dict[str, CachedRequestState],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            hints = _get_hints(requests.get(req_id))
            if not hints:
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            start = max(0, num_tokens - self.max_n)
            context = input_batch.token_ids_cpu[index, start:num_tokens]
            k = min(num_speculative_tokens, self.max_model_len - num_tokens - 1)
            draft_token_ids.append(find_hint_match(hints, context, self.max_n, k))

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
