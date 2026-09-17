# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for Model Runner V2 bad words tests",
        allow_module_level=True,
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128

# Committed tokens: prompt [5], output [10, 11]. Draft tokens: [12, 13].
# The sampler passes input_ids gathered at logits_indices, so local position 0
# holds the last committed token (11) and draft tokens start at position 1.
PROMPT_LEN = 1
COMMITTED = [5, 10, 11]
INPUT_IDS = [11, 12, 13]
LOCAL_POS = [0, 1, 2]


def _make_state(bad_words_token_ids: list[list[int]]) -> tuple[BadWordsState, int]:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=PROMPT_LEN,
        all_token_ids=COMMITTED,
        num_computed_tokens=len(COMMITTED),
        max_tokens=32,
    )
    req_states.apply_staged_writes()

    req_idx = req_states.req_id_to_index["req"]
    state = BadWordsState(req_states)
    state.add_request(req_idx, SamplingParams(_bad_words_token_ids=bad_words_token_ids))
    state.apply_staged_writes()
    return state, req_idx


def _apply(bad_words_token_ids: list[list[int]]) -> torch.Tensor:
    state, req_idx = _make_state(bad_words_token_ids)
    num_logits = len(INPUT_IDS)
    logits = torch.zeros((num_logits, VOCAB_SIZE), device=DEVICE)
    idx_mapping_np = np.array([req_idx], dtype=np.intp)
    expanded_idx_mapping = torch.tensor(
        [req_idx] * num_logits, dtype=torch.int32, device=DEVICE
    )
    state.apply_bad_words(
        logits,
        expanded_idx_mapping,
        idx_mapping_np,
        torch.tensor(INPUT_IDS, dtype=torch.int32, device=DEVICE),
        torch.tensor(LOCAL_POS, dtype=torch.int32, device=DEVICE),
    )
    return logits.cpu()


def test_v2_bad_words_prefix_inside_draft_tokens():
    """A prefix matching entirely within the draft tokens must mask the bad
    word's last token at the draft position that completes the prefix."""
    out = _apply([[12, 13, 40]])
    expected = torch.zeros_like(out)
    expected[2, 40] = -float("inf")
    torch.testing.assert_close(out, expected)


def test_v2_bad_words_prefix_spanning_committed_and_draft_tokens():
    """A prefix spanning the committed/draft boundary must mask at the row
    where the prefix completes, not one draft position later."""
    out = _apply([[11, 12, 30]])
    expected = torch.zeros_like(out)
    expected[1, 30] = -float("inf")
    torch.testing.assert_close(out, expected)


def test_v2_bad_words_no_spurious_match_from_last_committed_token():
    """The last committed token must not be double-counted as the first draft
    token; [11, 11] never occurs in output [10, 11] + drafts [12, 13]."""
    out = _apply([[11, 11, 50]])
    expected = torch.zeros_like(out)
    torch.testing.assert_close(out, expected)


def test_v2_bad_words_committed_prefix():
    """Baseline: a fully committed prefix masks at the first row."""
    out = _apply([[10, 11, 60]])
    expected = torch.zeros_like(out)
    expected[0, 60] = -float("inf")
    torch.testing.assert_close(out, expected)
