# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.gpu.sample.prompt_logprob import CHUNK_SIZE, PromptLogprobsWorker


class _CountingLMHead:
    def __init__(self, device: torch.device) -> None:
        generator = torch.Generator().manual_seed(0)
        self.weight = torch.randn(32, 4, generator=generator).to(device)
        self.num_scored_tokens = 0

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        self.num_scored_tokens += len(hidden)
        return self.reference(hidden)

    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight.T


def _batch(computed, lengths, query_lens, device, num_padded_reqs=0):
    num_tokens = int(np.sum(query_lens))
    query_start_loc = np.concatenate(
        ([0], np.cumsum(query_lens), [num_tokens] * num_padded_reqs)
    ).astype(np.int32)
    return SimpleNamespace(
        num_reqs=len(lengths),
        num_tokens=num_tokens,
        req_ids=[str(i) for i in range(len(lengths))],
        idx_mapping_np=np.arange(len(lengths)),
        idx_mapping=torch.arange(len(lengths), dtype=torch.int32, device=device),
        num_computed_prefill_tokens_np=np.array(computed),
        prefill_len_np=np.array(lengths),
        num_scheduled_tokens=np.array(query_lens),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc).to(device),
    )


def _worker(device, num_prompt_logprobs):
    worker = PromptLogprobsWorker(len(num_prompt_logprobs), device)
    for slot, k in enumerate(num_prompt_logprobs):
        worker.add_request(str(slot), slot, SamplingParams(prompt_logprobs=k))
    return worker


def _score(worker, lm_head, hidden, batch, token_ids):
    return worker.compute_prompt_logprobs(
        lm_head,
        hidden,
        batch,
        token_ids,
        torch.tensor(
            batch.num_computed_prefill_tokens_np,
            dtype=torch.int32,
            device=hidden.device,
        ),
        batch.prefill_len_np,
    )


def _hidden(num_rows, device):
    generator = torch.Generator().manual_seed(1)
    return torch.randn(num_rows, 4, generator=generator).to(device)


def _token_ids(num_reqs, width, device):
    generator = torch.Generator().manual_seed(2)
    ids = torch.randint(32, (num_reqs, width), generator=generator)
    return ids.to(device, torch.int32)


def _expected(lm_head, hidden, target_ids, k):
    reference = lm_head.reference(hidden).log_softmax(-1)
    ids = torch.cat((target_ids[:, None].long(), reference.topk(k).indices), dim=1)
    return ids, reference.gather(1, ids)


@pytest.fixture
def device():
    return torch.device(current_platform.device_type)


@pytest.fixture
def lm_head(device):
    return _CountingLMHead(device)


@pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton kernels")
class TestPromptLogprobs:
    @pytest.mark.parametrize("chunks", [[8], [4, 4], [7, 1]])
    def test_chunked_prompt_matches_the_reference(self, device, lm_head, chunks):
        worker = _worker(device, [2])
        hidden = _hidden(8, device)
        token_ids = _token_ids(1, 32, device)
        computed = 0
        for chunk in chunks:
            end = computed + chunk
            batch = _batch([computed], [8], [chunk], device)
            result = _score(worker, lm_head, hidden[computed:end], batch, token_ids)
            computed = end
        ids, logprobs = _expected(lm_head, hidden[:7], token_ids[0, 1:8], 2)
        torch.testing.assert_close(result["0"].logprob_token_ids, ids)
        torch.testing.assert_close(result["0"].logprobs, logprobs)
        assert lm_head.num_scored_tokens == 7

    @pytest.mark.parametrize("num_padded_reqs", [0, 2])
    def test_mixed_batch_scores_only_requested_prompt_rows(
        self, device, lm_head, num_padded_reqs
    ):
        worker = _worker(device, [1, -1, None, 2])
        batch = _batch(
            [0, 4, 0, 2],
            [4, 4, 4, 6],
            [4, 1, 4, 2],
            device,
            num_padded_reqs,
        )
        hidden = _hidden(batch.num_tokens, device)
        token_ids = _token_ids(4, 32, device)
        result = _score(worker, lm_head, hidden, batch, token_ids)
        saved = worker.in_progress_prompt_logprobs["3"]
        assert set(result) == {"0"}
        assert len(saved) == 1
        assert lm_head.num_scored_tokens == 5
        last = batch.query_start_loc_np[3]
        for scores, ids, logprobs in [
            (result["0"], *_expected(lm_head, hidden[0:3], token_ids[0, 1:4], 1)),
            (
                saved[0],
                *_expected(lm_head, hidden[last : last + 2], token_ids[3, 3:5], 2),
            ),
        ]:
            torch.testing.assert_close(scores.logprob_token_ids, ids)
            torch.testing.assert_close(scores.logprobs, logprobs)

    def test_rows_gathered_across_chunks_match_the_reference(self, device, lm_head):
        length = CHUNK_SIZE + 2
        batch = _batch([0, 0], [3, length], [3, length], device)
        hidden = _hidden(3 + length, device)
        token_ids = _token_ids(2, length + 1, device)
        result = _score(_worker(device, [None, 0]), lm_head, hidden, batch, token_ids)
        ids, logprobs = _expected(lm_head, hidden[3:-1], token_ids[1, 1:length], 0)
        torch.testing.assert_close(result["1"].logprob_token_ids, ids)
        torch.testing.assert_close(result["1"].logprobs, logprobs)
        assert lm_head.num_scored_tokens == length - 1
