# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaForConditionalGeneration,
    DiffusionGemmaModelState,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker


class _PromptModel(torch.nn.Module):
    compute_logits = DiffusionGemmaForConditionalGeneration.compute_logits
    compute_prompt_logits = DiffusionGemmaForConditionalGeneration.compute_prompt_logits

    def __init__(self) -> None:
        super().__init__()
        self.num_scored_tokens = 0
        self.lm_head = torch.nn.Linear(4, 32, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(128).view(32, 4) / 128)
        self.final_logit_softcapping = 2.5
        self.diffusion_states = SimpleNamespace(step_allowed=torch.tensor([10, 20]))

    def logits_processor(self, head, hidden):
        self.num_scored_tokens += len(hidden)
        return head(hidden)

    def reference(self, hidden):
        return 2.5 * torch.tanh(self.lm_head(hidden) / 2.5)


def _score_batch(computed, lengths, query_lens, device) -> SimpleNamespace:
    n = len(lengths)
    query_start_loc = np.concatenate(([0], np.cumsum(query_lens))).astype(np.int32)
    return SimpleNamespace(
        num_reqs=n,
        num_tokens=int(query_start_loc[-1]),
        req_ids=[str(i) for i in range(n)],
        idx_mapping_np=np.arange(n),
        idx_mapping=torch.arange(n, dtype=torch.int32, device=device),
        num_computed_prefill_tokens_np=np.array(computed),
        prefill_len_np=np.array(lengths),
        num_scheduled_tokens=np.array(query_lens),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc).to(device),
    )


def _score_prompt(worker, logits_fn, hidden, token_ids, computed, length):
    device = hidden.device
    return worker.compute_prompt_logprobs(
        logits_fn,
        hidden,
        _score_batch([computed], [length], [len(hidden)], device),
        token_ids,
        torch.tensor([computed], dtype=torch.int32, device=device),
        np.array([length]),
    )


def _score_last_prompt_token(worker, logits_fn, device, length):
    return _score_prompt(
        worker,
        logits_fn,
        torch.zeros(9, 4, device=device),
        torch.zeros(1, 32, dtype=torch.int32, device=device),
        length - 1,
        length,
    )


def _score_in_chunks(worker, logits_fn, hidden, token_ids, chunks, num_drafts):
    length, computed = len(hidden), 0
    for chunk in chunks:
        end = computed + chunk
        query = torch.cat(
            (hidden[computed:end], hidden[: num_drafts if end == length else 0])
        )
        result = _score_prompt(worker, logits_fn, query, token_ids, computed, length)
        assert end == length or result == {}
        computed = end
    return result["0"]


@pytest.fixture
def device():
    return torch.device(current_platform.device_type)


@pytest.fixture
def prompt_model(device):
    model = _PromptModel().to(device)
    model.diffusion_states.step_allowed = model.diffusion_states.step_allowed.to(device)
    return model


@pytest.fixture
def prompt_logits_fn(prompt_model):
    state = SimpleNamespace(model=prompt_model)
    return partial(DiffusionGemmaModelState.compute_prompt_logits, state)


@pytest.fixture
def logprobs_worker(device):
    worker = PromptLogprobsWorker(1, device)
    worker.add_request("0", 0, SamplingParams(prompt_logprobs=0))
    return worker


class TestPromptTokenIdScores:
    @pytest.mark.parametrize("num_drafts", [0, 8])
    def test_fixed_prompt_scores_ignore_the_step_vocabulary(
        self, device, prompt_model, prompt_logits_fn, num_drafts
    ):
        ids = np.tile([10, 20, 31], (7, 1))
        worker = PromptLogprobsWorker(1, device, "raw_logits")
        worker.add_request("0", 0, SamplingParams(prompt_logprob_token_ids=ids))
        hidden = torch.arange((8 + num_drafts) * 4, dtype=torch.float32, device=device)
        hidden = hidden.view(-1, 4) / 64
        result = worker.compute_prompt_token_id_logprobs(
            prompt_logits_fn,
            hidden,
            _score_batch([0], [8], [8 + num_drafts], device),
            np.array([8]),
        )
        expected = prompt_model.reference(hidden[:7])[:, [10, 20, 31]]
        torch.testing.assert_close(result["0"], expected)
        assert prompt_model.num_scored_tokens == 7


class TestPromptLogprobs:
    def test_one_token_prompt_scores_nothing(
        self, device, logprobs_worker, prompt_logits_fn, prompt_model
    ):
        result = _score_last_prompt_token(
            logprobs_worker, prompt_logits_fn, device, length=1
        )
        assert result == {}
        assert prompt_model.num_scored_tokens == 0

    def test_last_prompt_token_flushes_saved_scores(
        self, device, logprobs_worker, prompt_logits_fn, prompt_model
    ):
        saved = LogprobsTensors(
            torch.arange(7, device=device).view(-1, 1),
            torch.zeros(7, 1, device=device),
            torch.ones(7, dtype=torch.int64, device=device),
        )
        logprobs_worker.in_progress_prompt_logprobs["0"].append(saved)
        scores = _score_last_prompt_token(
            logprobs_worker, prompt_logits_fn, device, length=8
        )["0"]
        torch.testing.assert_close(scores.logprobs, saved.logprobs)
        torch.testing.assert_close(scores.logprob_token_ids, saved.logprob_token_ids)
        assert not logprobs_worker.in_progress_prompt_logprobs["0"]
        assert prompt_model.num_scored_tokens == 0

    @pytest.mark.parametrize("num_drafts", [0, 8])
    @pytest.mark.parametrize("chunks", [[8], [4, 4], [7, 1]])
    @pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton kernels")
    @pytest.mark.parametrize(
        "mode, normalize",
        [("raw_logits", lambda x: x), ("raw_logprobs", lambda x: x.log_softmax(-1))],
        ids=["raw_logits", "raw_logprobs"],
    )
    def test_chunked_scores_match_the_reference(
        self,
        device,
        prompt_model,
        prompt_logits_fn,
        num_drafts,
        chunks,
        mode,
        normalize,
    ):
        worker = PromptLogprobsWorker(1, device, mode)
        worker.add_request("0", 0, SamplingParams(prompt_logprobs=2))
        token_ids = torch.zeros(1, 32, dtype=torch.int32, device=device)
        token_ids[0, :8] = torch.arange(10, 18)
        hidden = torch.arange(32, dtype=torch.float32, device=device).view(8, 4) / 32
        scores = _score_in_chunks(
            worker, prompt_logits_fn, hidden, token_ids, chunks, num_drafts
        )
        reference = normalize(prompt_model.reference(hidden[:7]))
        assert scores.logprobs.shape == (7, 3)
        assert prompt_model.num_scored_tokens == 7
        torch.testing.assert_close(
            scores.logprob_token_ids[:, 0], token_ids[0, 1:8].long()
        )
        torch.testing.assert_close(
            scores.logprob_token_ids[:, 1:], reference.topk(2, dim=-1).indices
        )
        torch.testing.assert_close(
            scores.logprobs, reference.gather(1, scores.logprob_token_ids)
        )

    @pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton kernels")
    @pytest.mark.parametrize("padding", [0, 4])
    def test_unfinished_prefills_score_without_copying_hidden_states(
        self, device, prompt_model, prompt_logits_fn, padding
    ):
        batch = _score_batch([0, 0], [6, 8], [3, 4], device)
        worker = PromptLogprobsWorker(2, device)
        for slot, req_id in enumerate(batch.req_ids):
            worker.add_request(req_id, slot, SamplingParams(prompt_logprobs=0))
        hidden = torch.zeros(7 + padding, 4, device=device)

        def logits_fn(scored_hidden):
            assert (
                scored_hidden.untyped_storage().data_ptr()
                == hidden.untyped_storage().data_ptr()
            )
            return prompt_logits_fn(scored_hidden)

        result = worker.compute_prompt_logprobs(
            logits_fn,
            hidden,
            batch,
            torch.zeros(2, 8, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
            np.array([6, 8]),
        )
        assert result == {}
        assert prompt_model.num_scored_tokens == 7
        assert [
            worker.in_progress_prompt_logprobs[req_id][0].logprobs.shape[0]
            for req_id in batch.req_ids
        ] == [3, 4]

    @pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton kernels")
    @pytest.mark.parametrize("topk", [(0, 2), (2, -1), (-1, 0)])
    def test_mixed_batch_projects_only_requested_prompt_rows(
        self, device, prompt_model, prompt_logits_fn, topk
    ):
        batch = _score_batch([0, 4, 0, 2], [4, 4, 4, 6], [12, 1, 4, 2], device)
        batch.idx_mapping_np = np.array([2, 0, 3, 1])
        batch.idx_mapping = torch.tensor([2, 0, 3, 1], device=device)
        worker = PromptLogprobsWorker(4, device)
        for req_id, slot, k in zip(
            batch.req_ids, batch.idx_mapping_np, [topk[0], -1, None, topk[1]]
        ):
            worker.add_request(req_id, slot, SamplingParams(prompt_logprobs=k))
        lengths = np.empty(4, dtype=np.int32)
        lengths[batch.idx_mapping_np] = batch.prefill_len_np
        computed = np.empty(4, dtype=np.int32)
        computed[batch.idx_mapping_np] = batch.num_computed_prefill_tokens_np
        token_ids = torch.arange(32, dtype=torch.int32, device=device).view(4, 8)
        hidden = (
            torch.arange(batch.num_tokens * 4, dtype=torch.float32, device=device).view(
                -1, 4
            )
            / 64
        )

        result = worker.compute_prompt_logprobs(
            prompt_logits_fn,
            hidden,
            batch,
            token_ids,
            torch.from_numpy(computed).to(device),
            lengths,
        )

        assert set(result) == {"0"}
        assert prompt_model.num_scored_tokens == 5
        saved = worker.in_progress_prompt_logprobs["3"]
        assert len(saved) == 1
        for scores, start, end, slot, offset, k in [
            (result["0"], 0, 3, 2, 1, topk[0]),
            (saved[0], 17, 19, 1, 3, topk[1]),
        ]:
            reference = prompt_model.reference(hidden[start:end]).log_softmax(-1)
            num_topk = reference.shape[-1] if k == -1 else k
            expected_ids = torch.cat(
                (
                    token_ids[slot, offset : offset + end - start, None].long(),
                    reference.topk(num_topk, dim=-1).indices,
                ),
                dim=1,
            )
            torch.testing.assert_close(scores.logprob_token_ids, expected_ids)
            torch.testing.assert_close(
                scores.logprobs, reference.gather(1, expected_ids)
            )
            torch.testing.assert_close(
                scores.selected_token_ranks,
                (reference >= reference.gather(1, expected_ids[:, :1])).sum(-1),
            )
