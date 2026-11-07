# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

NUM_DRAFTS = 4


def _batch(computed, prompt_len, query_len, device):
    query_start_loc = np.array([0, query_len], dtype=np.int32)
    return SimpleNamespace(
        num_reqs=1,
        num_tokens=query_len,
        req_ids=["0"],
        idx_mapping_np=np.arange(1),
        idx_mapping=torch.zeros(1, dtype=torch.int32, device=device),
        num_computed_prefill_tokens_np=np.array([computed]),
        prefill_len_np=np.array([prompt_len]),
        num_scheduled_tokens=np.array([query_len]),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc).to(device),
    )


@pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton kernels")
class TestPromptLogprobs:
    @pytest.mark.parametrize(
        "prompt_len, computed, num_drafts, prompt_logprob_start",
        [
            (8, 0, 0, 0),
            (8, 7, NUM_DRAFTS, 0),
            (1, 0, NUM_DRAFTS, 0),
            (8, 0, NUM_DRAFTS, 2),
        ],
    )
    def test_scores_stop_at_the_end_of_the_prompt(
        self, prompt_len, computed, num_drafts, prompt_logprob_start
    ):
        device = torch.device(current_platform.device_type)
        worker = PromptLogprobsWorker(1, device)
        worker.add_request(
            "0",
            0,
            SamplingParams(
                prompt_logprobs=0, prompt_logprob_start=prompt_logprob_start
            ),
        )
        query_len = prompt_len - computed + num_drafts
        result = worker.compute_prompt_logprobs(
            lambda hidden: torch.zeros(len(hidden), 32, device=device),
            torch.zeros(query_len, 4, device=device),
            _batch(computed, prompt_len, query_len, device),
            torch.zeros(1, 32, dtype=torch.int32, device=device),
            torch.tensor([computed], dtype=torch.int32, device=device),
            np.array([prompt_len]),
        )
        num_rows = result["0"].logprobs.shape[0] if result else 0
        assert num_rows == max(prompt_len - max(computed, prompt_logprob_start) - 1, 0)
