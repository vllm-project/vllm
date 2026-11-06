# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams


@pytest.mark.parametrize("enforce_eager", [False, True])
def test_buffered_pipeline_matches_fixed_pipeline(
    vllm_runner, monkeypatch, enforce_eager
):
    """A real checkpoint exercises runner outputs, graph storage, and sampling."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    prompts = ["Explain pipeline parallelism.", "What is 17 plus 25?"]
    params = SamplingParams(temperature=0, seed=7, max_tokens=4, logprobs=5)

    def generate(additional_config):
        with vllm_runner(
            "Qwen/Qwen2.5-0.5B-Instruct",
            pipeline_parallel_size=2,
            tensor_parallel_size=1,
            distributed_executor_backend="mp",
            enforce_eager=enforce_eager,
            max_num_seqs=2,
            max_model_len=512,
            enable_prefix_caching=False,
            additional_config=additional_config,
        ) as runner:
            outputs = runner.generate_w_logprobs(prompts, params)
            stats = runner.model.collective_rpc("get_dynamic_pp_stats")
            return outputs, stats

    expected, _ = generate({})
    actual, stats = generate({"pp_send_buffer_size": 2})
    assert any(row.get("buffer_sends", 0) > 0 for row in stats)
    check_logprobs_close(
        outputs_0_lst=actual,
        outputs_1_lst=expected,
        name_0="buffered PP",
        name_1="fixed PP",
        always_check_logprobs=True,
    )
