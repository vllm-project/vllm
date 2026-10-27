# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.distributed.utils import get_pp_indices

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROMPT = "Explain how a pipeline processes independent batches. " * 256


def _generate(vllm_runner, additional):
    with vllm_runner(
        MODEL,
        pipeline_parallel_size=2,
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        enforce_eager=True,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        additional_config=additional,
    ) as runner:
        params = SamplingParams(temperature=0, seed=7, max_tokens=1, logprobs=5)
        outputs = [runner.generate_w_logprobs([PROMPT], params)[0] for _ in range(12)]
        stats = runner.model.collective_rpc("get_dynamic_pp_stats")
        config = runner.model.llm_engine.vllm_config.model_config.hf_config
        _, base = get_pp_indices(config.num_hidden_layers, 0, 2)
        return outputs, stats, base


@pytest.fixture(scope="module")
def fixed_outputs(vllm_runner):
    return _generate(vllm_runner, {})


@pytest.mark.parametrize("offset", [-2, 0, 2, "adaptive"])
def test_real_dsv4_weights_match_fixed_pipeline(vllm_runner, fixed_outputs, offset):
    additional = {
        "pp_send_buffer_size": 2,
        "dynamic_pp_config": {
            "enabled": True,
            "redundant_layers": 2,
            "initial_offset": -2 if offset == "adaptive" else offset,
            "adaptive": offset == "adaptive",
            "cooldown_steps": 1,
        },
    }
    actual, stats, base = _generate(vllm_runner, additional)
    expected, _, _ = fixed_outputs
    assert all(row["enabled"] and row["batches"] == 12 for row in stats)
    assert len({row["boundary"] for row in stats}) == 1
    assert len({row["boundary_version"] for row in stats}) == 1
    if offset != "adaptive":
        assert all(row["boundary"] == base + offset for row in stats)
    else:
        assert any(
            row["device_feedback"] is not None
            and row["device_feedback"]["completed_timing_samples"] > 0
            for row in stats
        )
    check_logprobs_close(
        outputs_0_lst=actual,
        outputs_1_lst=expected,
        name_0="dynamic PP",
        name_1="fixed PP",
        always_check_logprobs=True,
    )
