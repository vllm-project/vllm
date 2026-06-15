# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.worker.gpu.warmup import warmup_kernels

pytestmark = pytest.mark.cpu_test


class _StopWarmup(Exception):
    pass


def _make_model_runner(
    *,
    supports_prompt_logprobs: bool = True,
    supports_sampler_warmup: bool = True,
):
    return SimpleNamespace(
        num_speculative_steps=0,
        decode_query_len=1,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=16,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        is_pooling_model=False,
        supports_prompt_logprobs=supports_prompt_logprobs,
        supports_sampler_warmup=supports_sampler_warmup,
        kv_connector=SimpleNamespace(set_disabled=Mock()),
        is_last_pp_rank=False,
    )


@pytest.mark.parametrize(
    ("supports_prompt_logprobs", "expected_prompt_logprobs"),
    [(True, 1), (False, None)],
)
def test_warmup_respects_prompt_logprobs_capability(
    supports_prompt_logprobs,
    expected_prompt_logprobs,
):
    model_runner = _make_model_runner(supports_prompt_logprobs=supports_prompt_logprobs)
    scheduler_outputs = []

    def worker_execute_model(scheduler_output):
        scheduler_outputs.append(scheduler_output)

    def worker_sample_tokens(_grammar_output):
        raise _StopWarmup

    with pytest.raises(_StopWarmup):
        warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)

    sampling_params = scheduler_outputs[0].scheduled_new_reqs[0].sampling_params
    assert sampling_params is not None
    assert sampling_params.logprobs == 5
    assert sampling_params.prompt_logprobs == expected_prompt_logprobs


def test_warmup_can_skip_generic_sampler_warmup():
    model_runner = _make_model_runner(supports_sampler_warmup=False)
    scheduler_outputs = []
    worker_sample_tokens = Mock()

    def worker_execute_model(scheduler_output):
        scheduler_outputs.append(scheduler_output)

    warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens)

    assert len(scheduler_outputs) == 2
    prefill_output, cleanup_output = scheduler_outputs
    assert prefill_output.scheduled_new_reqs
    assert cleanup_output.finished_req_ids
    worker_sample_tokens.assert_not_called()
    model_runner.kv_connector.set_disabled.assert_any_call(True)
    model_runner.kv_connector.set_disabled.assert_any_call(False)
