# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.metrics.stats import IterationStats


def test_iteration_stats_repr():
    iteration_stats = IterationStats()
    iteration_stats.iteration_timestamp = 0
    expected_repr = (
        "IterationStats("
        "iteration_timestamp=0, "
        "num_generation_tokens=0, "
        "num_prompt_tokens=0, "
        "num_preempted_reqs=0, "
        "finished_requests=[], "
        "max_num_generation_tokens_iter=[], "
        "n_params_iter=[], "
        "time_to_first_tokens_iter=[], "
        "inter_token_latencies_iter=[], "
        "waiting_lora_adapters={}, "
        "running_lora_adapters={}, "
        "num_corrupted_reqs=0)"
    )
    assert repr(iteration_stats) == expected_repr
