# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
from vllm.v1.outputs import (
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    RoutedExpertsLists,
)


def _make_readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def test_detach_zero_copy_from_model_runner_output_copies_only_numpy_views():
    cu_num_generated_tokens = [0, 2]
    prompt_logprobs = LogprobsTensors.empty_cpu(1, 2)
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        logprobs=LogprobsLists(
            logprob_token_ids=_make_readonly(
                np.array([[1, 2], [3, 4]], dtype=np.int32)
            ),
            logprobs=_make_readonly(
                np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            ),
            sampled_token_ranks=_make_readonly(np.array([1, 2], dtype=np.int32)),
            cu_num_generated_tokens=cu_num_generated_tokens,
        ),
        prompt_logprobs_dict={"req-0": prompt_logprobs},
    )

    original_logprobs = output.logprobs
    assert original_logprobs is not None

    detach_zero_copy_from_model_runner_output(output)

    detached_logprobs = output.logprobs
    assert detached_logprobs is not None
    assert detached_logprobs is not original_logprobs
    assert (
        detached_logprobs.logprob_token_ids is not original_logprobs.logprob_token_ids
    )
    assert detached_logprobs.logprobs is not original_logprobs.logprobs
    assert (
        detached_logprobs.sampled_token_ranks
        is not original_logprobs.sampled_token_ranks
    )
    assert detached_logprobs.logprob_token_ids.flags.writeable
    assert detached_logprobs.logprobs.flags.writeable
    assert detached_logprobs.sampled_token_ranks.flags.writeable
    assert detached_logprobs.cu_num_generated_tokens is cu_num_generated_tokens
    assert output.prompt_logprobs_dict["req-0"] is prompt_logprobs


def test_detach_zero_copy_routed_experts_without_logprobs():
    output = ModelRunnerOutput(
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        routed_experts=RoutedExpertsLists(
            routing_data=_make_readonly(np.arange(12, dtype=np.int32).reshape(2, 3, 2)),
            slot_mapping=_make_readonly(np.array([7, 8], dtype=np.int64)),
        ),
    )
    original = output.routed_experts
    assert output.logprobs is None

    detach_zero_copy_from_model_runner_output(output)

    detached = output.routed_experts
    assert detached is not None
    assert original is not None
    assert detached is not original
    assert detached.routing_data is not original.routing_data
    assert detached.slot_mapping is not original.slot_mapping
    assert detached.routing_data.flags.writeable
    assert detached.slot_mapping.flags.writeable
    np.testing.assert_array_equal(detached.routing_data, original.routing_data)
    np.testing.assert_array_equal(detached.slot_mapping, original.slot_mapping)
