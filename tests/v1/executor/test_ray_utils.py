# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.executor.ray_utils import (
    RayWorkerWrapper,
    detach_zero_copy_from_model_runner_output,
)
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, ModelRunnerOutput


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


def test_adjust_rank_keeps_global_rank_in_sync():
    wrapper = RayWorkerWrapper(rpc_rank=0)
    assert wrapper.rpc_rank == 0
    assert wrapper.global_rank == 0

    # Re-rank from 0 -> 2
    # (emulating RayDistributedExecutor reordering workers by node placement).
    wrapper.adjust_rank({0: 2})
    assert wrapper.rpc_rank == 2
    assert wrapper.global_rank == 2

    # When the worker's current `rpc_rank` (of 2) is not a key in the `rank_mapping`
    # argument below ({5: 9}), `adjust_rank` must leave both ranks untouched
    wrapper.adjust_rank({5: 9})
    assert wrapper.rpc_rank == 2
    assert wrapper.global_rank == 2
