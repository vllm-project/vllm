# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.distributed.aux_output_connector.connector import (
    AuxRequestOutput,
    AuxStepOutput,
)
from vllm.v1.executor.ray_utils import detach_zero_copy_from_model_runner_output
from vllm.v1.outputs import (
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
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


def test_ray_deserialization_detaches_aux_output():
    """Both packed rows and historical rows must stop referencing Ray storage."""
    ray = pytest.importorskip("ray")
    rows = _make_readonly(np.ones((4096, 52, 6), dtype=np.uint8))
    output = ModelRunnerOutput(
        req_ids=["request"],
        req_id_to_index={"request": 0},
        aux_output_connector_output=AuxStepOutput(
            rows, {"request": (0, 0, 1)}, {"prefix": AuxRequestOutput(0, rows)}
        ),
    )
    with ray.init(
        address="local",
        num_cpus=1,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=150 * 1024 * 1024,
    ):
        decoded = ray.get(ray.put(output)).aux_output_connector_output
        assert decoded.rows.flags.owndata and decoded.rows.flags.writeable
        assert decoded.materialized["prefix"].rows.flags.owndata
        np.testing.assert_array_equal(decoded["request"].rows, rows[:1])
        np.testing.assert_array_equal(decoded["prefix"].rows, rows)
