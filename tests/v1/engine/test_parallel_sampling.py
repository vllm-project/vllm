# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import SamplingParams
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.parallel_sampling import ParentRequest


def test_parent_request_to_output_stream() -> None:
    parent_request = ParentRequest("parent_id", SamplingParams(n=2))
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished
    assert ("parent_id", [output_0], False) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ("parent_id", [output_1], False) == parent_request.get_outputs(
        "child_id_1", output_1
    )
    assert ("parent_id", [output_0], False) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ("parent_id", [output_1], False) == parent_request.get_outputs(
        "child_id_1", output_1
    )

    # output_1 finished
    output_1.finish_reason = "ended"
    assert ("parent_id", [output_0], False) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ("parent_id", [output_1], False) == parent_request.get_outputs(
        "child_id_1", output_1
    )
    # Finished output_1 had already returned, DO NOT returned again
    assert ("parent_id", [output_0], False) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert parent_request.get_outputs("child_id_1", output_1) == (
        "parent_id",
        [],
        False,
    )

    # output_0 finished
    output_0.finish_reason = "ended"
    assert ("parent_id", [output_0], True) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert parent_request.get_outputs("child_id_1", output_1) == ("parent_id", [], True)
    # Finished output_0 had already returned, DO NOT returned again
    assert parent_request.get_outputs("child_id_0", output_0) == ("parent_id", [], True)
    assert parent_request.get_outputs("child_id_1", output_1) == ("parent_id", [], True)


def test_parent_request_to_output_final_only() -> None:
    parent_request = ParentRequest(
        "parent_id", SamplingParams(n=2, output_kind=RequestOutputKind.FINAL_ONLY)
    )
    parent_request.child_requests = {"child_id_0", "child_id_1"}
    output_0 = CompletionOutput(
        index=0, text="child 0", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    output_1 = CompletionOutput(
        index=1, text="child 1", token_ids=[], cumulative_logprob=None, logprobs=None
    )
    # Request not finished, return nothing
    assert parent_request.get_outputs("child_id_0", output_0) == (
        "parent_id",
        [],
        False,
    )
    assert parent_request.get_outputs("child_id_1", output_1) == (
        "parent_id",
        [],
        False,
    )
    # output_1 finished, but outputs won't be returned until all child requests finished
    output_1.finish_reason = "ended"
    assert parent_request.get_outputs("child_id_0", output_0) == (
        "parent_id",
        [],
        False,
    )
    assert parent_request.get_outputs("child_id_1", output_1) == (
        "parent_id",
        [],
        False,
    )
    # output_0 finished, as all child requests finished, the output would be returned
    output_0.finish_reason = "ended"
    assert ("parent_id", [output_0, output_1], True) == parent_request.get_outputs(
        "child_id_0", output_0
    )
    assert ("parent_id", [output_0, output_1], True) == parent_request.get_outputs(
        "child_id_1", output_1
    )
