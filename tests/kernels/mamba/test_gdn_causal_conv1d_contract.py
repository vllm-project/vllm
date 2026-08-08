# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as official_causal_conv1d_fn,
)
from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d import (
    causal_conv1d_fn,
)

from .causal_conv1d_contract import (
    CONTRACT_CASES,
    REQUIRED_FEATURE_FAMILIES,
    ContractCase,
    active_output,
    build_case,
    reference_output_and_state,
)


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        return 1e-2, 5e-2
    return 3e-4, 1e-3


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.name)
def test_official_operator_matches_contract_reference(case: ContractCase) -> None:
    if case.width == 5:
        pytest.xfail(
            "official prefill kernel has width-5 branches but fails its PyTorch "
            "reference; the replacement must implement the intended semantics"
        )
    inputs = build_case(case, torch.device("cuda"))
    expected_output, expected_states = reference_output_and_state(case, inputs)
    actual_states = inputs["conv_states"]
    assert isinstance(actual_states, torch.Tensor)
    actual_states = actual_states.clone()
    actual = official_causal_conv1d_fn(
        inputs["x"],
        inputs["weight"],
        inputs["bias"],
        actual_states,
        inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        has_initial_state=inputs["has_initial_state"],
        activation=cast(str | None, inputs["activation"]),
    )
    rtol, atol = tolerances(case.dtype)
    torch.testing.assert_close(
        active_output(case, actual), expected_output, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(actual_states, expected_states, rtol=rtol, atol=atol)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=lambda case: case.name)
def test_current_replacement_against_contract(case: ContractCase) -> None:
    if case.width == 5:
        pytest.xfail("width-5 inputs retain the existing official fallback")
    inputs = build_case(case, torch.device("cuda"))
    expected_output, expected_states = reference_output_and_state(case, inputs)
    actual_states = inputs["conv_states"]
    assert isinstance(actual_states, torch.Tensor)
    actual_states = actual_states.clone()
    actual = causal_conv1d_fn(
        inputs["x"],
        inputs["weight"],
        inputs["bias"],
        actual_states,
        inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        has_initial_state=inputs["has_initial_state"],
        activation=cast(str | None, inputs["activation"]),
    )
    rtol, atol = tolerances(case.dtype)
    torch.testing.assert_close(
        active_output(case, actual), expected_output, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(actual_states, expected_states, rtol=rtol, atol=atol)


def test_contract_tracks_all_required_feature_families() -> None:
    represented = {case.family for case in CONTRACT_CASES}
    implemented_later = {
        "continuous_batching",
        "pad_null",
        "apc_prefix_cache",
    }
    assert represented | implemented_later == REQUIRED_FEATURE_FAMILIES
