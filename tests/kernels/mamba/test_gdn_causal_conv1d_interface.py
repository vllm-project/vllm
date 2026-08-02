# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops import gdn_causal_conv1d
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn as official_causal_conv1d_fn,
)
from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d import (
    causal_conv1d_fn,
)

from .causal_conv1d_contract import (
    CONTRACT_CASES,
    ContractCase,
    build_case,
    reference_output_and_state,
)


def test_replacement_interface_matches_official_parameters() -> None:
    official = inspect.signature(official_causal_conv1d_fn)
    replacement = inspect.signature(causal_conv1d_fn)
    assert tuple(replacement.parameters) == tuple(official.parameters)
    for name, parameter in official.parameters.items():
        assert replacement.parameters[name].default == parameter.default


def test_short_prefill_uses_official_fallback() -> None:
    case = next(case for case in CONTRACT_CASES if case.name == "activation_none")
    inputs = build_case(case, torch.device("cuda"))
    with patch.object(
        gdn_causal_conv1d,
        "official_causal_conv1d_fn",
        wraps=official_causal_conv1d_fn,
    ) as fallback:
        causal_conv1d_fn(
            inputs["x"],
            inputs["weight"],
            inputs["bias"],
            inputs["conv_states"],
            inputs["query_start_loc"],
            cache_indices=inputs["cache_indices"],
            has_initial_state=inputs["has_initial_state"],
            activation=cast(str | None, inputs["activation"]),
        )
    fallback.assert_called_once()


def test_replacement_interface_runs_current_fast_variant() -> None:
    case = next(case for case in CONTRACT_CASES if case.name == "main_bf16_w4")
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

    torch.testing.assert_close(actual, expected_output, rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)


def test_replacement_preserves_input_dtype_with_bf16_cache() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dim = 128
    tokens = 31
    x = torch.randn(tokens, dim, device=device, dtype=torch.float32).transpose(0, 1)
    weight = torch.randn(dim, 4, device=device, dtype=torch.bfloat16)
    states = torch.randn(2, 3, dim, device=device, dtype=torch.bfloat16).transpose(1, 2)
    indices = torch.tensor([1], device=device, dtype=torch.int32)
    has_initial = torch.tensor([True], device=device)
    query_start = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    x_bf16 = x.to(torch.bfloat16)
    full_input = torch.cat((states[1], x_bf16), dim=1)
    expected = (
        F.silu(
            F.conv1d(
                full_input.unsqueeze(0).float(),
                weight.unsqueeze(1).float(),
                groups=dim,
            )
        )
        .squeeze(0)
        .to(torch.bfloat16)
        .float()
    )
    expected_states = states.clone()
    expected_states[1] = full_input[:, -3:]
    actual_states = states.clone()
    actual = causal_conv1d_fn(
        x,
        weight,
        None,
        actual_states,
        query_start,
        cache_indices=indices,
        has_initial_state=has_initial,
    )

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)


def test_replacement_validate_data_rejects_unsupported_layout() -> None:
    base = torch.empty(64, 64, device="cuda", dtype=torch.bfloat16)
    x = base[::2, ::2]
    weight = torch.empty(32, 4, device="cuda", dtype=torch.bfloat16)
    states = torch.empty(1, 32, 3, device="cuda", dtype=torch.bfloat16)
    query_start = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError):
        causal_conv1d_fn(
            x,
            weight,
            None,
            states,
            query_start,
            validate_data=True,
        )


def test_long_cpu_input_uses_official_fallback(monkeypatch) -> None:
    dim, tokens = 128, 1024
    x = torch.empty(tokens, dim, dtype=torch.bfloat16).T
    weight = torch.empty(dim, 4, dtype=torch.bfloat16)
    states = torch.empty(2, dim, 3, dtype=torch.bfloat16)
    query_start = torch.tensor([0, tokens], dtype=torch.int32)
    indices = torch.tensor([1], dtype=torch.int32)
    has_initial = torch.tensor([False])
    expected = torch.empty_like(x)

    monkeypatch.setattr(gdn_causal_conv1d, "_IS_SM103", True)
    monkeypatch.setattr(
        gdn_causal_conv1d,
        "generic_causal_conv1d",
        lambda *args, **kwargs: pytest.fail("CPU input reached the Triton path"),
    )
    with patch.object(
        gdn_causal_conv1d,
        "official_causal_conv1d_fn",
        return_value=expected,
    ) as fallback:
        actual = causal_conv1d_fn(
            x,
            weight,
            None,
            states,
            query_start,
            cache_indices=indices,
            has_initial_state=has_initial,
        )

    assert actual is expected
    fallback.assert_called_once()


def test_replacement_interface_runs_generic_variant() -> None:
    case = ContractCase(
        "long_activation_none",
        (1024,),
        activation=None,
        initial_mask=(True,),
    )
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
    torch.testing.assert_close(actual, expected_output, rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(actual_states, expected_states, rtol=0, atol=0)
