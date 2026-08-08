# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d import (
    fast_causal_conv1d,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="SM103 causal-conv specialization requires compute capability 10.3",
)


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("has_initial", [False, True])
@pytest.mark.parametrize("tokens,dim", [(256, 4096), (8192, 12288)])
def test_fast_causal_conv_matches_stock_and_state(
    has_bias: bool, has_initial: bool, tokens: int, dim: int
) -> None:
    torch.manual_seed(0)
    projected = (
        torch.randn(tokens, dim + 8192, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    x = projected[:, :dim].transpose(0, 1)
    weight = torch.randn(dim, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = (
        torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.1
        if has_bias
        else None
    )
    state = (
        torch.randn(2, 3, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    ).transpose(1, 2)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([has_initial], device="cuda")

    prefix = state[1] if has_initial else torch.zeros_like(state[1])
    sequence = torch.cat((prefix, x), dim=1).unsqueeze(0).float()
    expected = (
        F.silu(
            F.conv1d(
                sequence,
                weight.unsqueeze(1).float(),
                bias.float() if bias is not None else None,
                groups=dim,
            )
        )
        .squeeze(0)
        .to(torch.bfloat16)
    )
    expected_state = state.clone()
    expected_state[1] = x[:, -3:]
    actual_state = state.clone()
    actual = fast_causal_conv1d(
        x, weight, bias, actual_state, indices, has_initial_state
    )

    assert actual is not None
    diff = actual.float() - expected.float()
    rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(expected.float())
    assert rel_l2.item() <= 1e-2
    torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


def test_fast_causal_conv_preserves_null_state_slot() -> None:
    x = torch.empty(256, 4096, device="cuda", dtype=torch.bfloat16).T
    weight = torch.empty(4096, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(1, 4096, 3, device="cuda", dtype=torch.bfloat16)
    expected_state = state.clone()
    indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False], device="cuda")

    fast_causal_conv1d(x, weight, None, state, indices, has_initial_state)
    torch.accelerator.synchronize()

    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def test_fast_causal_conv_rejects_misaligned_token_stride(monkeypatch) -> None:
    dim, tokens = 4096, 256
    backing = torch.empty(tokens, dim + 1, device="cuda", dtype=torch.bfloat16)
    x = backing[:, :dim].T
    weight = torch.empty(dim, 4, device="cuda", dtype=torch.bfloat16)
    state = torch.empty(2, dim, 3, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False], device="cuda")
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        return torch.empty_like(x)

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.ops.gdn_causal_conv1d.ops."
        "gdn_causal_conv1d_sm103",
        fail_if_called,
    )

    assert x.stride(1) % 4 != 0
    assert (
        fast_causal_conv1d(x, weight, None, state, indices, has_initial_state) is None
    )
    assert not called


def test_fast_causal_conv_rejects_cpu_input(monkeypatch) -> None:
    from vllm.model_executor.layers.mamba.ops import gdn_causal_conv1d

    monkeypatch.setattr(gdn_causal_conv1d, "_IS_SM103", True)
    monkeypatch.setattr(gdn_causal_conv1d, "_HAS_SM103_KERNEL", True)
    monkeypatch.setattr(
        gdn_causal_conv1d.ops,
        "gdn_causal_conv1d_sm103",
        lambda *args, **kwargs: torch.empty_like(args[0]),
    )
    dim, tokens = 4096, 256
    x = torch.empty(tokens, dim, dtype=torch.bfloat16).T
    weight = torch.empty(dim, 4, dtype=torch.bfloat16)
    state = torch.empty(2, dim, 3, dtype=torch.bfloat16)
    indices = torch.tensor([1], dtype=torch.int32)
    has_initial_state = torch.tensor([False])

    assert (
        fast_causal_conv1d(x, weight, None, state, indices, has_initial_state) is None
    )


def test_fast_causal_conv_rejects_token_contiguous_input() -> None:
    x = torch.empty(4096, 256, device="cuda", dtype=torch.bfloat16).contiguous()
    weight = torch.empty(4096, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.empty(4096, device="cuda", dtype=torch.bfloat16)
    state = torch.empty(2, 4096, 3, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False], device="cuda")

    assert (
        fast_causal_conv1d(x, weight, bias, state, indices, has_initial_state) is None
    )
