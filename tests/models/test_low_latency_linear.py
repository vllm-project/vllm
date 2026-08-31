# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shape-selected fused-A GEMM on unquantized BF16 projections."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.common.ops import low_latency_linear
from vllm.models.common.ops.low_latency_linear import (
    FusedALinearMethod,
    FusedATable,
    install_fused_a_linear,
    run_fused_a_gemm,
)
from vllm.models.deepseek_v32.nvidia.glm52_low_latency_gemm import GLM52_FUSED_A_TABLE
from vllm.models.kimi_k3.nvidia import low_latency_gemm as k3_gemm

ALL_TABLES = [
    ("glm52", GLM52_FUSED_A_TABLE),
    ("kimi_k3_sm103", k3_gemm.KIMI_K3_PROJECTIONS),
    ("kimi_k3_sm100", k3_gemm.KIMI_K3_PROJECTIONS_SM100),
    ("kimi_k3_sm90", k3_gemm.KIMI_K3_PROJECTIONS_SM90),
]


@pytest.mark.parametrize("name,table", ALL_TABLES, ids=[n for n, _ in ALL_TABLES])
def test_tables_only_select_supported_token_counts(
    name: str, table: FusedATable
) -> None:
    """The fused-A GEMM is a decode kernel; a table entry above M=16 would
    silently route a prefill batch onto it."""
    for (n, k), tokens in table.items():
        assert tokens, f"{name} {(n, k)} has an empty token set"
        assert min(tokens) >= 1 and max(tokens) <= 16, f"{name} {(n, k)}"


def test_kimi_k3_selection_follows_device_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for capability, table in (
        ((10, 3), k3_gemm.KIMI_K3_PROJECTIONS),
        ((10, 0), k3_gemm.KIMI_K3_PROJECTIONS_SM100),
        ((9, 0), k3_gemm.KIMI_K3_PROJECTIONS_SM90),
        ((8, 0), None),
    ):
        monkeypatch.setattr(
            k3_gemm.current_platform,
            "is_device_capability",
            lambda cc, cap=capability: cc == cap,
        )
        assert k3_gemm._low_latency_table() is table
        expected = frozenset() if table is None else table[(2304, 1536)]
        assert k3_gemm.select_kimi_k3_tokens(2304, 1536) == expected


class _FakeLinear(nn.Module):
    """Stands in for a LinearBase without needing a distributed environment."""

    def __init__(self, quant_method: object, n: int, k: int, dtype=torch.bfloat16):
        super().__init__()
        self.quant_method = quant_method
        self.weight = torch.empty(n, k, dtype=dtype)


def test_install_only_claims_listed_unquantized_bf16_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed_n, listed_k = next(iter(GLM52_FUSED_A_TABLE))
    quantized_method = object()
    root = nn.Module()
    root.listed = _FakeLinear(UnquantizedLinearMethod(), listed_n, listed_k)
    root.unlisted = _FakeLinear(UnquantizedLinearMethod(), listed_n, listed_k + 128)
    root.quantized = _FakeLinear(quantized_method, listed_n, listed_k)
    root.wrong_dtype = _FakeLinear(
        UnquantizedLinearMethod(), listed_n, listed_k, dtype=torch.float16
    )
    monkeypatch.setattr(low_latency_linear, "LinearBase", _FakeLinear)

    install_fused_a_linear(root, torch.bfloat16, GLM52_FUSED_A_TABLE)

    assert isinstance(root.listed.quant_method, FusedALinearMethod)
    assert type(root.unlisted.quant_method) is UnquantizedLinearMethod
    assert type(root.wrong_dtype.quant_method) is UnquantizedLinearMethod
    assert root.quantized.quant_method is quantized_method


def test_install_is_a_noop_for_non_bf16_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed_n, listed_k = next(iter(GLM52_FUSED_A_TABLE))
    root = nn.Module()
    root.listed = _FakeLinear(UnquantizedLinearMethod(), listed_n, listed_k)
    monkeypatch.setattr(low_latency_linear, "LinearBase", _FakeLinear)

    install_fused_a_linear(root, torch.float16, GLM52_FUSED_A_TABLE)

    assert type(root.listed.quant_method) is UnquantizedLinearMethod


def test_unselected_token_count_falls_back_to_the_base_method() -> None:
    # CPU tensors also fail the runtime check, so this exercises both guards.
    method = FusedALinearMethod(frozenset({1}))
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)

    output = method.apply(SimpleNamespace(weight=weight), x)

    torch.testing.assert_close(output, torch.nn.functional.linear(x, weight))


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "dsv3_fused_a_gemm"),
    reason="dsv3_fused_a_gemm was not built",
)
def test_nonpacked_activation_falls_back() -> None:
    """A size-1 leading dim reads as contiguous even when its stride is not
    packed, which the fused-A kernel cannot consume."""
    n, k = 1536, 128
    storage = torch.randn(1, k + 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    x = storage[:, :k]

    assert x.is_contiguous() and x.stride() == (k + 16, 1)
    assert run_fused_a_gemm(frozenset({1}), x, weight) is None


def _fused_a_cases() -> list[tuple[int, int, int]]:
    """Table entries for the device under test, probing the tile branches."""
    if not torch.cuda.is_available():
        return []
    table = k3_gemm._low_latency_table()
    if table is None:
        return []
    return sorted(
        (n, k, num_tokens)
        for (n, k), tokens in table.items()
        # Extremes plus both sides of the kernel's num_tokens <= 8 tile branch.
        for num_tokens in {min(tokens), max(tokens)} | ({8, 9} & set(tokens))
    )


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "dsv3_fused_a_gemm"),
    reason="dsv3_fused_a_gemm was not built",
)
@pytest.mark.parametrize("n,k,num_tokens", _fused_a_cases())
def test_selected_shapes_match_the_reference(n: int, k: int, num_tokens: int) -> None:
    torch.manual_seed(42)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    output = run_fused_a_gemm(frozenset({num_tokens}), x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "dsv3_fused_a_gemm"),
    reason="dsv3_fused_a_gemm was not built",
)
@pytest.mark.parametrize("num_tokens", [1, 8, 9, 16])
def test_cuda_graph_capture_across_tile_branches(num_tokens: int) -> None:
    n, k = 1536, 128
    tokens = frozenset({num_tokens})
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    run_fused_a_gemm(tokens, x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run_fused_a_gemm(tokens, x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999
