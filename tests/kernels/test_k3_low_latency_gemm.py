# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Kimi-K3 / GLM-5.2 SM103 low-latency BF16 GEMM selectors."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re
import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import glm52_low_latency_gemm as glm52_gemm
from vllm.models.kimi_k3.nvidia import low_latency_gemm as k3_gemm
from vllm.models.kimi_k3.nvidia.low_latency_gemm import KIMI_K3_PROJECTIONS

# Keyed by local (N, K): the token counts routed to dsv3_fused_a_gemm. 1536x7168
# is the unified shared_gate_up_proj/mla_g_proj entry (dsv3 M1..16). Every other
# shape and token count falls through to the default unquantized GEMM.
EXPECTED_SELECTIONS = {
    (1536, 128): set(range(1, 17)),
    (3072, 128): set(range(1, 17)),
    (1536, 7168): set(range(1, 17)),
    (2112, 7168): set(range(1, 17)),
    (2304, 1536): set(range(1, 17)),
    (4608, 1536): set(range(1, 17)),
    (3584, 7168): set(range(2, 9)),
    (7168, 768): set(range(1, 17)),
    # TP16.
    (3216, 7168): set(range(9, 16)),
    (768, 7168): set(range(5, 17)),
    (1152, 1536): set(range(2, 17)),
    (768, 128): set(range(1, 17)),
    (7168, 384): set(range(1, 9)),
    (4224, 7168): set(range(4, 9)),
}


def test_table_is_keyed_by_shape() -> None:
    for (n, k), spec in KIMI_K3_PROJECTIONS.items():
        assert (spec.n, spec.k) == (n, k)


def test_table_matches_measured_selections() -> None:
    actual = {
        (spec.n, spec.k): set(spec.dsv3_tokens) for spec in KIMI_K3_PROJECTIONS.values()
    }
    assert actual == EXPECTED_SELECTIONS


def test_every_dsv3_routed_shape_is_instantiated() -> None:
    """dsv3_fused_a_gemm specializes on (K, N); an unlisted shape raises.

    The table routes by shape while the kernel is built per shape, so a missing
    instantiation only shows up at the token counts that route to dsv3. Checking
    it here needs no GPU, which is the point -- a GPU-only check is exactly what
    let (3216, 7168) ship without its DISPATCH_DSV3_SHAPE(7168, 3216).
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "libtorch_stable"
        / "dsv3_fused_a_gemm.cu"
    ).read_text(encoding="utf-8")
    # Benchmark-only shapes live behind VLLM_K3_BENCH_SHAPES and are not built
    # by default, so they must not count as available.
    production_macros = source.split("#ifdef VLLM_K3_BENCH_SHAPES")[0]
    explicit = source.split("#undef DISPATCH_DSV3_SHAPE")[1].split(
        "#ifdef VLLM_K3_BENCH_SHAPES"
    )[0]
    compiled = {
        (int(hd_in), int(hd_out))
        for hd_in, hd_out in re.findall(
            r"DISPATCH_DSV3_SHAPE\((\d+),\s*(\d+)\)", production_macros
        )
    } | {
        (int(hd_in), int(hd_out))
        for hd_in, hd_out in re.findall(
            r"hd_in == (\d+) && hd_out == (\d+)",
            production_macros + explicit,
        )
    }
    assert compiled, "failed to parse the dispatch list"

    specs = [
        *KIMI_K3_PROJECTIONS.values(),
        glm52_gemm.GLM52_QKV_A_PROJECTION,
        glm52_gemm.GLM52_Q_B_PROJECTION,
    ]
    missing = sorted(
        (spec.n, spec.k)
        for spec in specs
        if spec.dsv3_tokens and (spec.k, spec.n) not in compiled
    )
    assert not missing, (
        f"routed to dsv3 with no instantiation: {missing}; add "
        "DISPATCH_DSV3_SHAPE(K, N) for each"
    )


def test_packed_row_major_rejects_single_row_slice() -> None:
    packed = torch.empty(1, 128)
    sliced = torch.empty(1, 144)[:, :128]

    assert packed.is_contiguous()
    assert sliced.is_contiguous()
    assert k3_gemm._is_packed_row_major(packed)
    assert not k3_gemm._is_packed_row_major(sliced)


def test_glm52_projection_plans_are_separate() -> None:
    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION

    qkv_a_plan = qkv_a.build_plan()
    q_b_plan = q_b.build_plan()

    assert (qkv_a.n, qkv_a.k, set(qkv_a_plan)) == (2624, 6144, set(range(3, 17)))
    assert (q_b.n, q_b.k, set(q_b_plan)) == (2048, 2048, set(range(3, 17)))
    assert all(backend == "dsv3_fused_a" for backend in qkv_a_plan.values())
    assert all(backend == "dsv3_fused_a" for backend in q_b_plan.values())


def test_glm52_layout_rejects_nonpacked_single_row_view() -> None:
    single_row = torch.empty(1, 144)[:, :128]
    multiple_rows = torch.empty(2, 144)[:, :128]

    assert single_row.stride() == multiple_rows.stride() == (144, 1)
    assert not glm52_gemm._is_supported_row_major(single_row)
    assert not glm52_gemm._is_supported_row_major(multiple_rows)


def test_glm52_installer_maps_only_selected_unquantized_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearBase(nn.Module):
        def __init__(
            self,
            n: int,
            k: int,
            quant_method: object,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(
                    n,
                    k,
                    dtype=torch.bfloat16,
                    device="meta",
                )
            )
            self.quant_method = quant_method

    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION
    root = nn.Module()
    root.attn = nn.Module()
    root.attn.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.attn.q_b_proj = FakeLinearBase(
        q_b.n, q_b.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.same_shape_other_name = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.quantized = nn.Module()
    quantized_method = object()
    root.quantized.q_b_proj = FakeLinearBase(q_b.n, q_b.k, quantized_method)
    root.wrong_shape = nn.Module()
    root.wrong_shape.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n + 1, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    monkeypatch.setattr(glm52_gemm, "LinearBase", FakeLinearBase)
    monkeypatch.setattr(glm52_gemm, "_is_sm103", lambda: True)

    glm52_gemm.enable_glm52_low_latency_gemm(
        root,
        torch.bfloat16,
    )

    assert isinstance(
        root.attn.fused_qkv_a_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert isinstance(
        root.attn.q_b_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.attn.fused_qkv_a_proj.quant_method._plan == qkv_a.build_plan()
    assert root.attn.q_b_proj.quant_method._plan == q_b.build_plan()
    assert isinstance(
        root.same_shape_other_name.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.quantized.q_b_proj.quant_method is quantized_method
    assert (
        type(root.wrong_shape.fused_qkv_a_proj.quant_method)
        is glm52_gemm.UnquantizedLinearMethod
    )


@pytest.mark.parametrize("key", EXPECTED_SELECTIONS)
def test_sm103_selector_table(key: tuple[int, int]) -> None:
    n, k = key
    dsv3_tokens = EXPECTED_SELECTIONS[key]
    for num_tokens in range(1, 17):
        backend = k3_gemm.select_kimi_k3_backend(num_tokens, n, k)
        assert backend == ("dsv3_fused_a" if num_tokens in dsv3_tokens else None)


@pytest.mark.parametrize("key", EXPECTED_SELECTIONS)
def test_selector_requires_supported_shape_and_tokens(key: tuple[int, int]) -> None:
    n, k = key
    assert k3_gemm.select_kimi_k3_backend(0, n, k) is None
    assert k3_gemm.select_kimi_k3_backend(17, n, k) is None
    assert k3_gemm.select_kimi_k3_backend(1, n + 1, k) is None
    assert k3_gemm.select_kimi_k3_backend(1, n, k + 1) is None


def test_unlisted_shape_and_unselected_tokens_fall_back() -> None:
    # Shape absent from the table.
    assert k3_gemm.select_kimi_k3_backend(1, 1000, 1000) is None
    # routed_expert_down_proj is dsv3 from M2 on; M1 falls back.
    assert k3_gemm.select_kimi_k3_backend(1, 3584, 7168) is None


def test_build_plan_matches_selector() -> None:
    for spec in KIMI_K3_PROJECTIONS.values():
        plan = k3_gemm._build_plan(spec)
        for num_tokens in range(1, 17):
            backend = k3_gemm.select_kimi_k3_backend(num_tokens, spec.n, spec.k)
            if backend is None:
                assert num_tokens not in plan
            else:
                assert plan[num_tokens] == backend


def test_installation_is_shape_specific_and_unquantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinear(nn.Module):
        def __init__(self, quant_method: object, n: int, k: int) -> None:
            super().__init__()
            self.quant_method = quant_method
            self.weight = torch.empty(n, k)

    root = nn.Module()
    root.selected = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 2304, 1536)
    # quantized: must be left untouched.
    quantized_method = object()
    root.quantized = FakeLinear(quantized_method, 2112, 7168)
    # shape absent from the table: must be left untouched.
    root.unlisted = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 1234, 5678)

    monkeypatch.setattr(k3_gemm, "LinearBase", FakeLinear)
    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: True)

    k3_gemm.enable_kimi_k3_low_latency_gemm(root, torch.bfloat16)

    assert isinstance(root.selected.quant_method, k3_gemm.KimiK3LowLatencyLinearMethod)
    assert root.selected.quant_method._plan == k3_gemm._build_plan(
        KIMI_K3_PROJECTIONS[(2304, 1536)]
    )
    assert root.quantized.quant_method is quantized_method
    assert type(root.unlisted.quant_method) is k3_gemm.UnquantizedLinearMethod


@pytest.mark.parametrize(
    "dtype,platform_enabled",
    [(torch.float16, True), (torch.bfloat16, False)],
)
def test_installation_requires_bf16_sm103(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    platform_enabled: bool,
) -> None:
    class FakeLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = k3_gemm.UnquantizedLinearMethod()
            self.weight = torch.empty(2304, 1536)

    root = nn.Module()
    root.projection = FakeLinear()
    monkeypatch.setattr(k3_gemm, "LinearBase", FakeLinear)
    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: platform_enabled)

    k3_gemm.enable_kimi_k3_low_latency_gemm(root, dtype)

    assert type(root.projection.quant_method) is k3_gemm.UnquantizedLinearMethod


def _require_sm103_and_dsv3() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Kimi-K3 production selection requires SM103")
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        pytest.skip("dsv3_fused_a_gemm was not built")


def _dsv3_probe_tokens(tokens: frozenset[int]) -> set[int]:
    """Extremes, plus both sides of the kernel's num_tokens<=8 tile_n branch."""
    if not tokens:
        return set()
    return {min(tokens), max(tokens)} | ({8, 9} & set(tokens))


# Derived from the table rather than hand-listed, so a shape routed to dsv3
# cannot be added without being exercised here.
DSV3_CASES = sorted(
    (num_tokens, spec.n, spec.k)
    for spec in KIMI_K3_PROJECTIONS.values()
    for num_tokens in _dsv3_probe_tokens(spec.dsv3_tokens)
)

GLM_DSV3_CASES = [
    (num_tokens, spec)
    for spec in (
        glm52_gemm.GLM52_QKV_A_PROJECTION,
        glm52_gemm.GLM52_Q_B_PROJECTION,
    )
    for num_tokens in sorted(_dsv3_probe_tokens(spec.dsv3_tokens))
]


@pytest.mark.parametrize("num_tokens,n,k", DSV3_CASES)
def test_dsv3_selected_shapes(num_tokens: int, n: int, k: int) -> None:
    _require_sm103_and_dsv3()
    spec = KIMI_K3_PROJECTIONS[(n, k)]
    assert num_tokens in spec.dsv3_tokens
    torch.manual_seed(42)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    output = k3_gemm.try_low_latency_gemm(x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("num_tokens,spec", GLM_DSV3_CASES)
def test_glm_dsv3_selected_shapes(
    num_tokens: int,
    spec: glm52_gemm.GLM52ProjectionSpec,
) -> None:
    _require_sm103_and_dsv3()
    torch.manual_seed(42)
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")

    output = glm52_gemm.run_glm52_plan(spec.build_plan(), x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def test_nonpacked_single_token_dsv3_falls_back() -> None:
    _require_sm103_and_dsv3()
    n, k = 1536, 128
    storage = torch.randn(1, k + 16, dtype=torch.bfloat16, device="cuda")
    x = storage[:, :k]
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    spec = KIMI_K3_PROJECTIONS[(n, k)]
    method = k3_gemm.KimiK3LowLatencyLinearMethod(k3_gemm._build_plan(spec))

    assert x.is_contiguous()
    assert x.stride() == (k + 16, 1)
    assert not k3_gemm._runtime_ok(x, weight)  # strict guard rejects the view
    output = method.apply(SimpleNamespace(weight=weight), x)

    reference = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(output, reference)


@pytest.mark.parametrize("num_tokens", [1, 8, 9, 16])
def test_dsv3_cuda_graph_capture_tile_branches(num_tokens: int) -> None:
    """Capture DSV3 across the num_tokens<=8 vs >8 tile_n branch."""
    _require_sm103_and_dsv3()
    spec = KIMI_K3_PROJECTIONS[(1536, 128)]
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    k3_gemm.try_low_latency_gemm(x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = k3_gemm.try_low_latency_gemm(x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def test_fallback_preserves_default_method(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback = torch.empty(2, 8)
    monkeypatch.setattr(
        k3_gemm.UnquantizedLinearMethod,
        "apply",
        lambda *args: fallback,
    )
    # 1-D input fails the runtime check, forcing the base-method fallback.
    method = k3_gemm.KimiK3LowLatencyLinearMethod({})

    output = method.apply(
        SimpleNamespace(weight=torch.empty(0)),
        torch.empty(0),
    )

    assert output is fallback
