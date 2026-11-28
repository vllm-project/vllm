# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import utils as layer_utils
from vllm.utils.flashinfer import is_flashinfer_cutedsl_bf16_gemm_supported


@pytest.fixture(scope="module", autouse=True)
def require_flashinfer_bf16_cutedsl() -> None:
    if not torch.accelerator.is_available():
        pytest.skip("CUDA is required")
    if not is_flashinfer_cutedsl_bf16_gemm_supported():
        pytest.skip("FlashInfer BF16 cute-dsl backend is unavailable")


@pytest.mark.parametrize("n,k", [(1024, 1024), (2048, 3072)])
@pytest.mark.parametrize(
    "m,use_bias,pdl",
    [
        (1, False, False),
        (8, True, False),
        (16, False, True),
        (32, True, True),
        (64, True, True),
        (512, False, False),
    ],
)
def test_flashinfer_bf16_cutedsl_correctness(
    m: int,
    n: int,
    k: int,
    use_bias: bool,
    pdl: bool,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    bias = (
        torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1 if use_bias else None
    )

    actual = layer_utils.cuda_flashinfer_bf16_gemm_impl(
        x, weight, bias, pdl, "flashinfer_cutedsl"
    )
    expected = F.linear(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)


# M is the row count of the flattened GEMM input: up to 32 runs on FlashInfer
# CuTe-DSL, every larger M on torch.
_FLASHINFER_M = (1, 2, 4, 8, 16, 31, 32)
_TORCH_M = (33, 64, 128, 512, 2048, 8192)


@pytest.fixture
def flashinfer_mm_calls(monkeypatch) -> list[tuple[int, str, bool]]:
    """Record the M, backend and PDL flag of every FlashInfer BF16 mm call."""
    calls: list[tuple[int, str, bool]] = []
    real_mm = layer_utils.flashinfer_bf16_mm

    def spy(a, b, bias, pdl, backend):
        calls.append((a.shape[0], backend, pdl))
        return real_mm(a, b, bias, pdl, backend)

    monkeypatch.setattr(layer_utils, "flashinfer_bf16_mm", spy)
    return calls


def _cutedsl_gemm():
    gemm = layer_utils.dispatch_unquantized_gemm("flashinfer_cutedsl")
    assert gemm is not layer_utils.cuda_unquantized_gemm
    return gemm


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1


@pytest.mark.parametrize("m", _FLASHINFER_M + _TORCH_M)
@pytest.mark.parametrize("use_bias", [False, True])
def test_flashinfer_cutedsl_bf16_dispatch_by_m(
    m: int, use_bias: bool, flashinfer_mm_calls
) -> None:
    from vllm.platforms import current_platform

    torch.manual_seed(0)
    n, k = 2560, 4096  # no skinny GEMM plan
    x, weight = _bf16(m, k), _bf16(n, k)
    bias = _bf16(n) if use_bias else None

    actual = _cutedsl_gemm()(None, x, weight, bias)

    expected = F.linear(x, weight, bias)
    if m <= 32:
        pdl = current_platform.is_arch_support_pdl()
        assert flashinfer_mm_calls == [(m, "cute-dsl", pdl)]
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-1)
    else:
        assert flashinfer_mm_calls == []
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32])
def test_flashinfer_cutedsl_bf16_prefers_a_skinny_gemm_plan(
    m: int, flashinfer_mm_calls
) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm_plans import (
        get_skinny_gemm_plan,
    )

    n, k = 2048, 4096
    plan = get_skinny_gemm_plan(n, k)
    if plan is None:
        pytest.skip("No skinny GEMM plan for this shape on this GPU")
    x, weight = _bf16(m, k), _bf16(n, k)

    actual = _cutedsl_gemm()(None, x, weight)

    assert [call[0] for call in flashinfer_mm_calls] == ([] if m in plan else [m])
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("leading,m", [((2, 16), 32), ((3, 11), 33), ((1, 1), 1)])
def test_flashinfer_cutedsl_bf16_m_counts_flattened_rows(
    leading: tuple[int, int], m: int, flashinfer_mm_calls
) -> None:
    n, k = 1024, 2048
    x, weight = _bf16(*leading, k), _bf16(n, k)

    actual = _cutedsl_gemm()(None, x, weight)

    assert actual.shape == (*leading, n)
    assert [call[0] for call in flashinfer_mm_calls] == ([m] if m <= 32 else [])
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=2e-2, atol=2e-1)


def _misaligned(m: int, k: int) -> torch.Tensor:
    # A contiguous view 16 bytes into its storage.
    return _bf16(m * k + 8)[8:].view(m, k)


@pytest.mark.parametrize(
    "case",
    [
        "k_not_multiple_of_128",
        "x_not_contiguous",
        "x_misaligned",
        "weight_not_contiguous",
        "fp16",
        "bias_not_contiguous",
    ],
)
def test_flashinfer_cutedsl_bf16_unsupported_inputs_use_torch(
    case: str, flashinfer_mm_calls
) -> None:
    m, n, k = 8, 1024, 2048
    x, weight, bias = _bf16(m, k), _bf16(n, k), None
    if case == "k_not_multiple_of_128":
        x, weight = _bf16(m, 1000), _bf16(n, 1000)
    elif case == "x_not_contiguous":
        x = _bf16(k, m).t()
    elif case == "x_misaligned":
        x = _misaligned(m, k)
    elif case == "weight_not_contiguous":
        weight = _bf16(k, n).t()
    elif case == "fp16":
        x, weight = x.half(), weight.half()
    elif case == "bias_not_contiguous":
        bias = _bf16(2 * n)[::2]

    actual = _cutedsl_gemm()(None, x, weight, bias)

    assert flashinfer_mm_calls == []
    torch.testing.assert_close(actual, F.linear(x, weight, bias), rtol=0, atol=0)


def test_flashinfer_cutedsl_bf16_needs_sm10x_at_runtime(
    monkeypatch, flashinfer_mm_calls
) -> None:
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: False
    )
    x, weight = _bf16(8, 2048), _bf16(1024, 2048)

    actual = _cutedsl_gemm()(None, x, weight)

    assert flashinfer_mm_calls == []
    torch.testing.assert_close(actual, F.linear(x, weight), rtol=0, atol=0)


def test_flashinfer_cutedsl_bf16_unavailable_backend_uses_auto_gemm(
    monkeypatch,
) -> None:
    spec = layer_utils._FLASHINFER_BF16_BACKENDS["flashinfer_cutedsl"]
    monkeypatch.setitem(
        layer_utils._FLASHINFER_BF16_BACKENDS,
        "flashinfer_cutedsl",
        layer_utils._FlashInferBf16Backend(
            flashinfer_backend=spec.flashinfer_backend,
            is_supported=lambda: False,
            can_implement=spec.can_implement,
        ),
    )

    gemm = layer_utils.dispatch_unquantized_gemm("flashinfer_cutedsl")

    assert gemm is layer_utils.cuda_unquantized_gemm


def _tune_low_m(gemm, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
    """The startup pass: one M = 32 call with the M buckets up to 32."""
    import vllm.utils.flashinfer as fi_utils

    buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(32)
    assert buckets == (1, 2, 4, 8, 16, 32)
    with fi_utils.autotune(tuning_buckets=buckets, round_up=True):
        gemm(None, _bf16(32, weight.shape[1]), weight, bias)


@pytest.fixture
def bf16_cache_lookups(monkeypatch) -> list[tuple[int, bool]]:
    """Record (M, cache hit) of every serving-time BF16 GEMM tactic lookup."""
    from flashinfer.autotuner import AutoTuner

    lookups: list[tuple[int, bool]] = []
    search_cache = AutoTuner.search_cache

    def spy(self, custom_op, runners, input_shapes, *args, **kwargs):
        result = search_cache(self, custom_op, runners, input_shapes, *args, **kwargs)
        if custom_op.startswith("bf16_gemm") and not self.is_tuning_mode:
            lookups.append((input_shapes[0][0], result[0]))
        return result

    monkeypatch.setattr(AutoTuner, "search_cache", spy)
    AutoTuner.get().clear_cache()
    return lookups


@pytest.mark.parametrize("use_bias", [False, True])
def test_flashinfer_cutedsl_bf16_autotune_covers_every_low_m(
    use_bias: bool, bf16_cache_lookups
) -> None:
    from flashinfer.autotuner import AutoTuner

    n, k = 3072, 2048
    weight = _bf16(n, k)
    bias = _bf16(n) if use_bias else None
    gemm = _cutedsl_gemm()

    # Control: an untuned shape misses.
    gemm(None, _bf16(8, k), weight, bias)
    assert bf16_cache_lookups == [(8, False)]

    AutoTuner.get().clear_cache()
    _tune_low_m(gemm, weight, bias)
    assert not AutoTuner.get().is_tuning_mode

    bf16_cache_lookups.clear()
    m_values = (1, 2, 3, 4, 5, 8, 12, 16, 17, 24, 31, 32)
    for m in m_values:
        x = _bf16(m, k)
        torch.testing.assert_close(
            gemm(None, x, weight, bias), F.linear(x, weight, bias), rtol=2e-2, atol=2e-1
        )
    assert bf16_cache_lookups == [(m, True) for m in m_values]


def test_flashinfer_cutedsl_bf16_cuda_graph_uses_tuned_tactics(
    flashinfer_mm_calls, bf16_cache_lookups
) -> None:
    n, k = 2560, 4096  # no skinny GEMM plan
    weight = _bf16(n, k)
    gemm = _cutedsl_gemm()
    _tune_low_m(gemm, weight, None)

    flashinfer_mm_calls.clear()
    bf16_cache_lookups.clear()
    inputs = {m: _bf16(m, k) for m in (8, 24, 64)}
    for x in inputs.values():
        gemm(None, x, weight)  # compile outside capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = {m: gemm(None, x, weight) for m, x in inputs.items()}
    for x in inputs.values():
        x.copy_(_bf16(*x.shape))
    graph.replay()
    torch.accelerator.synchronize()

    # Eager warm-up and capture hit the tuned tactics; M = 64 stays on torch.
    assert [call[0] for call in flashinfer_mm_calls] == [8, 24, 8, 24]
    assert bf16_cache_lookups == [(8, True), (24, True)] * 2
    for m, x in inputs.items():
        torch.testing.assert_close(
            outputs[m], F.linear(x, weight), rtol=2e-2, atol=2e-1
        )
