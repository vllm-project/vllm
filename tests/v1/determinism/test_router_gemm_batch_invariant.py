# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the batch-invariant MoE router GEMM dispatch.

The selector tests in this file deliberately use metadata-only tensor doubles.
That keeps policy, fallback, and cache coverage runnable on CPU-only CI and does
not make DeepGEMM a test dependency.  The CUDA tests exercise the real Triton
kernels and are skipped unless an SM90-or-newer NVIDIA GPU is available.
"""

import importlib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.layers.batch_invariant as batch_invariant
import vllm.model_executor.layers.linear as linear
from vllm.platforms import current_platform

requires_sm90 = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="Router GEMM integration tests require CUDA SM90 or newer",
)
requires_sm90_exact = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability(90)),
    reason="Router backend dispatch is currently qualified only on CUDA SM90",
)


class _TensorSpec:
    """The tensor metadata consumed by the selector, without allocating CUDA."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
        *,
        device: str = "cuda:0",
        contiguous: bool = True,
    ) -> None:
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        self.ndim = len(shape)
        self.layout = torch.strided
        self._contiguous = contiguous

    def dim(self) -> int:
        return self.ndim

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self) -> tuple[int, ...]:
        if self.ndim == 1:
            return (1,)
        if self._contiguous:
            return (self.shape[1], 1)
        return (1, self.shape[0])

    def numel(self) -> int:
        return self.shape.numel()


def _router_specs(
    *,
    m: int = 12,
    n: int = 128,
    k: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    weight_dtype: torch.dtype | None = None,
    input_contiguous: bool = True,
    weight_contiguous: bool = True,
    device: str = "cuda:0",
) -> tuple[_TensorSpec, _TensorSpec]:
    return (
        _TensorSpec(
            (m, k),
            dtype,
            device=device,
            contiguous=input_contiguous,
        ),
        _TensorSpec(
            (n, k),
            weight_dtype or dtype,
            device=device,
            contiguous=weight_contiguous,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_router_backend_cache():
    batch_invariant._reset_router_gemm_backend_cache(reset_deepgemm=True)
    yield
    batch_invariant._reset_router_gemm_backend_cache(reset_deepgemm=True)


@pytest.fixture
def selector_environment(monkeypatch: pytest.MonkeyPatch):
    """Present an SM90 device while keeping policy tests CPU-only."""
    monkeypatch.setattr(
        batch_invariant.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        batch_invariant.current_platform,
        "is_cuda_alike",
        lambda: True,
    )
    monkeypatch.setattr(
        batch_invariant.current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability == 90,
    )
    monkeypatch.setattr(
        batch_invariant.current_platform,
        "has_device_capability",
        lambda capability, device_id=0: capability <= 90,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: (9, 0),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(
        batch_invariant,
        "_router_deepgemm_available",
        lambda: (True, "available"),
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "persistent"),
        ("auto", "auto"),
        ("persistent", "persistent"),
        ("full_k", "full_k"),
        ("deepgemm", "deepgemm"),
        ("  auto  ", "auto"),
    ],
)
def test_read_router_gemm_choice(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
    expected: str,
):
    if value is None:
        monkeypatch.delenv(
            "VLLM_BATCH_INVARIANT_ROUTER_GEMM",
            raising=False,
        )
    else:
        monkeypatch.setenv(
            "VLLM_BATCH_INVARIANT_ROUTER_GEMM",
            value,
        )
    assert batch_invariant._read_router_gemm_choice() == expected


def test_read_router_gemm_choice_rejects_unknown_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "VLLM_BATCH_INVARIANT_ROUTER_GEMM",
        "cublaslt",
    )
    with pytest.raises(ValueError, match="auto.*persistent.*full_k.*deepgemm"):
        batch_invariant._read_router_gemm_choice()


@pytest.mark.usefixtures("selector_environment")
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.bfloat16, "deepgemm"),
        (torch.float16, "full_k"),
        (torch.float32, "full_k"),
    ],
    ids=["bf16-deepgemm", "fp16-full-k", "fp32-full-k"],
)
def test_auto_selects_dtype_tier(dtype: torch.dtype, expected: str):
    input_spec, weight_spec = _router_specs(dtype=dtype)
    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert decision.requested == "auto"
    assert decision.selected == expected
    assert not decision.preflighted


@pytest.mark.usefixtures("selector_environment")
def test_auto_uses_full_k_when_deepgemm_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        batch_invariant,
        "_router_deepgemm_available",
        lambda: (False, "deepgemm import failed"),
    )
    input_spec, weight_spec = _router_specs()

    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert decision.selected == "full_k"
    assert "deepgemm" in decision.reason.lower()


@pytest.mark.usefixtures("selector_environment")
@pytest.mark.parametrize(
    "kwargs,bias,reason",
    [
        ({"dtype": torch.float64}, None, "dtype"),
        ({"m": 2049, "dtype": torch.float16}, None, "M"),
        ({"n": 256}, None, "N"),
        ({"k": 1024}, None, "K"),
        ({"input_contiguous": False}, None, "contiguous"),
        ({"weight_contiguous": False}, None, "contiguous"),
        ({"weight_dtype": torch.float16}, None, "dtype"),
        ({"device": "cpu"}, None, "CUDA"),
        ({}, _TensorSpec((128,)), "bias"),
    ],
)
def test_auto_guard_falls_back_to_persistent(
    kwargs: dict,
    bias: _TensorSpec | None,
    reason: str,
):
    input_spec, weight_spec = _router_specs(**kwargs)
    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        bias,
        requested_mode="auto",
    )

    assert decision.selected == "persistent"
    assert reason.lower() in decision.reason.lower()


@pytest.mark.usefixtures("selector_environment")
@pytest.mark.parametrize(
    "input_shape,weight_shape",
    [
        ((1, 12, 2048), (128, 2048)),
        ((12, 2048), (1, 128, 2048)),
    ],
)
def test_auto_rank_guard_falls_back_to_persistent(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
):
    input_spec = _TensorSpec(input_shape)
    weight_spec = _TensorSpec(weight_shape)
    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )
    assert decision.selected == "persistent"
    assert "2d" in decision.reason.lower()


@pytest.mark.usefixtures("selector_environment")
def test_auto_bf16_large_m_still_uses_deepgemm():
    input_spec, weight_spec = _router_specs(m=12238)
    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )
    assert decision.selected == "deepgemm"


@pytest.mark.usefixtures("selector_environment")
@pytest.mark.parametrize(
    ("m", "expected"),
    [(1, "full_k"), (128, "full_k"), (129, "persistent"), (12238, "persistent")],
)
def test_auto_fp32_uses_full_k_until_the_m_guard(
    m: int,
    expected: str,
):
    input_spec, weight_spec = _router_specs(m=m, dtype=torch.float32)
    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )
    assert decision.selected == expected
    if expected == "persistent":
        assert "M=" in decision.reason


@pytest.mark.usefixtures("selector_environment")
def test_auto_large_m_without_deepgemm_uses_persistent(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        batch_invariant,
        "_router_deepgemm_available",
        lambda: (False, "not installed"),
    )
    input_spec, weight_spec = _router_specs(m=12238)
    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )
    assert decision.selected == "persistent"
    assert "deepgemm" in decision.reason.lower()
    assert "full_k" in decision.reason.lower()


@pytest.mark.usefixtures("selector_environment")
@pytest.mark.parametrize("mode", ["persistent", "full_k", "deepgemm"])
def test_forced_mode_selects_exact_backend(mode: str):
    input_spec, weight_spec = _router_specs()
    decision = batch_invariant._select_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode=mode,
    )
    assert decision.requested == mode
    assert decision.selected == mode
    assert not decision.preflighted


@pytest.mark.usefixtures("selector_environment")
@pytest.mark.parametrize(
    "mode,kwargs,bias",
    [
        ("deepgemm", {"dtype": torch.float16}, None),
        ("deepgemm", {"n": 256}, None),
        ("deepgemm", {"input_contiguous": False}, None),
        ("full_k", {"m": 2049}, None),
        ("full_k", {"k": 1024}, None),
        ("full_k", {}, _TensorSpec((128,))),
    ],
)
def test_forced_fast_backend_is_fail_fast(
    mode: str,
    kwargs: dict,
    bias: _TensorSpec | None,
):
    input_spec, weight_spec = _router_specs(**kwargs)
    with pytest.raises(ValueError, match=mode):
        batch_invariant._select_router_gemm_backend(
            input_spec,
            weight_spec,
            bias,
            requested_mode=mode,
        )


@pytest.mark.usefixtures("selector_environment")
def test_forced_deepgemm_missing_dependency_is_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        batch_invariant,
        "_router_deepgemm_available",
        lambda: (False, "not installed"),
    )
    input_spec, weight_spec = _router_specs()
    with pytest.raises(ValueError, match="deepgemm.*not installed"):
        batch_invariant._select_router_gemm_backend(
            input_spec,
            weight_spec,
            requested_mode="deepgemm",
        )


def test_deepgemm_missing_symbol_failure_is_cached(
    monkeypatch: pytest.MonkeyPatch,
):
    import_module = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(importlib, "import_module", import_module)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="bf16_gemm_nt"):
            batch_invariant._load_router_deepgemm_impl()

    import_module.assert_called_once_with("deep_gemm")


@pytest.mark.parametrize(
    ("preflight_name", "candidate_name", "comparison"),
    [
        (
            "_preflight_deepgemm",
            "_run_router_deepgemm",
            "DeepGEMM/persistent",
        ),
        ("_preflight_full_k", "_run_router_full_k", "full-K/persistent"),
    ],
)
def test_candidate_preflight_rejects_bitwise_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    preflight_name: str,
    candidate_name: str,
    comparison: str,
):
    input_tensor = torch.zeros((1, 2), dtype=torch.bfloat16)
    weight = torch.zeros((3, 2), dtype=torch.bfloat16)
    candidate = torch.ones((1, 3), dtype=torch.bfloat16)
    persistent = torch.zeros((1, 3), dtype=torch.bfloat16)
    monkeypatch.setattr(
        batch_invariant,
        candidate_name,
        Mock(return_value=candidate),
    )
    monkeypatch.setattr(
        batch_invariant,
        "_run_router_persistent",
        Mock(return_value=persistent),
    )

    with pytest.raises(RuntimeError, match=comparison):
        getattr(batch_invariant, preflight_name)(input_tensor, weight, None)


def test_fp32_full_k_accuracy_accepts_a_reference_within_tolerance():
    input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    reference = input_tensor @ weight.t()

    batch_invariant._validate_router_gemm_fp32_accuracy(
        reference,
        input_tensor,
        weight,
    )


def test_fp32_full_k_accuracy_rejects_an_incorrect_result():
    input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    incorrect = torch.zeros((1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="full-K/FP64"):
        batch_invariant._validate_router_gemm_fp32_accuracy(
            incorrect,
            input_tensor,
            weight,
        )


def _patch_preflights(
    monkeypatch: pytest.MonkeyPatch,
    *,
    deepgemm: Callable | None = None,
    full_k: Callable | None = None,
    persistent: Callable | None = None,
) -> dict[str, Mock]:
    mocks = {
        "deepgemm": Mock(side_effect=deepgemm),
        "full_k": Mock(side_effect=full_k),
        "persistent": Mock(side_effect=persistent),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(
            batch_invariant,
            f"_preflight_{name}",
            mock,
        )
    return mocks


@pytest.mark.usefixtures("selector_environment")
def test_auto_preflight_failure_cascades_to_full_k(
    monkeypatch: pytest.MonkeyPatch,
):
    def deepgemm_failure(*args, **kwargs):
        raise RuntimeError("deepgemm JIT failed")

    preflights = _patch_preflights(
        monkeypatch,
        deepgemm=deepgemm_failure,
    )
    input_spec, weight_spec = _router_specs()

    decision = batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert decision.selected == "full_k"
    assert decision.preflighted
    assert "deepgemm" in decision.reason.lower()
    assert "failed" in decision.reason.lower()
    preflights["deepgemm"].assert_called_once()
    preflights["full_k"].assert_called_once()
    preflights["persistent"].assert_not_called()


@pytest.mark.usefixtures("selector_environment")
def test_auto_unwritable_deepgemm_jit_cache_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    def unwritable_cache(*args, **kwargs):
        raise PermissionError("DG_JIT_CACHE_DIR is not writable")

    preflights = _patch_preflights(
        monkeypatch,
        deepgemm=unwritable_cache,
    )
    input_spec, weight_spec = _router_specs()

    decision = batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert decision.selected == "full_k"
    assert "PermissionError" in decision.reason
    assert "not writable" in decision.reason
    preflights["full_k"].assert_called_once()


@pytest.mark.usefixtures("selector_environment")
def test_auto_preflight_failures_cascade_to_persistent(
    monkeypatch: pytest.MonkeyPatch,
):
    def deepgemm_failure(*args, **kwargs):
        raise RuntimeError("deepgemm JIT failed")

    def full_k_failure(*args, **kwargs):
        raise RuntimeError("full-k compilation failed")

    preflights = _patch_preflights(
        monkeypatch,
        deepgemm=deepgemm_failure,
        full_k=full_k_failure,
    )
    input_spec, weight_spec = _router_specs()

    decision = batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert decision.selected == "persistent"
    assert decision.preflighted
    assert "deepgemm" in decision.reason.lower()
    assert "full_k" in decision.reason.lower()
    preflights["persistent"].assert_called_once()


@pytest.mark.usefixtures("selector_environment")
def test_auto_fp32_accuracy_preflight_failure_cascades_to_persistent(
    monkeypatch: pytest.MonkeyPatch,
):
    def full_k_accuracy_failure(*args, **kwargs):
        raise RuntimeError("full-K/FP64 accuracy validation failed")

    preflights = _patch_preflights(
        monkeypatch,
        full_k=full_k_accuracy_failure,
    )
    input_spec, weight_spec = _router_specs(dtype=torch.float32)

    decision = batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert decision.selected == "persistent"
    assert "full-K/FP64" in decision.reason
    preflights["deepgemm"].assert_not_called()
    preflights["full_k"].assert_called_once()
    preflights["persistent"].assert_called_once()


@pytest.mark.usefixtures("selector_environment")
@pytest.mark.parametrize("mode", ["deepgemm", "full_k", "persistent"])
def test_forced_preflight_failure_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    def failure(*args, **kwargs):
        raise RuntimeError(f"{mode} preflight failed")

    preflights = _patch_preflights(
        monkeypatch,
        **{mode: failure},
    )
    input_spec, weight_spec = _router_specs()

    with pytest.raises(RuntimeError, match=f"{mode} preflight failed"):
        batch_invariant.prewarm_router_gemm_backend(
            input_spec,
            weight_spec,
            requested_mode=mode,
        )
    for backend, preflight in preflights.items():
        assert preflight.call_count == (1 if backend == mode else 0)


@pytest.mark.usefixtures("selector_environment")
def test_get_decision_does_not_select_or_preflight_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
):
    preflights = _patch_preflights(monkeypatch)
    input_spec, weight_spec = _router_specs()

    decision = batch_invariant.get_router_gemm_backend_decision(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert decision.selected is None
    assert decision.reason == "not_preflighted"
    assert not decision.preflighted
    assert all(mock.call_count == 0 for mock in preflights.values())


@pytest.mark.usefixtures("selector_environment")
def test_prewarm_caches_decision_and_logs_once(
    monkeypatch: pytest.MonkeyPatch,
):
    preflights = _patch_preflights(monkeypatch)
    log = Mock()
    monkeypatch.setattr(batch_invariant.logger, "info", log)
    input_spec, weight_spec = _router_specs()

    first = batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )
    second = batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    assert first == second
    assert preflights["deepgemm"].call_count == 1
    log.assert_called_once()
    assert "requested=%s selected=%s" in log.call_args.args[0]


@pytest.mark.usefixtures("selector_environment")
def test_cache_signature_includes_every_tensor_device(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_preflights(monkeypatch)
    input_spec, weight_spec = _router_specs()
    cached = batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )
    assert cached.selected == "deepgemm"

    other_weight = _TensorSpec(
        (128, 2048),
        torch.bfloat16,
        device="cuda:1",
    )
    miss = batch_invariant.get_router_gemm_backend_decision(
        input_spec,
        other_weight,
        requested_mode="auto",
    )
    assert not miss.preflighted
    assert miss.selected is None


@pytest.mark.usefixtures("selector_environment")
def test_cache_miss_during_capture_does_not_preflight(
    monkeypatch: pytest.MonkeyPatch,
):
    preflights = _patch_preflights(monkeypatch)
    monkeypatch.setattr(
        batch_invariant,
        "_router_gemm_is_capturing_or_compiling",
        lambda input: True,
    )
    input_spec, weight_spec = _router_specs()

    with pytest.raises(RuntimeError, match="not preflighted"):
        batch_invariant.prewarm_router_gemm_backend(
            input_spec,
            weight_spec,
            requested_mode="auto",
        )
    assert all(mock.call_count == 0 for mock in preflights.values())


@pytest.mark.usefixtures("selector_environment")
def test_concurrent_prewarm_runs_preflight_once(
    monkeypatch: pytest.MonkeyPatch,
):
    entered = 0
    entered_lock = threading.Lock()

    def count_preflight(*args, **kwargs):
        nonlocal entered
        with entered_lock:
            entered += 1

    _patch_preflights(
        monkeypatch,
        deepgemm=count_preflight,
    )
    input_spec, weight_spec = _router_specs()
    barrier = threading.Barrier(4)

    def prewarm():
        barrier.wait()
        return batch_invariant.prewarm_router_gemm_backend(
            input_spec,
            weight_spec,
            requested_mode="auto",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        decisions = list(executor.map(lambda _: prewarm(), range(4)))

    assert entered == 1
    assert {decision.selected for decision in decisions} == {"deepgemm"}


@pytest.mark.usefixtures("selector_environment")
def test_runtime_failure_after_prewarm_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_preflights(monkeypatch)
    input_spec, weight_spec = _router_specs(dtype=torch.float16)
    batch_invariant.prewarm_router_gemm_backend(
        input_spec,
        weight_spec,
        requested_mode="auto",
    )

    full_k = Mock(side_effect=RuntimeError("asynchronous kernel failure"))
    persistent = Mock()
    monkeypatch.setattr(batch_invariant, "_run_router_full_k", full_k)
    monkeypatch.setattr(
        batch_invariant,
        "_run_router_persistent",
        persistent,
    )
    monkeypatch.setattr(
        batch_invariant,
        "VLLM_BATCH_INVARIANT_ROUTER_GEMM",
        "auto",
    )

    with pytest.raises(RuntimeError, match="asynchronous kernel failure"):
        batch_invariant._router_gemm_batch_invariant_op_impl(
            input_spec,
            weight_spec,
        )
    full_k.assert_called_once()
    persistent.assert_not_called()


@pytest.mark.parametrize(
    (
        "bi_enabled",
        "is_cuda",
        "is_cuda_alike",
        "is_router",
        "expected_backend",
    ),
    [
        (True, True, True, True, "router"),
        (True, False, True, True, "persistent"),
        (False, True, True, True, "baseline"),
        (True, True, True, False, "baseline"),
        (True, False, False, True, "baseline"),
    ],
)
def test_unquantized_linear_scopes_router_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    bi_enabled: bool,
    is_cuda: bool,
    is_cuda_alike: bool,
    is_router: bool,
    expected_backend: str,
):
    baseline = Mock(return_value="baseline")
    router = Mock(return_value="router")
    persistent = Mock(return_value="persistent")
    monkeypatch.setattr(linear, "vllm_is_batch_invariant", lambda: bi_enabled)
    monkeypatch.setattr(linear.current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(
        linear.current_platform,
        "is_cuda_alike",
        lambda: is_cuda_alike,
    )
    monkeypatch.setattr(
        linear,
        "is_layer_moe_router_gate",
        lambda prefix: is_router,
    )
    monkeypatch.setattr(linear, "router_gemm_batch_invariant", router)
    monkeypatch.setattr(linear, "linear_batch_invariant", persistent)
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda: baseline,
    )
    layer = SimpleNamespace(prefix="model.layers.0.mlp.gate", weight=object())
    x = object()
    bias = object()

    result = linear.UnquantizedLinearMethod().apply(layer, x, bias)

    assert result == expected_backend
    assert router.call_count == int(expected_backend == "router")
    assert persistent.call_count == int(expected_backend == "persistent")
    assert baseline.call_count == int(expected_backend == "baseline")


def _run_eager(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    return op(a, b).clone()


def _run_cuda_graph(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Warm the kernel, capture one replay, and return a detached result."""
    op(a, b)
    torch.cuda.synchronize(a.device)

    static_a = a.clone()
    static_b = b.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_a, static_b)

    graph.replay()
    torch.cuda.synchronize(a.device)
    return static_out.clone()


@requires_sm90
@pytest.mark.parametrize(
    "op_name",
    ["matmul_persistent", "matmul_full_k"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bf16", "fp16", "fp32"],
)
@pytest.mark.parametrize(
    "run",
    [_run_eager, _run_cuda_graph],
    ids=["eager", "cuda-graph"],
)
def test_router_gemm_row_is_bitwise_invariant_across_batch_positions(
    op_name: str,
    dtype: torch.dtype,
    run: Callable[
        [
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
    ],
):
    """A logical row must not change when its batch position or M changes."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(314159)
    k, n = 2048, 128
    weight = torch.randn(
        n,
        k,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    weight_t = weight.t()
    assert not weight_t.is_contiguous()
    needle = torch.randn(
        k,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    op = getattr(batch_invariant, op_name)

    expected = run(op, needle.unsqueeze(0), weight_t)[0]
    for m, position in ((2, 0), (4, 2), (12, 11)):
        batch = torch.randn(
            m,
            k,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        batch[position].copy_(needle)
        actual = run(op, batch, weight_t)[position]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_sm90
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bf16", "fp16", "fp32"],
)
def test_full_k_cuda_graph_replays_are_bitwise_stable(dtype: torch.dtype):
    """Repeated graph replay must not introduce an inter-CTA reduction race."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(271828)
    a = torch.randn(
        12,
        2048,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(
        128,
        2048,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    b = weight.t()
    assert not b.is_contiguous()
    op = batch_invariant.matmul_full_k

    op(a, b)
    torch.cuda.synchronize(device)
    static_a = a.clone()
    static_b = b.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = op(static_a, static_b)

    graph.replay()
    torch.cuda.synchronize(device)
    expected = static_out.clone()
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(static_out, expected, rtol=0, atol=0)


@requires_sm90
@pytest.mark.parametrize("m", [1, 12, 128])
def test_fp32_full_k_matches_the_fp64_preflight_envelope(m: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(161803 + m)
    input_tensor = torch.randn(
        m,
        2048,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.1)
    weight = torch.randn(
        128,
        2048,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).mul_(0.02)
    output = batch_invariant.matmul_full_k(input_tensor, weight.t())
    torch.cuda.synchronize(device)

    batch_invariant._validate_router_gemm_fp32_accuracy(
        output,
        input_tensor,
        weight,
    )


@requires_sm90_exact
def test_router_custom_op_is_an_opaque_compile_and_graph_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    """The selector must execute outside Dynamo and finish before capture."""
    monkeypatch.setattr(
        batch_invariant,
        "VLLM_BATCH_INVARIANT_ROUTER_GEMM",
        "full_k",
    )
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(161803)
    x = torch.randn(
        12,
        2048,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    weight = torch.randn(
        128,
        2048,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    compiled = torch.compile(
        lambda a, b: batch_invariant.router_gemm_batch_invariant(a, b),
        backend="eager",
        fullgraph=True,
    )

    eager = compiled(x, weight)
    decision = batch_invariant.get_router_gemm_backend_decision(x, weight)
    assert decision.selected == "full_k"
    assert decision.preflighted

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = compiled(x, weight)
    graph.replay()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(graph_output, eager, rtol=0, atol=0)
