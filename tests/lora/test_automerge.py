# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the automerge module (LoRA weight merge).

These tests exercise the merge/restore logic, golden cache, state machine,
and mixin helper methods using CPU tensors — no GPU required.
"""

import pytest
import torch

from vllm.lora.automerge.merge import (
    BF16GoldenCache,
    _get_base_weight_tensor,
    _get_scaling,
    _is_supported_dtype,
    _slice_and_compute_delta,
    bf16_restore_base,
    merge_lora_into_base,
)
from vllm.lora.automerge.state import AutoMergeState, get_state

# ---------------------------------------------------------------------------
# Mark all tests in this file to skip the heavy global cleanup (no GPU init).
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skip_global_cleanup


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


class FakeModule:
    """Minimal stand-in for a vLLM LoRA wrapper module."""

    def __init__(
        self,
        weight: torch.Tensor,
        tp_size: int = 1,
        output_slices: list[int] | None = None,
    ):
        self.weight = weight
        self.tp_size = tp_size
        self.output_slices = output_slices

    def slice_lora_a(self, a):
        return a  # identity for tests

    def slice_lora_b(self, b):
        return b  # identity for tests


class FakeBaseLayerModule:
    """Module with a base_layer attribute wrapping the real weight."""

    def __init__(self, weight: torch.Tensor):
        self.base_layer = FakeModule(weight)


class FakeLoRALayer:
    """Minimal stand-in for a LoRA layer with A/B weights."""

    def __init__(
        self,
        lora_a: torch.Tensor | list,
        lora_b: torch.Tensor | list,
        scaling: float | list[float] = 1.0,
        is_packed: bool = False,
    ):
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scaling = scaling
        self.is_packed = is_packed


class FakeLoRAModel:
    """Minimal stand-in for a LoRAModel with a loras dict."""

    def __init__(self, loras: dict):
        self.loras = loras


class FakeModelManager:
    """Minimal stand-in for LoRAModelManager."""

    def __init__(
        self,
        modules: dict,
        adapters: dict | None = None,
        lora_index_to_id: list | None = None,
    ):
        self.modules = modules
        self._adapters = adapters or {}
        self.lora_index_to_id = lora_index_to_id or []

    def get_adapter(self, adapter_id: int):
        return self._adapters.get(adapter_id)


class FakeLoRAManager:
    """Wraps FakeModelManager to mimic runner.lora_manager."""

    def __init__(self, model_manager: FakeModelManager):
        self._adapter_manager = model_manager


class FakeRunner:
    """Minimal stand-in for a model runner."""

    def __init__(self, lora_manager: FakeLoRAManager | None = None):
        self.lora_manager = lora_manager


def _make_weight(
    rows: int = 64, cols: int = 64, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    return torch.randn(rows, cols, dtype=dtype)


def _make_lora_pair(
    rank: int = 8, rows: int = 64, cols: int = 64, dtype: torch.dtype = torch.bfloat16
):
    """Return (lora_a, lora_b) with shapes compatible for delta = B @ A."""
    A = torch.randn(rank, cols, dtype=dtype)
    B = torch.randn(rows, rank, dtype=dtype)
    return A, B


# ---------------------------------------------------------------------------
# Tests: _get_base_weight_tensor
# ---------------------------------------------------------------------------


class TestGetBaseWeightTensor:
    def test_direct_weight(self):
        w = _make_weight()
        mod = FakeModule(w)
        assert _get_base_weight_tensor(mod) is w

    def test_base_layer_weight(self):
        w = _make_weight()
        mod = FakeBaseLayerModule(w)
        result = _get_base_weight_tensor(mod)
        assert result is w

    def test_no_weight(self):
        mod = object()
        assert _get_base_weight_tensor(mod) is None

    def test_non_tensor_weight(self):
        class BadModule:
            weight = "not a tensor"

        assert _get_base_weight_tensor(BadModule()) is None


# ---------------------------------------------------------------------------
# Tests: _get_scaling
# ---------------------------------------------------------------------------


class TestGetScaling:
    def test_scaling_attr(self):
        layer = FakeLoRALayer(torch.zeros(1), torch.zeros(1), scaling=2.5)
        assert _get_scaling(layer) == pytest.approx(2.5)

    def test_alpha_over_rank(self):
        class Layer:
            lora_alpha = 16
            rank = 8

        assert _get_scaling(Layer()) == pytest.approx(2.0)

    def test_default_1(self):
        assert _get_scaling(object()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: _is_supported_dtype
# ---------------------------------------------------------------------------


class TestIsSupportedDtype:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_supported(self, dtype):
        assert _is_supported_dtype(dtype) is True

    @pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.int8])
    def test_unsupported(self, dtype):
        assert _is_supported_dtype(dtype) is False


# ---------------------------------------------------------------------------
# Tests: BF16GoldenCache
# ---------------------------------------------------------------------------


class TestBF16GoldenCache:
    def test_empty_cache(self):
        cache = BF16GoldenCache()
        assert not cache.initialized
        assert cache.get("foo") is None

    def test_ensure_populated(self):
        w = _make_weight()
        modules = {"layer1": FakeModule(w)}
        cache = BF16GoldenCache(device="gpu")
        n = cache.ensure_populated(modules, {"layer1"})
        assert n == 1
        assert cache.initialized
        golden = cache.get("layer1")
        assert golden is not None
        assert torch.equal(golden, w)
        # Golden is a clone, not the same object
        assert golden is not w

    def test_idempotent(self):
        w = _make_weight()
        modules = {"layer1": FakeModule(w)}
        cache = BF16GoldenCache(device="gpu")
        cache.ensure_populated(modules, {"layer1"})
        # Mutate the original weight
        w.fill_(999.0)
        # Golden should still have the original values
        golden = cache.get("layer1")
        assert not torch.equal(golden, w)
        # Second call is a no-op
        n = cache.ensure_populated(modules, {"layer1"})
        assert n == 1

    def test_clear(self):
        w = _make_weight()
        modules = {"layer1": FakeModule(w)}
        cache = BF16GoldenCache()
        cache.ensure_populated(modules, {"layer1"})
        cache.clear()
        assert not cache.initialized
        assert cache.get("layer1") is None

    def test_missing_module(self):
        cache = BF16GoldenCache()
        n = cache.ensure_populated({}, {"layer1"})
        assert n == 0
        assert cache.initialized

    def test_cpu_device(self):
        w = _make_weight()
        modules = {"layer1": FakeModule(w)}
        cache = BF16GoldenCache(device="cpu")
        cache.ensure_populated(modules, {"layer1"})
        golden = cache.get("layer1")
        assert golden is not None
        assert golden.device == torch.device("cpu")
        assert torch.equal(golden, w)

    def test_gpu_device_on_cpu_tensor(self):
        """GPU mode with CPU tensors keeps them on the same device (CPU)."""
        w = _make_weight()
        modules = {"layer1": FakeModule(w)}
        cache = BF16GoldenCache(device="gpu")
        cache.ensure_populated(modules, {"layer1"})
        golden = cache.get("layer1")
        assert golden is not None
        # Since w is on CPU, clone stays on CPU
        assert torch.equal(golden, w)

    def test_off_mode(self):
        cache = BF16GoldenCache(device="off")
        assert cache.is_off
        w = _make_weight()
        modules = {"layer1": FakeModule(w)}
        n = cache.ensure_populated(modules, {"layer1"})
        assert n == 0
        assert cache.get("layer1") is None
        assert cache.initialized


# ---------------------------------------------------------------------------
# Tests: _slice_and_compute_delta
# ---------------------------------------------------------------------------


class TestSliceAndComputeDelta:
    def test_basic_delta(self):
        W = _make_weight(64, 64)
        A, B = _make_lora_pair(8, 64, 64, dtype=W.dtype)
        delta = _slice_and_compute_delta(
            module=FakeModule(W),
            A=A,
            B=B,
            W=W,
            scale=1.0,
            module_name="test",
        )
        assert delta is not None
        expected = torch.matmul(B, A)
        assert torch.allclose(delta, expected, atol=1e-5)

    def test_with_scaling(self):
        W = _make_weight(64, 64)
        A, B = _make_lora_pair(8, 64, 64, dtype=W.dtype)
        delta = _slice_and_compute_delta(
            module=FakeModule(W),
            A=A,
            B=B,
            W=W,
            scale=0.5,
            module_name="test",
        )
        expected = torch.matmul(B, A) * 0.5
        assert torch.allclose(delta, expected, atol=1e-5)

    def test_shape_mismatch_returns_none(self):
        W = _make_weight(64, 64)
        # A/B produce a 32x64 delta, but W is 64x64
        A = torch.randn(8, 64, dtype=W.dtype)
        B = torch.randn(32, 8, dtype=W.dtype)
        delta = _slice_and_compute_delta(
            module=FakeModule(W),
            A=A,
            B=B,
            W=W,
            scale=1.0,
            module_name="test",
        )
        assert delta is None

    def test_slice_failure_returns_none(self):
        W = _make_weight(64, 64)
        A, B = _make_lora_pair(8, 64, 64, dtype=W.dtype)

        class BadSliceModule(FakeModule):
            def slice_lora_a(self, a):
                raise RuntimeError("slice failed")

        delta = _slice_and_compute_delta(
            module=BadSliceModule(W, tp_size=2),
            A=A,
            B=B,
            W=W,
            scale=1.0,
            module_name="test",
        )
        assert delta is None


# ---------------------------------------------------------------------------
# Tests: merge_lora_into_base / bf16_restore_base
# ---------------------------------------------------------------------------


class TestMergeAndRestore:
    def _setup(self, dtype=torch.bfloat16, golden_device="gpu"):
        W = _make_weight(64, 64, dtype=dtype)
        A, B = _make_lora_pair(8, 64, 64, dtype=dtype)
        module = FakeModule(W.clone())
        lora_layer = FakeLoRALayer(A, B, scaling=1.0)
        lora_model = FakeLoRAModel({"layer1": lora_layer})
        mm = FakeModelManager({"layer1": module})
        golden_cache = BF16GoldenCache(device=golden_device)
        return W, A, B, module, lora_model, mm, golden_cache

    def test_merge_modifies_weight(self):
        W_orig, A, B, module, lora_model, mm, cache = self._setup()
        result = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        assert result.ok
        assert result.merged_modules == 1
        # Weight should now be golden + B @ A
        expected = W_orig + torch.matmul(B, A).to(W_orig.dtype)
        assert torch.allclose(module.weight, expected, atol=1e-3)

    def test_restore_reverts_weight(self):
        W_orig, A, B, module, lora_model, mm, cache = self._setup()
        merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        # Now restore
        result = bf16_restore_base(
            model_manager=mm,
            golden_cache=cache,
            lora_module_names={"layer1"},
        )
        assert result.ok
        assert result.merged_modules == 1
        assert torch.allclose(module.weight, W_orig, atol=1e-6)

    def test_merge_then_restore_no_drift(self):
        """Merge and restore 10 times — weight should match original exactly."""
        W_orig, A, B, module, lora_model, mm, cache = self._setup()
        for _ in range(10):
            merge_lora_into_base(
                model_manager=mm,
                lora_model=lora_model,
                golden_cache=cache,
            )
            bf16_restore_base(
                model_manager=mm,
                golden_cache=cache,
                lora_module_names={"layer1"},
            )
        assert torch.equal(module.weight, W_orig)

    def test_unsupported_dtype_skipped(self):
        """FP8 weights should be skipped when validate_dtypes=True."""
        W = torch.randn(64, 64, dtype=torch.float32).to(torch.float8_e4m3fn)
        A = torch.randn(8, 64, dtype=torch.float32)
        B = torch.randn(64, 8, dtype=torch.float32)
        module = FakeModule(W)
        lora_layer = FakeLoRALayer(A, B)
        lora_model = FakeLoRAModel({"layer1": lora_layer})
        mm = FakeModelManager({"layer1": module})
        cache = BF16GoldenCache()
        result = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
            validate_dtypes=True,
        )
        assert not result.ok
        assert result.skipped_modules == 1

    def test_empty_loras(self):
        mm = FakeModelManager({"layer1": FakeModule(_make_weight())})
        lora_model = FakeLoRAModel({})
        cache = BF16GoldenCache()
        result = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        assert not result.ok

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_supported_dtypes(self, dtype):
        W_orig, A, B, module, lora_model, mm, cache = self._setup(dtype=dtype)
        result = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        assert result.ok
        assert result.merged_modules == 1


# ---------------------------------------------------------------------------
# Tests: Off mode (subtract-to-restore)
# ---------------------------------------------------------------------------


class TestOffMode:
    def _setup(self, dtype=torch.bfloat16):
        W = _make_weight(64, 64, dtype=dtype)
        A, B = _make_lora_pair(8, 64, 64, dtype=dtype)
        module = FakeModule(W.clone())
        lora_layer = FakeLoRALayer(A, B, scaling=1.0)
        lora_model = FakeLoRAModel({"layer1": lora_layer})
        mm = FakeModelManager({"layer1": module})
        cache = BF16GoldenCache(device="off")
        return W, A, B, module, lora_model, mm, cache

    def test_merge_off_mode(self):
        W_orig, A, B, module, lora_model, mm, cache = self._setup()
        result = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        assert result.ok
        expected = W_orig + torch.matmul(B, A).to(W_orig.dtype)
        assert torch.allclose(module.weight, expected, atol=1e-3)

    def test_restore_off_mode(self):
        W_orig, A, B, module, lora_model, mm, cache = self._setup()
        merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        result = bf16_restore_base(
            model_manager=mm,
            golden_cache=cache,
            lora_module_names={"layer1"},
            lora_model=lora_model,
        )
        assert result.ok
        # Subtract approach — close but not exact for BF16
        assert torch.allclose(module.weight, W_orig, atol=0.1)

    def test_off_mode_drift(self):
        """Off mode accumulates drift over cycles — verify it's bounded."""
        W_orig, A, B, module, lora_model, mm, cache = self._setup(dtype=torch.float32)
        for _ in range(100):
            merge_lora_into_base(
                model_manager=mm,
                lora_model=lora_model,
                golden_cache=cache,
            )
            bf16_restore_base(
                model_manager=mm,
                golden_cache=cache,
                lora_module_names={"layer1"},
                lora_model=lora_model,
            )
        # FP32: drift should be negligible even after 100 cycles
        assert torch.allclose(module.weight, W_orig, atol=1e-4)

    def test_off_mode_no_lora_model_fails(self):
        """Off mode restore without lora_model should fail gracefully."""
        _, _, _, module, _, mm, cache = self._setup()
        cache._initialized = True
        result = bf16_restore_base(
            model_manager=mm,
            golden_cache=cache,
            lora_module_names={"layer1"},
            lora_model=None,
        )
        assert not result.ok
        assert "off mode" in result.reason


# ---------------------------------------------------------------------------
# Tests: CPU golden mode
# ---------------------------------------------------------------------------


class TestCPUGoldenMode:
    def test_merge_and_restore_cpu_golden(self):
        dtype = torch.bfloat16
        W = _make_weight(64, 64, dtype=dtype)
        A, B = _make_lora_pair(8, 64, 64, dtype=dtype)
        module = FakeModule(W.clone())
        lora_layer = FakeLoRALayer(A, B, scaling=1.0)
        lora_model = FakeLoRAModel({"layer1": lora_layer})
        mm = FakeModelManager({"layer1": module})
        cache = BF16GoldenCache(device="cpu")

        result = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        assert result.ok
        # Golden should be on CPU
        golden = cache.get("layer1")
        assert golden.device == torch.device("cpu")

        # Restore
        result = bf16_restore_base(
            model_manager=mm,
            golden_cache=cache,
            lora_module_names={"layer1"},
        )
        assert result.ok
        assert torch.equal(module.weight, W)


# ---------------------------------------------------------------------------
# Tests: Packed layer merge
# ---------------------------------------------------------------------------


class TestPackedLayerMerge:
    def test_packed_merge(self):
        """Test merge of a packed LoRA layer (e.g., merged QKV)."""
        dtype = torch.bfloat16
        # Base weight: 192 x 64 (3 slices of 64)
        W = torch.randn(192, 64, dtype=dtype)
        module = FakeModule(W.clone(), output_slices=[64, 64, 64])

        rank = 8
        A_list = [torch.randn(rank, 64, dtype=dtype) for _ in range(3)]
        B_list = [torch.randn(64, rank, dtype=dtype) for _ in range(3)]

        lora_layer = FakeLoRALayer(
            A_list, B_list, scaling=[1.0, 1.0, 1.0], is_packed=True
        )
        lora_model = FakeLoRAModel({"layer1": lora_layer})
        mm = FakeModelManager({"layer1": module})
        cache = BF16GoldenCache(device="gpu")

        result = merge_lora_into_base(
            model_manager=mm,
            lora_model=lora_model,
            golden_cache=cache,
        )
        assert result.ok
        assert result.merged_modules == 1

        # Verify each slice got the right delta
        golden = cache.get("layer1")
        for i in range(3):
            delta_i = torch.matmul(B_list[i], A_list[i]).to(dtype)
            start = i * 64
            end = start + 64
            expected_slice = golden[start:end] + delta_i
            assert torch.allclose(module.weight[start:end], expected_slice, atol=1e-3)


# ---------------------------------------------------------------------------
# Tests: AutoMergeState
# ---------------------------------------------------------------------------


class TestAutoMergeState:
    def _make_state_and_runner(self, dtype=torch.bfloat16):
        W = _make_weight(64, 64, dtype=dtype)
        A, B = _make_lora_pair(8, 64, 64, dtype=dtype)
        module = FakeModule(W.clone())
        lora_layer = FakeLoRALayer(A, B)
        lora_model = FakeLoRAModel({"layer1": lora_layer})
        mm = FakeModelManager(
            modules={"layer1": module},
            adapters={42: lora_model},
            lora_index_to_id=[42],
        )
        lora_mgr = FakeLoRAManager(mm)
        runner = FakeRunner(lora_mgr)
        state = AutoMergeState(golden_device="gpu")
        return state, runner, module, W

    def test_merge_active(self):
        state, runner, module, W_orig = self._make_state_and_runner()
        ok = state.merge_active(runner, "my_adapter")
        assert ok
        assert state.merged_lora_name == "my_adapter"
        assert state.merged_adapter_id == 42
        assert state.merge_count == 1
        # Weight should be modified
        assert not torch.equal(module.weight, W_orig)

    def test_unmerge_restores(self):
        state, runner, module, W_orig = self._make_state_and_runner()
        state.merge_active(runner, "my_adapter")
        state.unmerge_if_needed(runner)
        assert state.merged_lora_name is None
        assert state.merged_adapter_id is None
        assert state.unmerge_count == 1
        assert torch.equal(module.weight, W_orig)

    def test_unmerge_noop_when_not_merged(self):
        state, runner, _, _ = self._make_state_and_runner()
        state.unmerge_if_needed(runner)  # should not raise
        assert state.unmerge_count == 0

    def test_merge_fails_no_manager(self):
        runner = FakeRunner(lora_manager=None)
        state = AutoMergeState()
        ok = state.merge_active(runner, "test")
        assert not ok
        assert "no model_manager" in state.last_error

    def test_merge_fails_no_active_adapter(self):
        mm = FakeModelManager(
            modules={"layer1": FakeModule(_make_weight())},
            lora_index_to_id=[],
        )
        runner = FakeRunner(FakeLoRAManager(mm))
        state = AutoMergeState()
        ok = state.merge_active(runner, "test")
        assert not ok
        assert "no active adapter" in state.last_error


# ---------------------------------------------------------------------------
# Tests: get_state (per-runner weak-ref cache)
# ---------------------------------------------------------------------------


class TestGetState:
    def test_same_runner_same_state(self):
        runner = FakeRunner()
        s1 = get_state(runner)
        s2 = get_state(runner)
        assert s1 is s2

    def test_different_runners_different_state(self):
        r1 = FakeRunner()
        r2 = FakeRunner()
        assert get_state(r1) is not get_state(r2)


# ---------------------------------------------------------------------------
# Tests: LoRAModelRunnerMixin helper methods
# ---------------------------------------------------------------------------


class TestMixinHelpers:
    """Test the static helper methods on LoRAModelRunnerMixin."""

    def test_unique_lora_names_empty(self):
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        assert LoRAModelRunnerMixin._unique_lora_names(set()) == []
        assert LoRAModelRunnerMixin._unique_lora_names(None) == []

    def test_unique_lora_names_dedup(self):
        from vllm.lora.request import LoRARequest
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        reqs = {
            LoRARequest("adapter_a", 1, "/fake/path"),
            LoRARequest("adapter_a", 2, "/fake/path"),
            LoRARequest("adapter_b", 3, "/fake/path"),
        }
        names = LoRAModelRunnerMixin._unique_lora_names(reqs)
        assert names == ["adapter_a", "adapter_b"]

    def test_batch_has_mixed_base_and_lora(self):
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        fn = LoRAModelRunnerMixin._batch_has_mixed_base_and_lora

        # All LoRA
        assert fn((1, 1, 1), (1, 1, 1)) is False
        # All base
        assert fn((0, 0, 0), (0, 0, 0)) is False
        # Mixed
        assert fn((0, 1, 0), (1, 1, 1)) is True
        assert fn((1, 1, 1), (0, 1, 1)) is True

    def test_batch_mixed_with_negative(self):
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

        fn = LoRAModelRunnerMixin._batch_has_mixed_base_and_lora
        # -1 is treated as base (<=0)
        assert fn((-1, 1), (1, 1)) is True


# ---------------------------------------------------------------------------
# Tests: LoRAConfig validation for enable_lora_weight_merge
# ---------------------------------------------------------------------------


class TestLoRAConfigWeightMerge:
    def test_default_false(self):
        from vllm.config.lora import LoRAConfig

        cfg = LoRAConfig()
        assert cfg.enable_lora_weight_merge is False

    def test_default_golden_device(self):
        from vllm.config.lora import LoRAConfig

        cfg = LoRAConfig()
        assert cfg.lora_weight_merge_golden_device == "cpu"

    def test_enable_with_max_loras_1(self):
        from vllm.config.lora import LoRAConfig

        cfg = LoRAConfig(enable_lora_weight_merge=True)
        assert cfg.enable_lora_weight_merge is True

    def test_golden_device_options(self):
        from vllm.config.lora import LoRAConfig

        for dev in ("cpu", "gpu", "off"):
            cfg = LoRAConfig(
                enable_lora_weight_merge=True,
                lora_weight_merge_golden_device=dev,
            )
            assert cfg.lora_weight_merge_golden_device == dev

    def test_enable_with_max_loras_gt_1_no_error(self):
        """When max_loras > 1, config should still be valid."""
        from vllm.config.lora import LoRAConfig

        cfg = LoRAConfig(enable_lora_weight_merge=True, max_loras=2)
        assert cfg.enable_lora_weight_merge is True
