"""Tests for KV cache NaN detection debug infrastructure."""
import pytest
import torch


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_DEBUG_KV_CACHE_NANS", raising=False)


class TestNanDetection:
    def test_env_var_default_off(self) -> None:
        from vllm.envs import VLLM_DEBUG_KV_CACHE_NANS

        assert VLLM_DEBUG_KV_CACHE_NANS == 0

    def test_python_nan_check_detects_nan(self) -> None:
        from vllm._custom_ops import _check_nan_in_cache_source

        x = torch.tensor([1.0, 2.0, float("nan"), 4.0])
        import os
        os.environ["VLLM_DEBUG_KV_CACHE_NANS"] = "1"
        try:
            count = _check_nan_in_cache_source(x, "test_layer", num_tokens_to_check=4)
            assert count == 1
        finally:
            os.environ.pop("VLLM_DEBUG_KV_CACHE_NANS", None)

    def test_python_nan_check_clean_tensor(self) -> None:
        from vllm._custom_ops import _check_nan_in_cache_source

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        import os
        os.environ["VLLM_DEBUG_KV_CACHE_NANS"] = "1"
        try:
            count = _check_nan_in_cache_source(x, "test_layer", num_tokens_to_check=4)
            assert count == 0
        finally:
            os.environ.pop("VLLM_DEBUG_KV_CACHE_NANS", None)

    def test_python_nan_check_disabled_when_env_off(self) -> None:
        from vllm._custom_ops import _check_nan_in_cache_source

        x = torch.tensor([float("nan"), float("nan")])
        count = _check_nan_in_cache_source(x, "test_layer")
        assert count == 0

    def test_nan_counter_starts_at_zero(self) -> None:
        from vllm._custom_ops import get_nan_cache_write_count

        assert get_nan_cache_write_count() == 0
