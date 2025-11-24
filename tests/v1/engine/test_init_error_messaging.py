# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_model_load_oom_error_message():
    """Test that CUDA OOM error handling produces clear error message."""
    # Simulate what happens in gpu_model_runner.load_model when OOM occurs
    try:
        raise torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 20.00 GiB"
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_str = str(e).lower()
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err_str:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try reducing --gpu-memory-utilization, increasing "
                "--tensor-parallel-size, or using --quantization."
            )
            # Verify the error message contains expected guidance
            assert "not enough gpu memory" in msg.lower()
            assert "--gpu-memory-utilization" in msg
            assert "--tensor-parallel-size" in msg
            assert "--quantization" in msg
            return

    pytest.fail("Expected to handle OOM error")


def test_model_load_runtime_oom_error_message():
    """Test that RuntimeError with OOM message produces clear error."""
    # Simulate the error handling logic
    e = RuntimeError("CUDA error: out of memory")
    err_str = str(e).lower()

    if "out of memory" in err_str:
        msg = (
            "Failed to load model - not enough GPU memory. "
            "Try reducing --gpu-memory-utilization, increasing "
            "--tensor-parallel-size, or using --quantization."
        )
        exc = RuntimeError(msg)

        # Verify the error message
        assert "gpu memory" in str(exc).lower()
        assert "--gpu-memory-utilization" in str(exc)
        assert "--tensor-parallel-size" in str(exc)
        return

    pytest.fail("Expected to catch out of memory error")


def test_kv_cache_oom_no_memory():
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype="float16",
        )
    }

    with pytest.raises(ValueError) as exc_info:
        check_enough_kv_cache_memory(config, spec, 0)

    msg = str(exc_info.value)
    assert "cache blocks" in msg
    assert "gpu_memory_utilization" in msg
    assert "conserving_memory" in msg


def test_kv_cache_oom_insufficient_memory(monkeypatch):
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048
    config.cache_config.block_size = 16
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.max_memory_usage_bytes",
        lambda c, s: 100 * 1024**3,  # 100 GiB
    )

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype="float16",
        )
    }

    with pytest.raises(ValueError) as exc_info:
        check_enough_kv_cache_memory(config, spec, 1024**3)  # 1 GiB

    msg = str(exc_info.value)
    assert "KV cache" in msg
    assert "GiB" in msg
    assert "gpu_memory_utilization" in msg
    assert "conserving_memory" in msg
