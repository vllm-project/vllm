# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 layers under VLLM_BATCH_INVARIANT: state layout and supported settings.

Every Mamba2 model must get the partial-chunk buffers and an fp32 SSM state
from the shared state calculators, and engine settings the replayed SSD path
does not support must be rejected at startup.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionBackend

SHAPE_KWARGS = dict(
    intermediate_size=1024,
    tp_world_size=1,
    n_groups=1,
    num_heads=16,
    head_dim=64,
    state_size=128,
    conv_kernel=4,
)


def test_state_shape_appends_partial_chunk_buffers(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    base = MambaStateShapeCalculator.mamba2_state_shape(**SHAPE_KWARGS)
    assert len(base) == 2

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    shapes = MambaStateShapeCalculator.mamba2_state_shape(
        **SHAPE_KWARGS, chunk_size=256
    )
    assert shapes[:2] == base
    # x, raw dt, B for chunk_size tokens, token-major
    assert shapes[2:] == ((256, 16, 64), (256, 16), (256, 1, 128))
    with pytest.raises(ValueError, match="chunk size"):
        MambaStateShapeCalculator.mamba2_state_shape(**SHAPE_KWARGS)


def test_state_dtype_keeps_fp32_ssm_state(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert MambaStateDtypeCalculator.mamba2_state_dtype(
        torch.bfloat16, "auto", "auto"
    ) == (torch.bfloat16, torch.bfloat16)

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    expected = (torch.bfloat16, torch.float32, *([torch.bfloat16] * 3))
    for ssm_dtype in ("auto", "float32"):
        assert (
            MambaStateDtypeCalculator.mamba2_state_dtype(
                torch.bfloat16, "auto", ssm_dtype
            )
            == expected
        )
    with pytest.raises(ValueError, match="float32"):
        MambaStateDtypeCalculator.mamba2_state_dtype(torch.bfloat16, "auto", "bfloat16")


def test_backend_supports_batch_invariance():
    assert Mamba2AttentionBackend.supports_batch_invariance()


def _config(**cache_overrides) -> VllmConfig:
    kwargs = dict(enable_prefix_caching=False)
    kwargs.update(cache_overrides)
    return VllmConfig(
        cache_config=CacheConfig(**kwargs),
        scheduler_config=SchedulerConfig(max_model_len=2048, is_encoder_decoder=False),
    )


def test_check_accepts_the_supported_configuration():
    Mamba2AttentionBackend.check_batch_invariant_config(_config())


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda c: setattr(c.cache_config, "mamba_cache_mode", "align"), "prefix"),
        (lambda c: setattr(c.cache_config, "use_replayssm", True), "replayssm"),
        (lambda c: setattr(c.parallel_config, "enable_dbo", True), "micro-batching"),
        (lambda c: setattr(c.parallel_config, "ubatch_size", 2), "micro-batching"),
        (
            lambda c: setattr(c.parallel_config, "pipeline_parallel_size", 2),
            "PP=1",
        ),
    ],
)
def test_check_rejects_unsupported_settings(mutate, message):
    cfg = _config()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        Mamba2AttentionBackend.check_batch_invariant_config(cfg)
