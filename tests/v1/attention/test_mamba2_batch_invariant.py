# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 layers under VLLM_BATCH_INVARIANT: state layout and startup checks."""

import pytest
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.layers.mamba.exact_replay import (
    exact_replay_spec_decode_layout,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionBackend,
    Mamba2AttentionMetadataBuilder,
    exact_replay_decode_positions,
)

SHAPE_KWARGS = dict(
    intermediate_size=1024,
    tp_world_size=1,
    n_groups=1,
    num_heads=16,
    head_dim=64,
    state_size=128,
    conv_kernel=4,
)


def test_state_layout_adds_buffers_and_fp32_ssm_state(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    base = MambaStateShapeCalculator.mamba2_state_shape(**SHAPE_KWARGS)
    assert len(base) == 2
    assert MambaStateDtypeCalculator.mamba2_state_dtype(
        torch.bfloat16, "auto", "auto"
    ) == (torch.bfloat16, torch.bfloat16)

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    shapes = MambaStateShapeCalculator.mamba2_state_shape(
        **SHAPE_KWARGS, chunk_size=256
    )
    assert shapes[:2] == base
    # x, raw dt, B for chunk_size tokens, token-major
    assert shapes[2:] == ((256, 16, 64), (256, 16), (256, 1, 128))
    with pytest.raises(ValueError, match="chunk size"):
        MambaStateShapeCalculator.mamba2_state_shape(**SHAPE_KWARGS)
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


def test_align_copy_covers_every_state(monkeypatch):
    """Prefix caching copies a block's state; the replay buffers are state."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert len(MambaStateCopyFuncCalculator.mamba2_state_copy_func()) == 2

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    shapes = MambaStateShapeCalculator.mamba2_state_shape(
        **SHAPE_KWARGS, chunk_size=256
    )
    assert len(MambaStateCopyFuncCalculator.mamba2_state_copy_func()) == len(shapes)


def _config(**cache_overrides) -> VllmConfig:
    kwargs = dict(enable_prefix_caching=False)
    kwargs.update(cache_overrides)
    return VllmConfig(
        cache_config=CacheConfig(**kwargs),
        scheduler_config=SchedulerConfig(max_model_len=2048, is_encoder_decoder=False),
    )


@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_check_accepts_the_supported_configuration(mamba_cache_mode):
    assert Mamba2AttentionBackend.supports_batch_invariance()
    cfg = _config()
    cfg.cache_config.mamba_cache_mode = mamba_cache_mode
    Mamba2AttentionBackend.check_batch_invariant_config(cfg)


def test_decode_keeps_full_cudagraph_support(monkeypatch):
    """Decode rows run fixed-shape kernels, so decode-only graphs stay allowed."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    support = Mamba2AttentionMetadataBuilder.get_cudagraph_support(_config(), None)
    assert support == AttentionCGSupport.UNIFORM_BATCH


def test_decode_positions_ignore_cudagraph_padding_rows():
    # sequence lengths after this step's token: mid chunk, completing a chunk,
    # just past a boundary, and two padding rows (length 0) as the runner pads
    seq_lens = torch.tensor([301, 512, 513, 0, 0], dtype=torch.int32)
    pos = exact_replay_decode_positions(seq_lens, 256)
    assert pos.tolist() == [44, 255, 0, -1, -1]
    out = torch.full((5,), 7, dtype=torch.int32)
    assert exact_replay_decode_positions(seq_lens, 256, out=out) is out
    assert out.tolist() == [44, 255, 0, -1, -1]


def test_spec_decode_layout_stops_at_chunk_final_drafts():
    """Drafts on a chunk's last position, and everything after them, are
    skipped; a first token completing a chunk lets its drafts start the next
    one. Rows: mid chunk, a draft lands on the chunk end, the first token
    completes the chunk, a trimmed one-token row, a CUDA graph padding row."""
    num_computed = [44, 254, 255, 100]
    query_lens = [3, 3, 3, 1, 3]
    seq_lens = torch.tensor([n + q for n, q in zip(num_computed, query_lens)] + [0])
    qsl = torch.tensor([0, *query_lens]).cumsum(0)
    pos, token_idx = exact_replay_spec_decode_layout(seq_lens, qsl, 2, 256)
    assert pos.tolist() == [
        [44, 45, 46],
        [254, -1, -1],
        [255, 0, 1],
        [100, -1, -1],
        [-1, -1, -1],
    ]
    assert token_idx.tolist() == [
        [0, 1, 2],
        [3, -1, -1],
        [6, 7, 8],
        [9, -1, -1],
        [-1, -1, -1],
    ]


def test_check_rejects_spec_decode_without_model_runner_v2(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    cfg = _config()
    monkeypatch.setattr(VllmConfig, "num_speculative_tokens", property(lambda _: 1))
    with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER"):
        Mamba2AttentionBackend.check_batch_invariant_config(cfg)


@pytest.mark.parametrize(
    "mutate,message",
    [
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
