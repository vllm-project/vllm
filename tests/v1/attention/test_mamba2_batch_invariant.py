# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 layers under VLLM_BATCH_INVARIANT: state layout, startup checks and
the host-side metadata of the replayed SSD step."""

import pytest
import torch

import vllm.envs as envs
from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mamba2_attn import (
    Mamba2AttentionBackend,
    Mamba2AttentionMetadataBuilder,
    build_exact_replay_metadata,
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


def _config(**cache_overrides) -> VllmConfig:
    kwargs = dict(enable_prefix_caching=False)
    kwargs.update(cache_overrides)
    return VllmConfig(
        cache_config=CacheConfig(**kwargs),
        scheduler_config=SchedulerConfig(max_model_len=2048, is_encoder_decoder=False),
    )


def test_check_accepts_the_supported_configuration():
    assert Mamba2AttentionBackend.supports_batch_invariance()
    Mamba2AttentionBackend.check_batch_invariant_config(_config())


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
    assert pos.tolist() == [44, 255, 0, 0, 0]
    out = torch.full((5,), -1, dtype=torch.int32)
    assert exact_replay_decode_positions(seq_lens, 256, out=out) is out
    assert out.tolist() == [44, 255, 0, 0, 0]


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


CHUNK = 256


def _metadata(num_computed, query_lens):
    return build_exact_replay_metadata(
        num_computed, query_lens, CHUNK, torch.device("cpu")
    )


def test_metadata_for_decode_rows():
    # three decode rows: 300 computed (44 tokens into a chunk), exactly at a
    # boundary, and one token short of completing a chunk
    m = _metadata([300, 512, 767], [1, 1, 1])
    assert m.num_aug_tokens == 44 + 1 + 1 + 255 + 1
    assert m.cu_seqlens.tolist() == [0, 45, 46, 302]
    # every augmented sequence is a single chunk
    assert m.cu_chunk_seqlens.tolist() == [0, 45, 46, 302]
    assert m.last_chunk_indices.tolist() == [0, 1, 2]
    assert m.seq_idx.tolist() == [0, 1, 2]
    assert m.has_boundary_state.tolist() == [True, True, True]
    # only the third row fills its chunk this step
    assert m.boundary_rows.tolist() == [2]
    assert m.boundary_chunk_idx.tolist() == [2]
    # buffered tokens: 44 of row 0, none of row 1, 255 of row 2
    assert m.buffered_seq.tolist() == [0] * 44 + [2] * 255
    assert m.buffered_pos.tolist() == list(range(44)) + list(range(255))
    assert m.buffered_dst.tolist() == list(range(44)) + list(range(46, 301))
    assert m.step_dst.tolist() == [44, 45, 301]
    # rows 0 and 1 append their token to the buffer; row 2 completed a chunk
    assert m.store_src.tolist() == [44, 45]
    assert m.store_seq.tolist() == [0, 1]
    assert m.store_pos.tolist() == [44, 0]


def test_metadata_for_prefill_rows():
    # fresh 600-token prefill; resume at 300 with 213 tokens; resume at 100
    # with 413 tokens
    m = _metadata([0, 300, 100], [600, 213, 413])
    assert m.num_aug_tokens == 600 + (44 + 213) + (100 + 413)
    assert m.cu_seqlens.tolist() == [0, 600, 857, 1370]
    assert m.cu_chunk_seqlens.tolist() == [0, 256, 512, 600, 856, 857, 1113, 1369, 1370]
    assert m.last_chunk_indices.tolist() == [2, 4, 7]
    assert m.seq_idx.tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    # row 1 resumes at 300 -> boundary 256 holds a state; the others start
    # from zero (row 2's boundary is 0)
    assert m.has_boundary_state.tolist() == [False, True, False]
    # last completed chunks: row 0 -> [256,512) (chunk 1); row 1 -> the chunk
    # ending at 512 (chunk 3); row 2 -> the chunk ending at 512 (chunk 6)
    assert m.boundary_rows.tolist() == [0, 1, 2]
    assert m.boundary_chunk_idx.tolist() == [1, 3, 6]
    # buffered tokens re-fed for rows 1 and 2
    assert m.buffered_seq.tolist() == [1] * 44 + [2] * 100
    assert m.buffered_dst.tolist() == list(range(600, 644)) + list(range(857, 957))
    # trailing partial chunks stored back: 88 tokens of row 0, 1 of row 1,
    # 1 of row 2, all at buffer positions starting from 0
    assert m.store_src.tolist() == list(range(512, 600)) + [856] + [1369]
    assert m.store_seq.tolist() == [0] * 88 + [1] + [2]
    assert m.store_pos.tolist() == list(range(88)) + [0] + [0]
