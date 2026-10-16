# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 layers under VLLM_BATCH_INVARIANT: state layout and startup checks."""

import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
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


MAMBA2_MODEL = dict(
    model="AntonV/mamba2-130m-hf",
    hf_overrides={"architectures": ["Mamba2ForCausalLM"]},
    max_model_len=2048,
)
DENSE_MODEL = dict(model="Qwen/Qwen2.5-0.5B", max_model_len=2048)


def _config(
    compilation_config=None, model: dict | None = None, **cache_overrides
) -> VllmConfig:
    kwargs = dict(enable_prefix_caching=False)
    kwargs.update(cache_overrides)
    extra = {"compilation_config": compilation_config} if compilation_config else {}
    if model is not None:
        extra["model_config"] = ModelConfig(**model)
    return VllmConfig(
        cache_config=CacheConfig(**kwargs),
        scheduler_config=SchedulerConfig(max_model_len=2048, is_encoder_decoder=False),
        **extra,
    )


def test_check_accepts_the_supported_configuration():
    assert Mamba2AttentionBackend.supports_batch_invariance()
    Mamba2AttentionBackend.check_batch_invariant_config(_config())


def _norm_op_entries(cfg: VllmConfig) -> list[str]:
    return [
        op for op in cfg.compilation_config.custom_ops if "mixer2_gated_rms_norm" in op
    ]


def test_batch_invariant_mode_enables_the_mixer_norm_op_for_mamba_models(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert _norm_op_entries(_config(model=MAMBA2_MODEL)) == ["+mixer2_gated_rms_norm"]
    # an unrelated op whose name merely ends the same way does not count
    cfg = _config(
        compilation_config=CompilationConfig(custom_ops=["+foo_mixer2_gated_rms_norm"]),
        model=MAMBA2_MODEL,
    )
    assert "+mixer2_gated_rms_norm" in cfg.compilation_config.custom_ops
    # pure transformers and the default mode keep their configuration
    assert _norm_op_entries(_config(model=DENSE_MODEL)) == []
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert _norm_op_entries(_config(model=MAMBA2_MODEL)) == []


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
        (
            lambda c: c.compilation_config.custom_ops.append("-mixer2_gated_rms_norm"),
            "mixer2_gated_rms_norm",
        ),
    ],
)
def test_check_rejects_unsupported_settings(mutate, message):
    cfg = _config()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        Mamba2AttentionBackend.check_batch_invariant_config(cfg)
