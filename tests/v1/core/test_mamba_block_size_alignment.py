# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "all" mode: the block size has to stay a multiple of the chunk size.

In "all" mode the cached states are read off chunk ends: the write-back in
``MambaMixer2.conv_ssm_forward`` walks the intermediate states with a stride of
``mamba_block_size // chunk_size`` chunks per block. A block size that is not a
multiple of the chunk size makes that stride drift away from the block spacing,
so a slot is filled from a chunk end at the wrong offset, and when the walk
leaves the intermediate states the slice is empty rather than an error -- the
state write then dies with a shape mismatch and takes EngineCore with it
(#57266).

``--mamba-block-size`` used to pick the alignment by itself, which is how a
block size of 288 tokens against a chunk size of 128 was reachable.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf

pytestmark = pytest.mark.cpu_test

# Falcon-H1-Tiny-90M-Instruct, the smallest model that reproduces #57266.
MAMBA_CHUNK_SIZE = 128
CONV_STATE_SHAPE = (896, 3)
SSM_STATE_SHAPE = (24, 32, 64)
NUM_KV_HEADS = 2
HEAD_SIZE = 64

# (--mamba-block-size, derived mamba block size). The mamba state is 103680
# bytes and a token of attention is 512, so 203 tokens of attention pay for one
# state and the derivation rounds that up to the alignment. Requests that are
# not a multiple of the chunk size round up to a common multiple, which can be
# well above the 256 the constraint alone would need: 176 -> 1408.
DERIVED_BLOCK_SIZES = [
    (None, 256),
    (16, 256),
    (32, 256),
    (64, 256),
    (96, 384),
    (128, 256),
    (176, 1408),
    (256, 256),
]


class _FakeModel:
    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], ...]:
        return (CONV_STATE_SHAPE, SSM_STATE_SHAPE)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return (torch.bfloat16, torch.bfloat16)


class _FakeBackend:
    @staticmethod
    def customize_spec(spec):
        return spec

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[MultipleOf]:
        return [MultipleOf(16)]


def _hybrid_config(
    monkeypatch: pytest.MonkeyPatch, mamba_block_size: int | None
) -> VllmConfig:
    cache_config = CacheConfig(
        enable_prefix_caching=True, mamba_block_size=mamba_block_size
    )
    cache_config.mamba_cache_mode = "all"
    vllm_config = VllmConfig(cache_config=cache_config)
    vllm_config.model_config = SimpleNamespace(
        architecture="FalconH1ForCausalLM",
        dtype=torch.bfloat16,
        use_mla=False,
        get_num_kv_heads=lambda parallel_config: NUM_KV_HEADS,
        get_head_size=lambda: HEAD_SIZE,
        get_mamba_chunk_size=lambda: MAMBA_CHUNK_SIZE,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        lambda architecture, model_config: (_FakeModel, architecture),
    )
    # The public entry point needs backend discovery, so drive the aligner that
    # owns the derivation directly.
    return vllm_config


@pytest.mark.parametrize("requested,expected", DERIVED_BLOCK_SIZES)
def test_mamba_all_mode_block_size_is_chunk_aligned(
    monkeypatch: pytest.MonkeyPatch, requested: int | None, expected: int
) -> None:
    vllm_config = _hybrid_config(monkeypatch, requested)

    Platform._align_hybrid_block_size(vllm_config, _FakeBackend)

    block_size = vllm_config.cache_config.mamba_block_size
    assert block_size % MAMBA_CHUNK_SIZE == 0, (
        f"--mamba-block-size {requested} produced {block_size}, "
        f"which no chunk end lands on"
    )
    assert block_size == expected
