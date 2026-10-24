# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DualChunkRotaryEmbedding device handling."""

import pytest
import torch

from vllm.model_executor.layers.rotary_embedding.dual_chunk_rope import (
    DualChunkRotaryEmbedding,
)
from vllm.platforms import current_platform

# Lightweight unit test (no distributed env / large GPU buffers): skip the
# global post-test cleanup for speed, per the skip_global_cleanup convention.
pytestmark = pytest.mark.skip_global_cleanup

HEAD_SIZE = 32
ROTARY_DIM = 32
MAX_POS = 128
BASE = 10000.0
CHUNK_SIZE = 64
LOCAL_SIZE = 8
DTYPE = torch.float


def _make_embedding() -> DualChunkRotaryEmbedding:
    return DualChunkRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POS,
        base=BASE,
        is_neox_style=True,
        dtype=DTYPE,
        chunk_size=CHUNK_SIZE,
        local_size=LOCAL_SIZE,
    )


def test_device_follows_platform_type(monkeypatch, default_vllm_config):
    """Regression guard for the "Torch not compiled with CUDA enabled" crash.

    DualChunkRotaryEmbedding must place its cos/sin caches on the active
    platform's device (current_platform.device_type), not a hard-coded "cuda".
    The device index is pinned and the platform device type is forced to "cpu"
    so the test runs on any runner - the real cos/sin build calls
    `.to(device=self.device)`, which must target an available device. With the
    old hard-coded "cuda" this would build `self.device = cuda:0` and the
    assertions (`type == "cpu"`, `type != "cuda"`) would fail.
    """
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(current_platform, "device_type", "cpu")

    emb = _make_embedding()

    assert emb.device == torch.device("cpu", 0)
    assert emb.device.type != "cuda"
    # The cos/sin caches were built end-to-end on the platform device.
    chunk_len = CHUNK_SIZE - LOCAL_SIZE
    assert emb.cos_sin_q_cache.shape == (chunk_len, ROTARY_DIM)
    assert emb.cos_sin_k_cache.shape == (MAX_POS, ROTARY_DIM)
    assert emb.cos_sin_q_cache.device.type == "cpu"


def test_device_matches_real_platform(monkeypatch, default_vllm_config):
    """On the real platform (no device-type mock), the device type must equal
    current_platform.device_type. Pins the portable contract on whatever
    backend CI runs on; the device index still comes from
    torch.accelerator.current_device_index()."""
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    emb = _make_embedding()

    assert emb.device.type == current_platform.device_type
    assert emb.device.index == 0
    assert emb.cos_sin_q_cache.device.type == current_platform.device_type