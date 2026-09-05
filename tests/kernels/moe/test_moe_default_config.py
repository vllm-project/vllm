# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating rules for the fused-MoE default-config N tile.

`get_default_config` widens `BLOCK_SIZE_N` to 256 for large batches. The wider
tile is a 2-9% regression below the batch threshold and does not fit every
device's shared memory, so both gates have to hold or the change would make
small-batch and consumer-GPU configs worse.
"""

import pytest

from vllm.model_executor.layers.fused_moe import fused_moe

# Qwen3-30B-A3B: E=128, hidden=2048, moe_intermediate=768, topk=8.
SHAPE = dict(E=128, N=2 * 768, K=2048, topk=8)

# Hopper/A100 hold the wider tile; consumer Ampere/Ada (101376 B) do not.
BIG_SMEM = 232448
SMALL_SMEM = 101376


@pytest.fixture
def smem(monkeypatch):
    """Pin the device shared-memory budget so the test is host-independent."""

    def _set(nbytes):
        # raising=False so the invariant tests below still exercise a build
        # that does not consult shared memory at all -- they must hold either
        # way, and only the widening tests should depend on this symbol.
        monkeypatch.setattr(
            fused_moe,
            "get_max_shared_memory_bytes",
            lambda *a, **k: nbytes,
            raising=False,
        )
        monkeypatch.setattr(fused_moe.current_platform, "is_cuda_alike", lambda: True)

    return _set


def cfg(M, dtype=None):
    return fused_moe.get_default_config(M=M, dtype=dtype, **SHAPE)


@pytest.mark.parametrize("M", [1, 64, 128, 512])
def test_small_batches_keep_the_narrow_tile(smem, M):
    """Below the threshold the wider tile measured 2-9% slower on two shapes."""
    smem(BIG_SMEM)
    assert cfg(M)["BLOCK_SIZE_N"] == (64 if M <= 64 else 128)


@pytest.mark.parametrize("M", [1024, 2048, 8192])
def test_large_batches_widen_the_n_tile(smem, M):
    smem(BIG_SMEM)
    assert cfg(M)["BLOCK_SIZE_N"] == 256


@pytest.mark.parametrize("M", [1024, 8192])
def test_devices_without_shared_memory_are_unchanged(smem, M):
    """Consumer Ampere/Ada cannot hold the wider tile and must be untouched."""
    smem(SMALL_SMEM)
    assert cfg(M)["BLOCK_SIZE_N"] == 128


def test_widening_fits_the_reported_shared_memory(smem):
    """The tile the heuristic picks must actually fit what it checked."""
    smem(BIG_SMEM)
    c = cfg(2048)
    operands = (
        c["BLOCK_SIZE_M"] * c["BLOCK_SIZE_K"] + c["BLOCK_SIZE_K"] * c["BLOCK_SIZE_N"]
    )
    assert operands * 2 * c["num_stages"] <= BIG_SMEM


@pytest.mark.parametrize("dtype", ["fp8_w8a8", "int8_w8a16", "int4_w4a16"])
def test_quantized_dtypes_are_unchanged(smem, dtype):
    """Only bf16/fp16 was measured; quantized tiles keep today's config."""
    smem(BIG_SMEM)
    assert cfg(8192, dtype=dtype)["BLOCK_SIZE_N"] != 256
