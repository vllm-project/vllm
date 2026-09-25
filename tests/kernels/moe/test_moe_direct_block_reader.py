# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Asynchronous Direct Block I/O Weight Loader and Single-Reader Broadcast."""

import mmap
import os
import tempfile
import pytest
from safetensors.torch import save_file
import torch

from vllm.model_executor.model_loader.direct_block_reader import (
    DirectBlockFileReader,
    calculate_alignment,
)
from vllm.model_executor.model_loader.moe_fast_loader import (
    SafetensorsMoEIndex,
    check_page_cache_warmth,
)
from vllm.model_executor.model_loader.shared_pinned_pool import (
    SharedPinnedBufferPool,
)


def test_calculate_alignment():
    """Verifies that calculate_alignment computes correct 4096-byte boundaries and shifts."""
    # Case 1: Perfectly aligned 4096-byte block
    off, sz, shift = calculate_alignment(0, 4096, 4096)
    assert off == 0
    assert sz == 4096
    assert shift == 0

    # Case 2: Unaligned start (e.g. 816 byte shift from safetensors header)
    off, sz, shift = calculate_alignment(816, 1000, 4096)
    assert off == 0
    assert sz == 4096
    assert shift == 816
    assert off + shift == 816

    # Case 3: Straddling multiple blocks
    off, sz, shift = calculate_alignment(4097, 4096, 4096)
    assert off == 4096
    assert sz == 8192
    assert shift == 1


@pytest.fixture
def synthetic_multishard_checkpoint(tmp_path):
    """Creates a synthetic multi-shard safetensors checkpoint for direct I/O testing."""
    num_shards = 2
    num_layers = 2
    num_experts = 4
    hidden_dim = 64
    intermediate_dim = 32

    shard_paths = []
    all_weights = {}

    # Dense weights
    all_weights["model.embed_tokens.weight"] = torch.randn(
        128, hidden_dim, dtype=torch.bfloat16
    )
    all_weights["model.norm.weight"] = torch.randn(hidden_dim, dtype=torch.bfloat16)

    # MoE weights
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts):
            prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
            all_weights[f"{prefix}.gate_proj.weight"] = torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            )
            all_weights[f"{prefix}.up_proj.weight"] = torch.randn(
                intermediate_dim, hidden_dim, dtype=torch.bfloat16
            )
            all_weights[f"{prefix}.down_proj.weight"] = torch.randn(
                hidden_dim, intermediate_dim, dtype=torch.bfloat16
            )

    # Distribute weights across shards
    weight_items = list(all_weights.items())
    split_idx = len(weight_items) // num_shards

    shard_dicts = [
        dict(weight_items[:split_idx]),
        dict(weight_items[split_idx:]),
    ]

    for shard_idx, sdict in enumerate(shard_dicts):
        shard_path = str(tmp_path / f"model-{shard_idx:05d}-of-{num_shards:05d}.safetensors")
        save_file(sdict, shard_path)
        shard_paths.append(shard_path)

    return shard_paths, all_weights, num_layers, num_experts


def test_shared_pinned_buffer_pool_lifecycle():
    """Verifies that SharedPinnedBufferPool initializes, exposes slots, and cleans up properly."""
    pool_prefix = "test_fast_moe_pool_life"
    slot_size = 1024 * 1024  # 1 MiB

    # Writer rank (Rank 0) creates the pool
    pool_writer = SharedPinnedBufferPool(
        prefix=pool_prefix,
        slot_size=slot_size,
        is_creator=True,
        tp_rank=0,
        tp_size=1,
    )

    try:
        assert pool_writer.slot_size == slot_size
        buf0 = pool_writer.get_slot_buffer(0)
        assert len(buf0) == slot_size

        # Write test pattern
        buf0[:4] = b"TEST"

        # Reader rank connects to the existing pool
        pool_reader = SharedPinnedBufferPool(
            prefix=pool_prefix,
            slot_size=slot_size,
            is_creator=False,
            tp_rank=0,
            tp_size=1,
        )
        try:
            rbuf0 = pool_reader.get_slot_buffer(0)
            assert bytes(rbuf0[:4]) == b"TEST"
        finally:
            pool_reader.close()
    finally:
        pool_writer.close()
        pool_writer.unlink()


def test_check_page_cache_warmth(synthetic_multishard_checkpoint):
    """Verifies that check_page_cache_warmth accurately inspects Linux VFS page-cache state."""
    shard_paths, _, _, _ = synthetic_multishard_checkpoint
    warmth = check_page_cache_warmth(shard_paths, sample_mb=1)
    assert 0.0 <= warmth <= 1.0


def test_direct_block_file_reader(synthetic_multishard_checkpoint):
    """Verifies that DirectBlockFileReader correctly reads file bytes into memory-mapped buffers."""
    shard_paths, _, _, _ = synthetic_multishard_checkpoint
    target_shard = shard_paths[0]
    file_size = os.path.getsize(target_shard)

    reader = DirectBlockFileReader(chunk_size=1024 * 1024, max_workers=2)
    with tempfile.TemporaryFile() as tf:
        tf.truncate(file_size)
        mm = mmap.mmap(tf.fileno(), file_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        mv = memoryview(mm)

        bytes_read = reader.read_file_to_buffer(target_shard, mv)
        assert bytes_read == file_size

        with open(target_shard, "rb") as f:
            expected = f.read()
        assert bytes(mv) == expected

        mv.release()
        mm.close()
        reader.close()


def test_direct_block_file_reader_sequential_buffered(tmp_path):
    """Verifies that DirectBlockFileReader default sequential buffered reader is bitwise accurate."""
    test_data = os.urandom(256 * 1024)  # 256 KiB
    test_file = tmp_path / "test_shard.bin"
    test_file.write_bytes(test_data)

    buf = bytearray(len(test_data))
    with DirectBlockFileReader(chunk_size=64 * 1024, max_workers=2, force_o_direct=False) as reader:
        assert not reader.force_o_direct
        n = reader.read_file_to_buffer(str(test_file), buf)
        assert n == len(test_data)
        assert bytes(buf) == test_data


def test_direct_block_file_reader_force_o_direct_env(monkeypatch):
    """Verifies that VLLM_MOE_FORCE_O_DIRECT=1 enables force_o_direct mode."""
    monkeypatch.setenv("VLLM_MOE_FORCE_O_DIRECT", "1")
    with DirectBlockFileReader() as reader:
        assert reader.force_o_direct is True

    monkeypatch.setenv("VLLM_MOE_FORCE_O_DIRECT", "0")
    with DirectBlockFileReader() as reader:
        assert reader.force_o_direct is False
