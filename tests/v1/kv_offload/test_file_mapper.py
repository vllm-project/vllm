# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FileMapper."""

from unittest.mock import MagicMock

from vllm.v1.kv_offload.base import (
    OffloadingSpec,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.file_mapper import FileMapper

# ---------------------------------------------------------------------------
# Shared mocks (mirrors test_fs_tier.py pattern)
# ---------------------------------------------------------------------------

_MOCK_VLLM_CONFIG = MagicMock()
_MOCK_VLLM_CONFIG.model_config.model = "test-model"
_MOCK_VLLM_CONFIG.cache_config.block_size = 16
_MOCK_VLLM_CONFIG.cache_config.cache_dtype = "torch.float32"
_MOCK_VLLM_CONFIG.parallel_config.tensor_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.pipeline_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.prefill_context_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.decode_context_parallel_size = 1
_MOCK_VLLM_CONFIG.parallel_config.rank = 0

_MOCK_KV_CACHE_CONFIG = MagicMock()
_MOCK_KV_CACHE_CONFIG.kv_cache_groups = []

_MOCK_OFFLOADING_SPEC = MagicMock(spec=OffloadingSpec)
_MOCK_OFFLOADING_SPEC.vllm_config = _MOCK_VLLM_CONFIG
_MOCK_OFFLOADING_SPEC.kv_cache_config = _MOCK_KV_CACHE_CONFIG


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_mapper(**kwargs) -> FileMapper:
    defaults = dict(
        root_dir="/tmp/cache",
        model_name="test-model",
        hash_block_size=16,
        gpu_blocks_per_file=1,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        dtype="float16",
    )
    defaults.update(kwargs)
    return FileMapper(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_file_name_full_structure():
    """
    Path must match: <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin

    Concretely:
      - The segment immediately after base_path must end with `_r0`
      - The next segment is the first 3 hex chars of the block hash
      - The next segment is <2 hex chars>_g<group_idx>
      - The final segment is <full hash hex>.bin
    """
    rank = 3
    group_idx = 2
    block_hash = bytes(range(8))  # deterministic, non-zero bytes
    fm = make_mapper(rank=rank)
    key = make_offload_key(block_hash, group_idx)
    path = fm.get_file_name(key)

    hash_hex = get_offload_block_hash(key).hex()
    expected_path = (
        f"{fm.base_path}_r{rank}/{hash_hex[:3]}/"
        f"{hash_hex[3:5]}_g{group_idx}/{hash_hex}.bin"
    )
    assert path == expected_path


def test_get_run_config_fields():
    fm = make_mapper(
        model_name="my-model",
        dtype="bfloat16",
        tp_size=2,
        gpu_blocks_per_file=8,
    )
    cfg = fm.get_run_config()
    assert cfg["model_name"] == "my-model"
    assert cfg["dtype"] == "bfloat16"
    assert cfg["tp_size"] == 2
    assert cfg["gpu_blocks_per_file"] == 8


def test_get_config_file_path():
    fm = make_mapper()
    config_path = fm.get_config_file_path()
    assert config_path.endswith("/config.json")
    assert config_path.startswith(fm.base_path)


def test_from_offloading_spec(tmp_path):
    fm = FileMapper.from_offloading_spec(
        root_dir=str(tmp_path),
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        gpu_blocks_per_file=1,
    )
    assert fm.base_path.startswith(str(tmp_path))
    key = make_offload_key(b"\x01" * 8, 0)
    assert fm.get_file_name(key).endswith(".bin")
