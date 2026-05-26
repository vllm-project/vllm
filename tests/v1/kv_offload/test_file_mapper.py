# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FileMapper."""

from unittest.mock import MagicMock

from vllm.v1.kv_offload.base import (
    OffloadingSpec,
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
_MOCK_OFFLOADING_SPEC.block_size_factor = 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_mapper_from_offloading_spec(**kwargs) -> FileMapper:
    """Helper to create FileMapper with customizable mock config."""
    # Create a copy of the mock config to avoid modifying the global one
    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config.model = kwargs.get("model_name", "test-model")
    mock_vllm_config.cache_config.block_size = kwargs.get("hash_block_size", 16)
    mock_vllm_config.cache_config.cache_dtype = (
        f"torch.{kwargs.get('dtype', 'float16')}"
    )
    mock_vllm_config.parallel_config.tensor_parallel_size = kwargs.get("tp_size", 1)
    mock_vllm_config.parallel_config.pipeline_parallel_size = kwargs.get("pp_size", 1)
    mock_vllm_config.parallel_config.prefill_context_parallel_size = kwargs.get(
        "pcp_size", 1
    )
    mock_vllm_config.parallel_config.decode_context_parallel_size = kwargs.get(
        "dcp_size", 1
    )
    mock_vllm_config.parallel_config.rank = kwargs.get("rank", 0)

    mock_kv_cache_config = MagicMock()
    mock_kv_cache_config.kv_cache_groups = []

    mock_offloading_spec = MagicMock(spec=OffloadingSpec)
    mock_offloading_spec.vllm_config = mock_vllm_config
    mock_offloading_spec.kv_cache_config = mock_kv_cache_config
    mock_offloading_spec.block_size_factor = kwargs.get("block_size_factor", 1)

    return FileMapper.from_offloading_spec(
        root_dir=kwargs.get("root_dir", "/tmp/cache"),
        offloading_spec=mock_offloading_spec,
        gpu_blocks_per_file=mock_offloading_spec.block_size_factor,
    )


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
    fm = make_mapper_from_offloading_spec(rank=rank)
    key = make_offload_key(block_hash, group_idx)
    path = fm.get_file_name(key)

    expected_path = (
        "/tmp/cache/test-model_588656ebcc66_r3/000/10_g2/0001020304050607.bin"
    )
    assert path == expected_path


def test_get_run_config_fields():
    fm = make_mapper_from_offloading_spec(
        model_name="my-model",
        dtype="bfloat16",
        tp_size=2,
    )
    cfg = fm.get_run_config()
    assert cfg == {
        "model_name": "my-model",
        "hash_block_size": 16,
        "gpu_blocks_per_file": 1,
        "tp_size": 2,
        "pp_size": 1,
        "pcp_size": 1,
        "dcp_size": 1,
        "dtype": "bfloat16",
        "kv_cache_groups": [],
        "inference_engine": "vllm",
    }


def test_get_config_file_path():
    fm = make_mapper_from_offloading_spec()
    config_path = fm.get_config_file_path()
    assert config_path == f"{fm.base_path}/config.json"
