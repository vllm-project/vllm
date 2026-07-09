# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FileMapper."""

from unittest.mock import MagicMock, patch

import torch

from vllm.config import KVTransferConfig, ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.file_mapper import FileMapper

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_mapper_from_offloading_spec(**kwargs) -> FileMapper:
    """Helper to create FileMapper with customizable mock config."""
    # Create a copy of the mock config to avoid modifying the global one
    mock_vllm_config = MagicMock()
    mock_vllm_config.model_config.model = kwargs.get("model_name", "test-model")
    mock_vllm_config.cache_config.block_size = kwargs.get("cache_block_size", 16)
    mock_vllm_config.cache_config.cache_dtype = (
        f"torch.{kwargs.get('dtype', 'float16')}"
    )
    mock_vllm_config.cache_config.enable_prefix_caching = True
    mock_vllm_config.cache_config.prefix_match_unit = None
    tp_size = kwargs.get("tp_size", 1)
    pp_size = kwargs.get("pp_size", 1)
    pcp_size = kwargs.get("pcp_size", 1)
    world_size = tp_size * pp_size * pcp_size
    with patch.object(current_platform, "device_count", return_value=world_size):
        mock_vllm_config.parallel_config = ParallelConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=kwargs.get("dcp_size", 1),
            rank=kwargs.get("rank", 0),
        )
    mock_vllm_config.use_v2_model_runner = kwargs.get("use_v2_model_runner", False)
    mock_vllm_config.kv_events_config = None

    kv_cache_groups = kwargs.get("kv_cache_groups", [])
    block_size_factor = kwargs.get("block_size_factor", 1)
    extra_config = {
        "spec_name": "CPUOffloadingSpec",
        "cpu_bytes_to_use": 1,
    }
    if block_size_factor != 1:
        gpu_block_sizes = {
            group.kv_cache_spec.block_size * kwargs.get("dcp_size", 1) * pcp_size
            for group in kv_cache_groups
        }
        assert len(gpu_block_sizes) == 1
        extra_config["block_size"] = gpu_block_sizes.pop() * block_size_factor
    mock_vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=kv_cache_groups,
    )
    offloading_config = build_offloading_config(mock_vllm_config, kv_cache_config)
    offloading_spec = OffloadingSpecFactory.create_spec(offloading_config)

    return FileMapper.from_offloading_spec(
        root_dir=kwargs.get("root_dir", "/tmp/cache"),
        offloading_spec=offloading_spec,
        gpu_blocks_per_file=offloading_spec.block_size_factor,
        parallel_agnostic=kwargs.get("parallel_agnostic", False),
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
        tp_size=4,
        pp_size=3,
        pcp_size=2,
        dcp_size=2,
        kv_cache_groups=[_full_attention_group()],
        block_size_factor=3,
    )
    cfg = fm.get_run_config()
    assert cfg == {
        "model_name": "my-model",
        "hash_block_size": 16,
        "gpu_blocks_per_file": 3,
        "tp_size": 4,
        "pp_size": 3,
        "pcp_size": 2,
        "dcp_size": 2,
        "dtype": "bfloat16",
        "kv_cache_groups": [
            {
                "block_size": 16,
                "layer_names": ["layer0"],
            }
        ],
        "inference_engine": "vllm",
    }


def test_get_config_file_path():
    fm = make_mapper_from_offloading_spec()
    config_path = fm.get_config_file_path()
    assert config_path == f"{fm.base_path}/config.json"


# ---------------------------------------------------------------------------
# parallel_agnostic: honored only for a single non-MLA full-attention group
# ---------------------------------------------------------------------------


def _full_attention_group(
    block_size: int = 16, layer_name: str = "layer0"
) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=[layer_name],
        kv_cache_spec=FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.float32,
        ),
    )


def _sliding_window_group() -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=["layer0"],
        kv_cache_spec=SlidingWindowSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.float32,
            sliding_window=128,
        ),
    )


def _mla_group(block_size: int = 16, layer_name: str = "layer0") -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=[layer_name],
        kv_cache_spec=MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.float32,
        ),
    )


def test_parallel_agnostic_enabled_for_single_full_attention():
    # tp/rank are collapsed out of the namespace so the cache is shared
    # across tensor-parallel sizes.
    fm = make_mapper_from_offloading_spec(
        tp_size=4,
        pp_size=3,
        pcp_size=2,
        dcp_size=2,
        rank=1,
        kv_cache_groups=[_full_attention_group()],
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 1
    assert fm.fields["pp_size"] == 1
    assert fm.fields["pcp_size"] == 1
    assert fm.fields["dcp_size"] == 1
    assert fm.rank == 0


def test_parallel_agnostic_disabled_for_multiple_groups():
    # More than one KV-cache group (hybrid model) => keep per-layout namespacing.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        kv_cache_groups=[_full_attention_group(), _full_attention_group()],
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2


def test_hybrid_file_identity_preserves_cache_block_size():
    fm = make_mapper_from_offloading_spec(
        cache_block_size=16,
        kv_cache_groups=[
            _full_attention_group(block_size=12, layer_name="full_layer"),
            _mla_group(block_size=16, layer_name="mla_layer"),
        ],
    )

    # FileMapper has historically used cache_config.block_size here, which can
    # differ from the resolved hash block size for heterogeneous groups.
    assert fm.fields["hash_block_size"] == 16
    assert fm.fields["kv_cache_groups"] == [
        {"block_size": 12, "layer_names": ["full_layer"]},
        {"block_size": 16, "layer_names": ["mla_layer"]},
    ]


def test_parallel_agnostic_disabled_for_non_full_attention():
    # Single group but not full attention (sliding window) => keep namespacing.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        kv_cache_groups=[_sliding_window_group()],
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2


def test_parallel_agnostic_excludes_mla():
    # MLA latent KV is replicated per rank, so its offloaded blocks are not
    # parallelism-invariant: the opt-in must not collapse tp/rank.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        kv_cache_groups=[_mla_group()],
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1


def test_parallel_agnostic_disabled_on_v2_model_runner():
    # V2's KV layout is not known to be parallelism-invariant: don't collapse.
    fm = make_mapper_from_offloading_spec(
        tp_size=2,
        rank=1,
        kv_cache_groups=[_full_attention_group()],
        use_v2_model_runner=True,
        parallel_agnostic=True,
    )
    assert fm.fields["tp_size"] == 2
    assert fm.rank == 1
