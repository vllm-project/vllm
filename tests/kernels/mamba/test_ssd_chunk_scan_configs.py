# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Mamba2 SSD chunk scan autotune configuration pruning (#61603)."""

from unittest import mock

from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import (
    _CHUNK_SCAN_FWD_CONFIGS,
    _get_chunk_scan_fwd_configs,
    _prune_configs,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton


def _is_broken_predicate(cfg: triton.Config) -> bool:
    return (
        cfg.kwargs.get("BLOCK_SIZE_M", 0) >= 64
        and cfg.kwargs.get("BLOCK_SIZE_N", 0) >= 64
        and cfg.num_warps == 4
        and cfg.num_stages >= 2
    )


def test_chunk_scan_total_configs():
    """Verify total number of defined configurations in ssd_chunk_scan."""
    assert len(_CHUNK_SCAN_FWD_CONFIGS) == 23


def test_chunk_scan_sm90_pruning():
    """On SM90 (Hopper), the 8 configs triggering Triton software pipelining
    codegen defects must be pruned, leaving 15 clean configs.
    """
    with (
        mock.patch.object(
            current_platform,
            "is_cuda",
            return_value=True,
        ),
        mock.patch.object(
            current_platform,
            "is_device_capability_family",
            side_effect=lambda cap: cap == 90,
        ),
    ):
        pruned = _prune_configs(_CHUNK_SCAN_FWD_CONFIGS)
        assert len(pruned) == 15

        # None of the pruned configs should match the broken predicate
        for cfg in pruned:
            assert not _is_broken_predicate(cfg), f"Broken config not pruned: {cfg}"

        # Verify key clean categories are all preserved
        stages_1 = [c for c in pruned if c.num_stages == 1]
        assert len(stages_1) == 4

        warps_8 = [c for c in pruned if c.num_warps == 8]
        assert len(warps_8) == 4

        warps_2 = [c for c in pruned if c.num_warps == 2]
        assert len(warps_2) == 3

        small_tiles = [
            c
            for c in pruned
            if (
                c.num_warps == 4
                and (
                    c.kwargs.get("BLOCK_SIZE_M", 0) < 64
                    or c.kwargs.get("BLOCK_SIZE_N", 0) < 64
                )
                and c.num_stages >= 2
            )
        ]
        assert len(small_tiles) == 4


def test_chunk_scan_non_sm90_preserves_all_configs():
    """Architectures other than SM90 (e.g. SM80, SM89, SM100) must retain all
    23 configurations.
    """
    for family in (80, 100, 120):
        with (
            mock.patch.object(
                current_platform,
                "is_cuda",
                return_value=True,
            ),
            mock.patch.object(
                current_platform,
                "is_device_capability_family",
                side_effect=lambda cap, f=family: cap == f,
            ),
        ):
            configs = _prune_configs(_CHUNK_SCAN_FWD_CONFIGS)
            assert len(configs) == 23


def test_chunk_scan_non_cuda_preserves_all_configs():
    """Non-CUDA platforms (CPU, ROCm, XPU) must retain all configs."""
    with (
        mock.patch.object(
            current_platform,
            "is_cuda",
            return_value=False,
        ),
        mock.patch.object(
            current_platform,
            "is_device_capability_family",
            return_value=False,
        ),
    ):
        configs = _prune_configs(_CHUNK_SCAN_FWD_CONFIGS)
        assert len(configs) == 23


def test_chunk_scan_empty_prune_fallback():
    """If all configs matched the predicate, fallback should return original configs."""
    broken_only = [
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=2,
            num_warps=4,
        )
    ]
    with (
        mock.patch.object(
            current_platform,
            "is_cuda",
            return_value=True,
        ),
        mock.patch.object(
            current_platform,
            "is_device_capability_family",
            side_effect=lambda cap: cap == 90,
        ),
    ):
        configs = _prune_configs(broken_only)
        assert configs == broken_only


def test_get_chunk_scan_fwd_configs_sm90():
    """_get_chunk_scan_fwd_configs helper properly delegates to _prune_configs."""
    with (
        mock.patch.object(
            current_platform,
            "is_cuda",
            return_value=True,
        ),
        mock.patch.object(
            current_platform,
            "is_device_capability_family",
            side_effect=lambda cap: cap == 90,
        ),
    ):
        configs = _get_chunk_scan_fwd_configs()
        assert len(configs) == 15
