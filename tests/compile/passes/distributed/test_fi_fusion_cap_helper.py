# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the FlashInfer AR+RMSNorm fusion byte-cap clamp helper.

These tests are pure (no GPU required) — they exercise the table-driven
``effective_fi_allreduce_fusion_max_bytes`` against the (capability,
world_size) cells from ``CUSTOM_ALL_REDUCE_MAX_SIZES`` and
``NCCL_SYMM_MEM_ALL_REDUCE_CONFIG``.
"""

import pytest

from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
    effective_fi_allreduce_fusion_max_bytes,
)

KiB = 1024
MiB = 1024 * 1024


def test_sm90_ws8_multimem_clamps_to_256kb():
    # FI cap for sm90 ws8 is 0.5MB; multimem takes over at
    # CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"][8] = 256KB. Helper must clamp.
    base = MiB // 2  # 0.5 MB
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="9.0",
        world_size=8,
        base_cap_bytes=base,
        multimem_active=True,
        nccl_symm_active=False,
    )
    assert out == CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"][8] == MiB // 4


def test_sm90_ws8_no_fast_path_returns_base():
    # symm-mem off, NCCL-symm-mem off: no fast path active -> no clamp.
    base = MiB // 2
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="9.0",
        world_size=8,
        base_cap_bytes=base,
        multimem_active=False,
        nccl_symm_active=False,
    )
    assert out == base


def test_sm90_ws8_nccl_symm_clamps_to_128kib():
    # NCCL_SYMM_MEM_ALL_REDUCE_CONFIG: ws8 custom_AR range is (16K, 128K)
    # i.e. NCCL symm-mem takes over above 128K.
    base = MiB // 2
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="9.0",
        world_size=8,
        base_cap_bytes=base,
        multimem_active=False,
        nccl_symm_active=True,
    )
    expected = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["custom_ar_preferred_ranges"][8][1]
    assert out == expected == 128 * KiB


def test_sm90_ws8_both_active_takes_lower_threshold():
    # multimem -> 256KB, NCCL symm-mem -> 128KiB. min wins -> 128 KiB.
    base = MiB // 2
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="9.0",
        world_size=8,
        base_cap_bytes=base,
        multimem_active=True,
        nccl_symm_active=True,
    )
    assert out == 128 * KiB


def test_sm90_ws2_not_in_multimem_list_no_clamp():
    # sm90 ws2 is not in _WORLD_SIZES_MULTIMEM; caller passes
    # multimem_active=False. Result unchanged.
    base = 64 * MiB  # FI cap for sm90 ws2
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="9.0",
        world_size=2,
        base_cap_bytes=base,
        multimem_active=False,
        nccl_symm_active=False,
    )
    assert out == base


def test_sm103_ws8_custom_ge_fi_cap_is_noop():
    # sm10.3 ws8: CUSTOM_ALL_REDUCE_MAX_SIZES["10.3"][8] = 4 MB which is >=
    # the FI cap (2 MB). min(2MB, 4MB) = 2MB -> no-op.
    fi_cap_bytes = 2 * MiB
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="10.3",
        world_size=8,
        base_cap_bytes=fi_cap_bytes,
        multimem_active=True,
        nccl_symm_active=False,
    )
    assert out == fi_cap_bytes


def test_unknown_capability_returns_base():
    # Capability not in the constants table -> no clamp.
    base = MiB // 2
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="7.5",
        world_size=8,
        base_cap_bytes=base,
        multimem_active=True,
        nccl_symm_active=False,
    )
    assert out == base


def test_nccl_symm_above_always_use_world_size_clamps_to_zero():
    # ws > always_use_above_world_size (default 8): NCCL symm-mem wins for
    # all sizes -> fusion never wins -> cap = 0.
    base = MiB
    always_above = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str="9.0",
        world_size=always_above + 1,
        base_cap_bytes=base,
        multimem_active=False,
        nccl_symm_active=True,
    )
    assert out == 0


@pytest.mark.parametrize(
    ("capability_str", "world_size", "base_bytes", "expected"),
    [
        ("9.0", 8, MiB // 2, MiB // 4),  # 256 KB
        ("9.0", 4, 2 * MiB, 32 * MiB),  # CUSTOM 32MB > FI 2MB -> no-op (2MB)
        ("10.0", 8, MiB, MiB),  # CUSTOM 1MB == FI 1MB -> 1MB
    ],
)
def test_multimem_only_table(capability_str, world_size, base_bytes, expected):
    out = effective_fi_allreduce_fusion_max_bytes(
        capability_str=capability_str,
        world_size=world_size,
        base_cap_bytes=base_bytes,
        multimem_active=True,
        nccl_symm_active=False,
    )
    assert out == min(base_bytes, expected)
