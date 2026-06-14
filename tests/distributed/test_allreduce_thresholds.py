# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for allreduce backend dispatch thresholds.

Tests the dispatch guard logic for all NVIDIA-compatible allreduce backends:
- NCCL symmetric memory (should_nccl_symm_mem_allreduce)
- PyTorch symmetric memory (SymmMemCommunicator.should_use_symm_mem)
- Custom allreduce (CustomAllReduceCommunicator.should_custom_ar)

No GPUs required — runs with mocked dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
    KiB,
    MiB,
    should_nccl_symm_mem_allreduce,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor(nbytes: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Create a CPU tensor with a specific byte size."""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    assert nbytes % elem_size == 0, f"nbytes must be divisible by {elem_size}"
    return torch.empty(nbytes // elem_size, dtype=dtype, device="cpu")


def _make_symm_mem_comm(max_size: int):
    """Create a mock SymmMemCommunicator with the given max_size."""
    from vllm.distributed.device_communicators.symm_mem import (
        SymmMemCommunicator,
    )

    comm = MagicMock(spec=SymmMemCommunicator)
    comm.disabled = False
    comm.dtype = torch.bfloat16
    comm.max_size = max_size
    # Bind the real method to the mock instance
    comm.should_use_symm_mem = SymmMemCommunicator.should_use_symm_mem.__get__(comm)
    return comm


# ---------------------------------------------------------------------------
# NCCL Symmetric Memory: should_nccl_symm_mem_allreduce()
# ---------------------------------------------------------------------------


class TestNCCLSymmMemDispatch:
    """Test the donut-shaped activation pattern for NCCL symm mem.

    NCCL symm mem is preferred for small and large tensors, with custom_AR
    winning in a mid-range defined by custom_ar_preferred_ranges.
    """

    @pytest.fixture(autouse=True)
    def _enable_nccl_symm_mem(self):
        with (
            patch(
                "vllm.distributed.device_communicators.pynccl_allocator"
                ".is_symmetric_memory_enabled",
                return_value=True,
            ),
            patch(
                "vllm.distributed.device_communicators.all_reduce_utils"
                ".vllm_is_batch_invariant",
                return_value=False,
            ),
        ):
            yield

    # -- World size gating --

    @pytest.mark.parametrize("ws", [1, 2, 3])
    def test_below_min_world_size(self, ws):
        t = _make_tensor(1 * KiB)
        assert should_nccl_symm_mem_allreduce(ws, t) is False

    def test_at_min_world_size(self):
        min_ws = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["min_world_size"]
        t = _make_tensor(1 * KiB)
        assert should_nccl_symm_mem_allreduce(min_ws, t) is True

    # -- Donut boundaries at world_size=4 --
    # custom_AR preferred range: (16 KiB, 512 KiB)

    @pytest.mark.parametrize(
        "size,expected",
        [
            (8 * KiB, True),  # small: below lower bound
            (16 * KiB, True),  # at lower bound (<=)
            (16 * KiB + 2, False),  # just above lower bound
            (128 * KiB, False),  # mid-range: custom_AR preferred
            (512 * KiB - 2, False),  # just below upper bound
            (512 * KiB, True),  # at upper bound (>=)
            (1024 * KiB, True),  # large: above upper bound
        ],
    )
    def test_ws4_boundaries(self, size, expected):
        t = _make_tensor(size)
        assert should_nccl_symm_mem_allreduce(4, t) is expected

    # -- Donut boundaries at world_size=8 --
    # custom_AR preferred range: (16 KiB, 128 KiB)

    @pytest.mark.parametrize(
        "size,expected",
        [
            (8 * KiB, True),  # small: below lower bound
            (16 * KiB, True),  # at lower bound (<=)
            (16 * KiB + 2, False),  # just above lower bound
            (64 * KiB, False),  # mid-range: custom_AR preferred
            (128 * KiB - 2, False),  # just below upper bound
            (128 * KiB, True),  # at upper bound (>=)
            (1024 * KiB, True),  # large: above upper bound
        ],
    )
    def test_ws8_boundaries(self, size, expected):
        t = _make_tensor(size)
        assert should_nccl_symm_mem_allreduce(8, t) is expected

    # -- Above always_use_above_world_size --

    def test_above_always_use_world_size(self):
        above = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]
        t = _make_tensor(64 * KiB)
        assert should_nccl_symm_mem_allreduce(above + 1, t) is True

    # -- Disabled conditions --

    def test_disabled_when_not_enabled(self):
        with patch(
            "vllm.distributed.device_communicators.pynccl_allocator"
            ".is_symmetric_memory_enabled",
            return_value=False,
        ):
            t = _make_tensor(1024 * KiB)
            assert should_nccl_symm_mem_allreduce(4, t) is False

    def test_disabled_in_batch_invariant(self):
        with patch(
            "vllm.distributed.device_communicators.all_reduce_utils"
            ".vllm_is_batch_invariant",
            return_value=True,
        ):
            t = _make_tensor(1024 * KiB)
            assert should_nccl_symm_mem_allreduce(4, t) is False


# ---------------------------------------------------------------------------
# PyTorch Symmetric Memory: SymmMemCommunicator.should_use_symm_mem()
# ---------------------------------------------------------------------------


class TestSymmMemThresholds:
    """Test SYMM_MEM_ALL_REDUCE_MAX_SIZES thresholds per arch and world size.

    SymmMemCommunicator uses a max_size threshold: tensors below max_size
    use symm mem, tensors at or above fall through to PyNCCL.
    """

    @pytest.mark.parametrize(
        "arch_str,world_sizes",
        [
            ("9.0", [2, 4, 6, 8]),
            ("10.0", [2, 4, 6, 8]),
        ],
    )
    def test_max_size_boundaries(self, arch_str, world_sizes):
        """For each (arch, world_size), test at max_size boundary."""
        for ws in world_sizes:
            max_size = SYMM_MEM_ALL_REDUCE_MAX_SIZES[arch_str][ws]
            comm = _make_symm_mem_comm(max_size)

            # Just below max_size (4-byte aligned) → should use symm mem
            t_below = _make_tensor(max_size - 4)
            assert comm.should_use_symm_mem(t_below) is True, (
                f"arch={arch_str} ws={ws}: expected True for "
                f"size={max_size - 4} < max_size={max_size}"
            )

            # At max_size → should NOT use symm mem (strict <)
            t_at = _make_tensor(max_size)
            assert comm.should_use_symm_mem(t_at) is False, (
                f"arch={arch_str} ws={ws}: expected False for "
                f"size={max_size} >= max_size={max_size}"
            )

            # Above max_size → should NOT use symm mem
            t_above = _make_tensor(max_size + 4)
            assert comm.should_use_symm_mem(t_above) is False, (
                f"arch={arch_str} ws={ws}: expected False for "
                f"size={max_size + 4} > max_size={max_size}"
            )

    def test_dtype_rejection(self):
        """Only bf16 should be accepted."""
        comm = _make_symm_mem_comm(64 * MiB)

        # fp16 should be rejected
        t_fp16 = torch.empty(1024, dtype=torch.float16, device="cpu")
        assert comm.should_use_symm_mem(t_fp16) is False

        # fp32 should be rejected
        t_fp32 = torch.empty(1024, dtype=torch.float32, device="cpu")
        assert comm.should_use_symm_mem(t_fp32) is False

        # bf16 should be accepted
        t_bf16 = torch.empty(1024, dtype=torch.bfloat16, device="cpu")
        assert comm.should_use_symm_mem(t_bf16) is True

    def test_alignment_rejection(self):
        """Tensors not 4-byte aligned should be rejected."""
        comm = _make_symm_mem_comm(64 * MiB)

        # 1 element bf16 = 2 bytes, not 4-byte aligned
        t_unaligned = torch.empty(1, dtype=torch.bfloat16, device="cpu")
        assert comm.should_use_symm_mem(t_unaligned) is False

        # 2 elements bf16 = 4 bytes, 4-byte aligned
        t_aligned = torch.empty(2, dtype=torch.bfloat16, device="cpu")
        assert comm.should_use_symm_mem(t_aligned) is True

    def test_unsupported_arch_not_in_config(self):
        """Architectures not in SYMM_MEM_ALL_REDUCE_MAX_SIZES should not
        appear as keys. Verify the config only has expected entries."""
        supported_archs = set(SYMM_MEM_ALL_REDUCE_MAX_SIZES.keys())
        assert "8.0" not in supported_archs  # A100
        assert "8.9" not in supported_archs  # L4
        assert "9.0" in supported_archs  # H100
        assert "10.0" in supported_archs  # B200


# ---------------------------------------------------------------------------
# Custom AllReduce: max_size thresholds per arch
# ---------------------------------------------------------------------------


class TestCustomAllReduceThresholds:
    """Test CUSTOM_ALL_REDUCE_MAX_SIZES thresholds per arch and world size.

    When symm_mem is enabled, custom_AR uses arch-specific max_size thresholds
    that are tighter than the default 8 MiB.
    """

    @pytest.mark.parametrize(
        "arch_str,world_sizes",
        [
            ("9.0", [2, 4, 6, 8]),
            ("10.0", [2, 4, 6, 8]),
        ],
    )
    def test_max_sizes_are_consistent(self, arch_str, world_sizes):
        """Verify custom_AR max_sizes exist for all expected arch/ws combos."""
        for ws in world_sizes:
            assert ws in CUSTOM_ALL_REDUCE_MAX_SIZES[arch_str], (
                f"Missing custom_AR max_size for arch={arch_str} ws={ws}"
            )
            max_size = CUSTOM_ALL_REDUCE_MAX_SIZES[arch_str][ws]
            assert max_size > 0, (
                f"custom_AR max_size should be positive for arch={arch_str} ws={ws}"
            )

    def test_custom_ar_thresholds_h100(self):
        """Verify H100 custom_AR thresholds match expected values."""
        h100 = CUSTOM_ALL_REDUCE_MAX_SIZES["9.0"]
        assert h100[2] == 64 * MiB
        assert h100[4] == 32 * MiB
        assert h100[6] == MiB // 2  # 512 KiB
        assert h100[8] == MiB // 4  # 256 KiB

    def test_custom_ar_thresholds_b200(self):
        """Verify B200 custom_AR thresholds match expected values."""
        b200 = CUSTOM_ALL_REDUCE_MAX_SIZES["10.0"]
        assert b200[2] == 2 * MiB
        assert b200[4] == 2 * MiB
        assert b200[6] == 1 * MiB
        assert b200[8] == 1 * MiB

    def test_unsupported_arch_not_in_config(self):
        """Architectures not supported should not be in config."""
        assert "8.0" not in CUSTOM_ALL_REDUCE_MAX_SIZES
        assert "8.9" not in CUSTOM_ALL_REDUCE_MAX_SIZES


# ---------------------------------------------------------------------------
# Cross-backend consistency: no overlap gaps
# ---------------------------------------------------------------------------


class TestCrossBackendConsistency:
    """Verify that NCCL symm_mem and custom_AR thresholds are consistent.

    The donut-shaped NCCL symm_mem config carves out ranges where custom_AR
    wins. These ranges should fall within the custom_AR max_size limits.
    """

    @pytest.mark.parametrize("ws", [4, 8])
    def test_nccl_custom_ar_range_within_custom_ar_max_size(self, ws):
        """The NCCL symm_mem 'custom_AR preferred' range should not exceed
        the custom_AR max_size for any architecture."""
        nccl_range = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["custom_ar_preferred_ranges"][ws]
        _, upper = nccl_range

        for arch_str, arch_sizes in CUSTOM_ALL_REDUCE_MAX_SIZES.items():
            if ws in arch_sizes:
                ca_max = arch_sizes[ws]
                assert upper <= ca_max, (
                    f"NCCL symm_mem defers to custom_AR up to {upper} bytes "
                    f"at ws={ws}, but custom_AR max_size for {arch_str} is "
                    f"only {ca_max} bytes"
                )
