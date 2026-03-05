# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP A2A communication backend (no GPU required).

Tests cover:
1. DCP A2A config validation (--dcp-comm-backend)
2. TPA config validation (--tensor-parallel-size-attention)
3. LSE-weighted combination correctness
"""

import math
from unittest.mock import patch

import pytest
import torch

from vllm.config.parallel import ParallelConfig


def _mock_gpu_count(n: int = 16):
    """Mock cuda_device_count_stateless to return n GPUs."""
    return patch("vllm.config.parallel.cuda_device_count_stateless", return_value=n)


class TestDCPCommBackendConfig:
    """Test --dcp-comm-backend config validation."""

    def test_default_is_ag_rs(self):
        """Default comm backend is ag_rs."""
        config = ParallelConfig()
        assert config.dcp_comm_backend == "ag_rs"

    def test_a2a_requires_dcp_greater_than_1(self):
        """A2A backend requires decode_context_parallel_size > 1."""
        with pytest.raises(
            ValueError, match="requires decode_context_parallel_size > 1"
        ):
            ParallelConfig(
                dcp_comm_backend="a2a",
                decode_context_parallel_size=1,
            )

    def test_a2a_with_dcp_valid(self):
        """A2A backend is valid when DCP > 1."""
        with _mock_gpu_count():
            config = ParallelConfig(
                dcp_comm_backend="a2a",
                tensor_parallel_size=8,
                decode_context_parallel_size=4,
            )
        assert config.dcp_comm_backend == "a2a"

    def test_invalid_backend_rejected(self):
        """Invalid backend values are rejected (pydantic literal_error)."""
        with pytest.raises((ValueError, Exception), match="ag_rs|a2a"):
            ParallelConfig(
                dcp_comm_backend="invalid",
            )

    def test_ag_rs_with_dcp_1_valid(self):
        """ag_rs backend is valid with DCP=1 (no DCP)."""
        config = ParallelConfig(
            dcp_comm_backend="ag_rs",
            decode_context_parallel_size=1,
        )
        assert config.dcp_comm_backend == "ag_rs"


class TestTPAConfig:
    """Test --tensor-parallel-size-attention config validation."""

    def test_default_is_none(self):
        """Default TPA is None (TPA = TP)."""
        config = ParallelConfig()
        assert config.tensor_parallel_size_attention is None
        assert config.tpa_size == config.tensor_parallel_size

    def test_tpa_equals_tp(self):
        """TPA = TP means no context parallelism."""
        with _mock_gpu_count():
            config = ParallelConfig(
                tensor_parallel_size=8,
                tensor_parallel_size_attention=8,
            )
        assert config.tpa_size == 8
        assert config.dcp_size == 1

    def test_tpa_with_matching_dcp(self):
        """TPA < TP with correct DCP = TP / TPA."""
        with _mock_gpu_count():
            config = ParallelConfig(
                tensor_parallel_size=16,
                tensor_parallel_size_attention=4,
                decode_context_parallel_size=4,
            )
        assert config.tpa_size == 4
        assert config.dcp_size == 4

    def test_tpa_must_divide_tp(self):
        """TPA must evenly divide TP."""
        with (
            _mock_gpu_count(),
            pytest.raises(ValueError, match="must be divisible by"),
        ):
            ParallelConfig(
                tensor_parallel_size=16,
                tensor_parallel_size_attention=3,
            )

    def test_tpa_without_dcp_rejected(self):
        """TPA < TP without matching DCP is rejected."""
        with (
            _mock_gpu_count(),
            pytest.raises(
                ValueError,
                match="decode_context_parallel_size must equal",
            ),
        ):
            ParallelConfig(
                tensor_parallel_size=16,
                tensor_parallel_size_attention=4,
            )

    def test_tpa_dcp_mismatch_rejected(self):
        """DCP must equal TP / TPA when TPA < TP."""
        with (
            _mock_gpu_count(),
            pytest.raises(
                ValueError,
                match="decode_context_parallel_size must equal",
            ),
        ):
            ParallelConfig(
                tensor_parallel_size=16,
                tensor_parallel_size_attention=4,
                decode_context_parallel_size=2,
            )

    def test_tpa_mla_mode(self):
        """TPA=1 with DCP=8 is valid for MLA."""
        with _mock_gpu_count():
            config = ParallelConfig(
                tensor_parallel_size=8,
                tensor_parallel_size_attention=1,
                decode_context_parallel_size=8,
            )
        assert config.tpa_size == 1
        assert config.dcp_size == 8


class TestLSEWeightedCombine:
    """Test LSE-weighted combination logic (CPU only, no GPU).

    The _lse_weighted_combine function is the reference implementation
    that verifies the Triton kernel's correctness. It computes:

        result[b,h,d] = sum_n(w_n * output_n[b,h,d])

    where w_n = softmax(lse_n) = exp(lse_n) / sum_k(exp(lse_k))
    """

    def test_importable(self):
        """Verify _lse_weighted_combine is importable."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        assert callable(_lse_weighted_combine)

    def test_single_rank(self):
        """Single rank: output unchanged."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        # N=1, B=2, H=4, D=8
        outputs = torch.randn(1, 2, 4, 8)
        lses = torch.randn(1, 2, 4)

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (2, 4, 8)
        torch.testing.assert_close(result, outputs.squeeze(0), rtol=1e-5, atol=1e-5)

    def test_equal_lse(self):
        """Equal LSE values: outputs averaged equally."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        _N, B, H, D = 2, 1, 1, 4
        outputs = torch.tensor(
            [
                [[[1.0, 2.0, 3.0, 4.0]]],  # Rank 0
                [[[5.0, 6.0, 7.0, 8.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[0.0]],  # Rank 0
                [[0.0]],  # Rank 1
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        expected = (outputs[0] + outputs[1]) / 2
        assert result.shape == (B, H, D)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_dominant_rank(self):
        """Different LSE values: larger LSE gets more weight."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[0.0, 0.0]]],  # Rank 0
                [[[1.0, 1.0]]],  # Rank 1
            ]
        )
        lses = torch.tensor(
            [
                [[-100.0]],  # Rank 0: negligible contribution
                [[0.0]],  # Rank 1: dominant
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        assert result.shape == (B, H, D)
        # outputs[1] is [1, 1, 2] (N removed), result is [B, H, D] = [1, 1, 2]
        torch.testing.assert_close(result, outputs[1], atol=1e-5, rtol=1e-5)

    def test_mathematically_correct(self):
        """Verify mathematical correctness of LSE combination."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        outputs = torch.tensor(
            [
                [[[2.0, 4.0]]],
                [[[6.0, 8.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],  # exp(1) ≈ 2.718
                [[2.0]],  # exp(2) ≈ 7.389
            ]
        )

        result = _lse_weighted_combine(outputs, lses)

        w0 = math.exp(1) / (math.exp(1) + math.exp(2))
        w1 = math.exp(2) / (math.exp(1) + math.exp(2))
        expected = torch.tensor([[[w0 * 2.0 + w1 * 6.0, w0 * 4.0 + w1 * 8.0]]])

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_return_lse(self):
        """return_lse=True returns global LSE (logsumexp of inputs)."""
        from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

        B, H, D = 1, 1, 2
        outputs = torch.tensor(
            [
                [[[1.0, 2.0]]],
                [[[3.0, 4.0]]],
            ]
        )
        lses = torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        )

        result, global_lse = _lse_weighted_combine(outputs, lses, return_lse=True)

        expected_global_lse = math.log(math.exp(1) + math.exp(2))

        assert result.shape == (B, H, D)
        assert global_lse.shape == (B, H)
        assert abs(global_lse.item() - expected_global_lse) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
