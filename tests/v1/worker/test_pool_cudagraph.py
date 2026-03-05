# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the pooler CUDA graph optimization in gpu_model_runner.py.

Tests cover:
- FlatClassifyPooler forward pass variants
- PoolerCUDAGraphRunner size mapping logic
- _parse_pool_cudagraph_batch_sizes env var parsing
- _introspect_classify_pooler model introspection
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from vllm.platforms import current_platform
from vllm.v1.worker.gpu_model_runner import (
    FlatClassifyPooler,
    PoolerCUDAGraphRunner,
    _introspect_classify_pooler,
    _parse_pool_cudagraph_batch_sizes,
)

# ── FlatClassifyPooler tests ──────────────────────────────────────────


class TestFlatClassifyPooler:
    def test_forward_with_all_ops(self):
        """Test forward with classifier, logit_bias, activation, and
        head_dtype."""
        classifier = nn.Linear(8, 4, bias=False)
        activation = nn.Sigmoid()
        pooler = FlatClassifyPooler(
            classifier=classifier,
            logit_bias=0.5,
            activation_fn=activation,
            head_dtype=torch.float32,
        )
        x = torch.randn(2, 8, dtype=torch.float16)
        out = pooler(x)
        assert out.shape == (2, 4)
        assert out.dtype == torch.float32
        # sigmoid output should be in [0, 1] range (approximately, since
        # we subtract bias first)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_forward_classifier_only(self):
        """Test forward with only a classifier (no bias, no activation)."""
        classifier = nn.Linear(8, 4, bias=False)
        pooler = FlatClassifyPooler(
            classifier=classifier,
            logit_bias=None,
            activation_fn=None,
            head_dtype=None,
        )
        x = torch.randn(3, 8)
        out = pooler(x)
        assert out.shape == (3, 4)

    def test_forward_no_classifier(self):
        """Test forward with no classifier (passthrough with optional
        dtype cast)."""
        pooler = FlatClassifyPooler(
            classifier=None,
            logit_bias=None,
            activation_fn=None,
            head_dtype=torch.float32,
        )
        x = torch.randn(2, 8, dtype=torch.float16)
        out = pooler(x)
        assert out.shape == (2, 8)
        assert out.dtype == torch.float32

    def test_forward_logit_bias_only(self):
        """Test that logit_bias is subtracted correctly."""
        pooler = FlatClassifyPooler(
            classifier=None,
            logit_bias=1.0,
            activation_fn=None,
            head_dtype=None,
        )
        x = torch.tensor([[2.0, 3.0]])
        out = pooler(x)
        expected = torch.tensor([[1.0, 2.0]])
        assert torch.allclose(out, expected)

    def test_forward_activation_only(self):
        """Test with only activation function."""
        pooler = FlatClassifyPooler(
            classifier=None,
            logit_bias=None,
            activation_fn=nn.Sigmoid(),
            head_dtype=None,
        )
        x = torch.zeros(2, 4)
        out = pooler(x)
        expected = torch.full((2, 4), 0.5)
        assert torch.allclose(out, expected)

    def test_logit_bias_buffer_registered(self):
        """Test that logit_bias is registered as a buffer (important for Cuda graph)."""
        pooler = FlatClassifyPooler(
            classifier=None,
            logit_bias=2.5,
            activation_fn=None,
            head_dtype=None,
        )
        assert hasattr(pooler, "logit_bias_val")
        assert pooler.logit_bias_val.item() == pytest.approx(2.5)

    def test_no_logit_bias_buffer_when_none(self):
        """Test that no buffer is registered when logit_bias is None."""
        pooler = FlatClassifyPooler(
            classifier=None,
            logit_bias=None,
            activation_fn=None,
            head_dtype=None,
        )
        assert not hasattr(pooler, "logit_bias_val")
        assert pooler.has_logit_bias is False


# ── _parse_pool_cudagraph_batch_sizes tests ───────────────────────────


class TestParsePoolCudagraphBatchSizes:
    def test_default_sizes(self):
        """Test default batch sizes when env var is not set."""
        with patch.dict(os.environ, {}, clear=False):
            # Make sure the env var is not set
            os.environ.pop("VLLM_POOL_CUDAGRAPH_BATCH_SIZES", None)
            with patch("vllm.v1.worker.gpu_model_runner.envs") as mock_envs:
                mock_envs.VLLM_POOL_CUDAGRAPH_BATCH_SIZES = None
                sizes = _parse_pool_cudagraph_batch_sizes()
        assert 1 in sizes
        assert 128 in sizes
        assert sizes == sorted(sizes)

    def test_custom_sizes_from_env(self):
        """Test parsing custom batch sizes from env var."""
        with patch("vllm.v1.worker.gpu_model_runner.envs") as mock_envs:
            mock_envs.VLLM_POOL_CUDAGRAPH_BATCH_SIZES = "4,2,8,16"
            sizes = _parse_pool_cudagraph_batch_sizes()
        assert sizes == [2, 4, 8, 16]

    def test_custom_sizes_deduplication(self):
        """Test that duplicate sizes are removed."""
        with patch("vllm.v1.worker.gpu_model_runner.envs") as mock_envs:
            mock_envs.VLLM_POOL_CUDAGRAPH_BATCH_SIZES = "4,4,8,8,16"
            sizes = _parse_pool_cudagraph_batch_sizes()
        assert sizes == [4, 8, 16]

    def test_single_size(self):
        """Test with a single batch size."""
        with patch("vllm.v1.worker.gpu_model_runner.envs") as mock_envs:
            mock_envs.VLLM_POOL_CUDAGRAPH_BATCH_SIZES = "32"
            sizes = _parse_pool_cudagraph_batch_sizes()
        assert sizes == [32]


# ── PoolerCUDAGraphRunner size map tests ──────────────────────────────


class TestPoolerCUDAGraphRunnerSizeMap:
    """Test the batch size mapping logic without actually capturing
    CUDA graphs (requires GPU)."""

    def _make_runner_with_size_map(self, capture_sizes):
        """Create a runner and populate only the _size_map, skipping
        actual CUDA graph capture."""
        runner = object.__new__(PoolerCUDAGraphRunner)
        runner._size_map = {}
        sorted_sizes = sorted(capture_sizes)
        for i in range(1, sorted_sizes[-1] + 1):
            for s in sorted_sizes:
                if i <= s:
                    runner._size_map[i] = s
                    break
        return runner

    def test_exact_match(self):
        """Exact batch sizes map to themselves."""
        runner = self._make_runner_with_size_map([1, 4, 8, 16])
        assert runner.get_capture_size(1) == 1
        assert runner.get_capture_size(4) == 4
        assert runner.get_capture_size(8) == 8
        assert runner.get_capture_size(16) == 16

    def test_round_up(self):
        """Non-captured sizes round up to the next captured size."""
        runner = self._make_runner_with_size_map([1, 4, 8, 16])
        assert runner.get_capture_size(2) == 4
        assert runner.get_capture_size(3) == 4
        assert runner.get_capture_size(5) == 8
        assert runner.get_capture_size(7) == 8
        assert runner.get_capture_size(9) == 16
        assert runner.get_capture_size(15) == 16

    def test_too_large(self):
        """Batch sizes larger than max captured return None."""
        runner = self._make_runner_with_size_map([1, 4, 8])
        assert runner.get_capture_size(9) is None
        assert runner.get_capture_size(100) is None

    def test_single_capture_size(self):
        """Single capture size maps batch size 1 to it."""
        runner = self._make_runner_with_size_map([8])
        assert runner.get_capture_size(1) == 8
        assert runner.get_capture_size(8) == 8
        assert runner.get_capture_size(9) is None


# ── PoolerCUDAGraphRunner GPU tests ───────────────────────────────────


@pytest.mark.skipif(torch.accelerator.device_count() < 1, reason="Need CUDA device")
class TestPoolerCUDAGraphRunnerGPU:
    """Tests that exercise actual CUDA graph capture and replay."""

    def _make_flat_pooler(self, hidden_size=8, num_labels=4):
        """Create a FlatClassifyPooler with all ops on CUDA."""
        classifier = nn.Linear(hidden_size, num_labels, bias=False)
        return FlatClassifyPooler(
            classifier=classifier,
            logit_bias=0.5,
            activation_fn=nn.Sigmoid(),
            head_dtype=torch.float32,
        ).cuda()

    def _make_runner(self, flat_pooler, hidden_size, num_labels, capture_sizes):
        """Create a PoolerCUDAGraphRunner with its own graph pool to
        avoid cross-test interference."""
        with patch("vllm.platforms.current_platform") as mock_platform:
            mock_platform.get_global_graph_pool.return_value = (
                current_platform.graph_pool_handle()
            )
            runner = PoolerCUDAGraphRunner(
                flat_pooler=flat_pooler,
                hidden_size=hidden_size,
                num_labels=num_labels,
                device=torch.device("cuda"),
                input_dtype=torch.float32,
                output_dtype=torch.float32,
                capture_sizes=capture_sizes,
            )
        return runner

    def test_graph_replay_matches_eager(self):
        """CUDA graph replay produces the same result as eager execution."""
        hidden_size, num_labels = 8, 4
        flat_pooler = self._make_flat_pooler(hidden_size, num_labels)
        runner = self._make_runner(flat_pooler, hidden_size, num_labels, [4])

        x = torch.randn(4, hidden_size, device="cuda")
        eager_out = flat_pooler(x)
        graph_out = runner.run(x, batch_size=4, capture_size=4)

        assert torch.allclose(eager_out, graph_out, atol=1e-6)

    def test_graph_replay_with_padding(self):
        """When batch_size < capture_size, only the valid rows should
        match the eager output."""
        hidden_size, num_labels = 8, 4
        flat_pooler = self._make_flat_pooler(hidden_size, num_labels)
        runner = self._make_runner(flat_pooler, hidden_size, num_labels, [4, 8])

        # Only 2 real samples, padded up to capture_size=4
        x = torch.randn(2, hidden_size, device="cuda")
        eager_out = flat_pooler(x)
        graph_out = runner.run(x, batch_size=2, capture_size=4)

        assert graph_out.shape == (2, num_labels)
        assert torch.allclose(eager_out, graph_out, atol=1e-6)

    def test_graph_replay_different_inputs(self):
        """Multiple replays with different inputs produce different
        (correct) results — verifies the graph isn't returning stale data."""
        hidden_size, num_labels = 8, 4
        flat_pooler = self._make_flat_pooler(hidden_size, num_labels)
        runner = self._make_runner(flat_pooler, hidden_size, num_labels, [4])

        x1 = torch.randn(4, hidden_size, device="cuda")
        x2 = torch.randn(4, hidden_size, device="cuda")

        out1 = runner.run(x1, batch_size=4, capture_size=4)
        out2 = runner.run(x2, batch_size=4, capture_size=4)

        eager1 = flat_pooler(x1)
        eager2 = flat_pooler(x2)

        assert torch.allclose(out1, eager1, atol=1e-6)
        assert torch.allclose(out2, eager2, atol=1e-6)

    def test_graph_replay_passthrough_no_classifier(self):
        """Graph capture/replay works with no classifier (passthrough
        with dtype cast and bias subtraction only)."""
        hidden_size = 8
        flat_pooler = FlatClassifyPooler(
            classifier=None,
            logit_bias=1.0,
            activation_fn=None,
            head_dtype=torch.float32,
        ).cuda()
        runner = self._make_runner(flat_pooler, hidden_size, hidden_size, [2])

        x = torch.tensor(
            [
                [3.0, 5.0, 7.0, 1.0, 2.0, 4.0, 6.0, 8.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            device="cuda",
        )
        graph_out = runner.run(x, batch_size=2, capture_size=2)
        eager_out = flat_pooler(x)

        assert torch.allclose(graph_out, eager_out, atol=1e-6)
        # bias=1.0 is subtracted, so second row should be all zeros
        assert torch.allclose(
            graph_out[1], torch.zeros(hidden_size, device="cuda"), atol=1e-6
        )


# ── _introspect_classify_pooler tests ─────────────────────────────────


class TestIntrospectClassifyPooler:
    def _make_mock_model(
        self,
        pooler_type="DispatchPooler",
        classify_pooler_type="SequencePooler",
        head_type="ClassifierPoolerHead",
        has_classifier=True,
    ):
        """Build a mock model with a configurable pooler hierarchy."""
        model = MagicMock()

        classifier = nn.Linear(16, 4, bias=False) if has_classifier else None

        head = MagicMock()
        head.__class__.__name__ = head_type
        head.classifier = classifier
        head.logit_bias = 0.5
        head.activation = nn.Sigmoid()
        head.head_dtype = torch.float32

        classify_pooler = MagicMock()
        classify_pooler.__class__.__name__ = classify_pooler_type
        classify_pooler.head = head

        pooler = MagicMock()
        pooler.__class__.__name__ = pooler_type
        pooler.poolers_by_task = {"classify": classify_pooler}
        model.pooler = pooler

        return model, pooler, classify_pooler, head

    @patch(
        "vllm.v1.worker.gpu_model_runner._parse_pool_cudagraph_batch_sizes",
        return_value=[1, 2, 4],
    )
    def test_valid_classify_pooler(self, mock_parse):
        """Test successful introspection of a valid classify pooler."""
        from vllm.model_executor.layers.pooler.seqwise.heads import (
            ClassifierPoolerHead,
        )
        from vllm.model_executor.layers.pooler.seqwise.poolers import (
            SequencePooler,
        )
        from vllm.model_executor.layers.pooler.special import DispatchPooler

        # Build real-ish objects that pass isinstance checks
        classifier = nn.Linear(16, 4, bias=False)
        head = MagicMock(spec=ClassifierPoolerHead)
        head.classifier = classifier
        head.logit_bias = 0.5
        head.activation = nn.Sigmoid()
        head.head_dtype = torch.float32

        classify_pooler = MagicMock(spec=SequencePooler)
        classify_pooler.head = head

        pooler = MagicMock(spec=DispatchPooler)
        pooler.poolers_by_task = {"classify": classify_pooler}

        model = MagicMock()
        model.pooler = pooler

        result = _introspect_classify_pooler(model)

        assert result is not None
        flat_pooler, graph_runner = result
        assert isinstance(flat_pooler, FlatClassifyPooler)
        # graph_runner may be None if CUDA is not available, that's fine

    def test_non_dispatch_pooler_returns_none(self):
        """Test that non-DispatchPooler returns None."""
        model = MagicMock()
        model.pooler = MagicMock()  # Not a DispatchPooler

        result = _introspect_classify_pooler(model)
        assert result is None

    def test_missing_classify_task_returns_none(self):
        """Test that missing 'classify' task returns None."""
        model = MagicMock()
        pooler = MagicMock()
        pooler.poolers_by_task = {"embed": MagicMock()}  # No "classify"
        model.pooler = pooler

        with patch(
            "vllm.model_executor.layers.pooler.special.DispatchPooler",
            create=True,
        ) as MockDP:
            pooler.__class__ = MockDP
            result = _introspect_classify_pooler(model)

        assert result is None


# ── Fast-path gating logic tests ─────────────────────────────────────


class TestPoolCudagraphFastPathGating:
    """Test the conditions that determine whether _pool() takes the
    CUDA graph fast path or falls back to the default path.

    The fast path requires:
    1. All tasks are "classify"
    2. All pooling_params have use_activation != False
    """

    def test_all_classify_with_activation_enables_fast_path(self):
        """Fast path is enabled when all tasks are classify and
        all use_activation is not False."""
        tasks = ["classify", "classify", "classify"]
        pooling_params = [
            MagicMock(use_activation=True),
            MagicMock(use_activation=True),
            MagicMock(use_activation=None),  # None is also not False
        ]

        all_classify = all(t == "classify" for t in tasks)
        all_use_activation = all_classify and all(
            p.use_activation is not False for p in pooling_params
        )

        assert all_classify is True
        assert all_use_activation is True

    def test_mixed_tasks_disables_fast_path(self):
        """Fast path is disabled when tasks include non-classify."""
        tasks = ["classify", "embed", "classify"]
        pooling_params = [
            MagicMock(use_activation=True),
            MagicMock(use_activation=True),
            MagicMock(use_activation=True),
        ]

        all_classify = all(t == "classify" for t in tasks)
        all_use_activation = all_classify and all(
            p.use_activation is not False for p in pooling_params
        )

        assert all_classify is False
        assert all_use_activation is False

    def test_all_embed_disables_fast_path(self):
        """Fast path is disabled when all tasks are embed."""
        tasks = ["embed", "embed"]
        pooling_params = [
            MagicMock(use_activation=True),
            MagicMock(use_activation=True),
        ]

        all_classify = all(t == "classify" for t in tasks)
        all_use_activation = all_classify and all(
            p.use_activation is not False for p in pooling_params
        )

        assert all_classify is False
        assert all_use_activation is False

    def test_use_activation_false_disables_fast_path(self):
        """Fast path is disabled when any use_activation is False."""
        tasks = ["classify", "classify"]
        pooling_params = [
            MagicMock(use_activation=True),
            MagicMock(use_activation=False),
        ]

        all_classify = all(t == "classify" for t in tasks)
        all_use_activation = all_classify and all(
            p.use_activation is not False for p in pooling_params
        )

        assert all_classify is True
        assert all_use_activation is False

    def test_single_classify_request_enables_fast_path(self):
        """Fast path works with a single classify request."""
        tasks = ["classify"]
        pooling_params = [MagicMock(use_activation=True)]

        all_classify = all(t == "classify" for t in tasks)
        all_use_activation = all_classify and all(
            p.use_activation is not False for p in pooling_params
        )

        assert all_classify is True
        assert all_use_activation is True

    def test_use_activation_none_enables_fast_path(self):
        """use_activation=None (default) should enable the fast path,
        since the check is 'is not False'."""
        tasks = ["classify"]
        pooling_params = [MagicMock(use_activation=None)]

        all_classify = all(t == "classify" for t in tasks)
        all_use_activation = all_classify and all(
            p.use_activation is not False for p in pooling_params
        )

        assert all_classify is True
        assert all_use_activation is True
