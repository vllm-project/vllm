# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bounded multi-thread safetensors weight loader."""

from unittest.mock import patch
import pytest
import safetensors.torch
import torch

from vllm.model_executor.model_loader.weight_utils import (
    multi_thread_safetensors_weights_iterator,
)


@pytest.fixture
def synthetic_safetensors_shards(tmp_path):
    """Create 8 synthetic safetensors shard files with mixed dense and expert weights."""
    shard_paths = []
    for i in range(8):
        file_path = str(tmp_path / f"model-{i:05d}-of-00008.safetensors")
        tensors = {
            f"dense_layer_{i}.weight": torch.randn(8, 8),
            f"model.layers.0.mlp.experts.{i % 4}.weight": torch.randn(8, 8),
            f"model.layers.0.mlp.experts.{i % 4}.scale": torch.tensor([1.0]),
        }
        safetensors.torch.save_file(tensors, file_path)
        shard_paths.append(file_path)
    return shard_paths


def test_multi_thread_safetensors_bounded_sliding_window(synthetic_safetensors_shards):
    """Verify that in-flight submitted tasks never exceed max_workers + 1."""
    max_workers = 2
    expected_max_buffer = max_workers + 1

    import concurrent.futures
    original_submit = concurrent.futures.ThreadPoolExecutor.submit
    max_observed_pending = 0
    currently_pending = 0

    def tracked_submit(self, fn, *args, **kwargs):
        nonlocal max_observed_pending, currently_pending
        currently_pending += 1
        if currently_pending > max_observed_pending:
            max_observed_pending = currently_pending
        future = original_submit(self, fn, *args, **kwargs)

        def on_done(fut):
            nonlocal currently_pending
            currently_pending -= 1

        future.add_done_callback(on_done)
        return future

    with patch.object(concurrent.futures.ThreadPoolExecutor, "submit", tracked_submit):
        iterator = multi_thread_safetensors_weights_iterator(
            synthetic_safetensors_shards,
            use_tqdm_on_load=False,
            max_workers=max_workers,
        )
        loaded = list(iterator)

    # 8 dense weights + 8 expert weights + 8 scale weights = 24 items
    assert len(loaded) == 24
    # Bounded sliding window invariant: pending futures must never exceed max_workers + 1
    assert max_observed_pending <= expected_max_buffer


def test_multi_thread_safetensors_local_expert_ids_filtering(synthetic_safetensors_shards):
    """Verify non-local expert weights are skipped while dense weights and scales are preserved."""
    # Only load expert 0
    iterator = multi_thread_safetensors_weights_iterator(
        synthetic_safetensors_shards,
        use_tqdm_on_load=False,
        max_workers=2,
        local_expert_ids=[0],
    )
    loaded_dict = dict(list(iterator))

    # All dense layers must be present
    for i in range(8):
        assert f"dense_layer_{i}.weight" in loaded_dict

    # Expert 0 weight must be present
    assert "model.layers.0.mlp.experts.0.weight" in loaded_dict

    # Non-local expert weights (1, 2, 3) must be filtered out
    assert "model.layers.0.mlp.experts.1.weight" not in loaded_dict
    assert "model.layers.0.mlp.experts.2.weight" not in loaded_dict
    assert "model.layers.0.mlp.experts.3.weight" not in loaded_dict

    # Expert scales for all experts should not be filtered out
    for exp_id in range(4):
        assert f"model.layers.0.mlp.experts.{exp_id}.scale" in loaded_dict
