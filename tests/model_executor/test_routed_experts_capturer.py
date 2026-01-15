# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.fused_moe import routed_experts_capturer as rec


def _make_model_config(
    num_layers: int = 2,
    num_experts_per_tok: int = 2,
    n_routed_experts: int = 64,
) -> SimpleNamespace:
    hf_config = SimpleNamespace(
        num_hidden_layers=num_layers,
        num_experts_per_tok=num_experts_per_tok,
        n_routed_experts=n_routed_experts,
    )
    return SimpleNamespace(hf_text_config=hf_config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_routed_experts_capturer_mmap_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rec, "_should_use_shared_memory", lambda size: False)
    monkeypatch.setattr(
        rec, "_LOCK_FILE_PREFIX", str(tmp_path / "vllm_routed_experts")
    )
    monkeypatch.setattr(
        rec, "_MMAP_FILE_PREFIX", str(tmp_path / "vllm_routed_experts_mmap")
    )
    monkeypatch.setattr(rec, "get_tensor_model_parallel_rank", lambda: 0)

    model_config = _make_model_config()
    instance_id = f"test-{uuid.uuid4().hex}"

    capturer = rec.RoutedExpertsCapturer.create()
    reader = rec.RoutedExpertsReader.create()
    try:
        max_num_batched_tokens = 4
        max_num_kv_tokens = 4
        capturer.init_buffer(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_kv_tokens=max_num_kv_tokens,
            model_config=model_config,
            instance_id=instance_id,
        )
        reader.attach_buffer(
            max_num_kv_tokens=max_num_kv_tokens,
            model_config=model_config,
            instance_id=instance_id,
        )

        topk_ids_layer0 = torch.tensor(
            [[1, 2], [3, 4], [5, 6]],
            device="cuda",
            dtype=torch.int32,
        )
        topk_ids_layer1 = torch.tensor(
            [[7, 8], [9, 10], [11, 12]],
            device="cuda",
            dtype=torch.int32,
        )
        capturer.capture(0, topk_ids_layer0)
        capturer.capture(1, topk_ids_layer1)

        indices = np.array([0, 5, 2], dtype=np.int64)
        capturer.save_captured_experts(indices)

        routed = reader.get_routed_experts(indices)
        assert routed.shape == (3, 2, 2)
        assert routed.dtype == np.uint16

        expected = np.stack(
            [
                topk_ids_layer0.cpu().numpy(),
                topk_ids_layer1.cpu().numpy(),
            ],
            axis=1,
        ).astype(np.uint16)
        assert np.array_equal(routed, expected)
    finally:
        capturer.cleanup()
        reader.cleanup()
        rec._global_experts_capturer = None
        rec._global_experts_reader = None
