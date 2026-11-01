# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.compilation.cuda_graph import (
    _capture_forward_context_tensors,
    _refresh_captured_forward_context_tensors,
)
from vllm.config import DeviceConfig, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context


@dataclass
class NestedMetadata:
    seq_lens: torch.Tensor
    block_tables: torch.Tensor


def _make_vllm_config() -> VllmConfig:
    return VllmConfig(device_config=DeviceConfig("cpu"))


def test_cudagraph_replay_refreshes_forward_context_tensors():
    vllm_config = _make_vllm_config()
    captured_metadata = {
        "layer.0": NestedMetadata(
            seq_lens=torch.tensor([1, 2], dtype=torch.int32),
            block_tables=torch.tensor([[1, 0], [2, 0]], dtype=torch.int32),
        )
    }
    captured_slot_mapping = {"layer.0": torch.tensor([10, 11], dtype=torch.int32)}
    captured_descriptor = BatchDescriptor(
        num_tokens=2,
        num_reqs=2,
        uniform=True,
    )

    with set_forward_context(
        captured_metadata,
        vllm_config,
        slot_mapping=captured_slot_mapping,
        batch_descriptor=captured_descriptor,
    ):
        captured = _capture_forward_context_tensors()

    live_metadata = {
        "layer.0": NestedMetadata(
            seq_lens=torch.tensor([7, 8], dtype=torch.int32),
            block_tables=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        )
    }
    live_slot_mapping = {"layer.0": torch.tensor([20, 21], dtype=torch.int32)}
    live_descriptor = BatchDescriptor(
        num_tokens=2,
        num_reqs=2,
        uniform=True,
    )

    with set_forward_context(
        live_metadata,
        vllm_config,
        slot_mapping=live_slot_mapping,
        batch_descriptor=live_descriptor,
    ):
        _refresh_captured_forward_context_tensors(captured)

    assert torch.equal(captured_metadata["layer.0"].seq_lens, torch.tensor([7, 8]))
    assert torch.equal(
        captured_metadata["layer.0"].block_tables,
        torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
    )
    assert torch.equal(captured_slot_mapping["layer.0"], torch.tensor([20, 21]))
