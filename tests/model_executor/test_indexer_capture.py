# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for raw capture binding and snapshot ownership."""

import pytest
import torch
from transformers import LlamaConfig

from vllm.config import DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.layers.indexer_capture import (
    IndexerCapturer,
    IndexerCaptureSink,
    IndexerCaptureSource,
    bind_indexer_capturer,
)

pytestmark = pytest.mark.cpu_test


class CaptureSource(torch.nn.Module, IndexerCaptureSource):
    def __init__(self, layer_id: int, topk: int = 2, *, producer: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.topk = topk
        self.producer = producer

    def get_indexer_capture_metadata(self) -> tuple[int, int] | None:
        return (self.layer_id, self.topk) if self.producer else None

    def forward(self, indices: torch.Tensor) -> None:
        if self.indexer_capture is not None:
            self.indexer_capture.capture(indices)


@pytest.fixture
def capture_config(tmp_path):
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=8,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    ).save_pretrained(tmp_path)
    return VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float32",
            max_model_len=16,
        ),
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_model_len=16,
            is_encoder_decoder=False,
            max_num_batched_tokens=2,
            max_num_seqs=1,
            enable_chunked_prefill=True,
            async_scheduling=False,
        ),
    )


def test_raw_capture_preserves_selection_order_and_padding():
    consumed = torch.tensor([[8, 2, -1, 4], [0, -1, -1, -1]], dtype=torch.int32)
    sink = IndexerCaptureSink(torch.empty_like(consumed))
    sink.capture(consumed)
    assert torch.equal(sink.rows, consumed)


def test_binding_maps_sparse_layer_ids_before_shared_topk_is_overwritten(
    capture_config,
):
    first, second = CaptureSource(2), CaptureSource(5)
    consumer, mtp = CaptureSource(4, producer=False), CaptureSource(8, topk=7)
    model = torch.nn.ModuleList([second, mtp, consumer, first])
    capture = IndexerCapturer(model, capture_config)
    assert first.indexer_capture is None and second.indexer_capture is None
    bind_indexer_capturer(model, capture)
    assert capture.source_columns == {2: 0, 5: 1}
    assert consumer.indexer_capture is None and mtp.indexer_capture is None
    shared = torch.tensor([[3, -1], [4, 2]], dtype=torch.int32)
    first(shared)
    shared.add_(10)
    second(shared)
    shared.zero_()
    snapshot = capture.snapshot(2)
    assert torch.equal(
        snapshot.rows, torch.tensor([[[3, -1], [13, 9]], [[4, 2], [14, 12]]])
    )
    capture.close()


def test_capture_snapshots_survive_shared_buffer_reuse(capture_config):
    first, second = CaptureSource(2), CaptureSource(5)
    model = torch.nn.ModuleList([first, second])
    capture = IndexerCapturer(model, capture_config)
    bind_indexer_capturer(model, capture)
    shared = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    first(shared)
    shared.add_(10)
    second(shared)
    snapshots = [
        capture.snapshot(2) for _ in range(capture_config.max_concurrent_batches)
    ]
    shared.zero_()
    first(shared)
    shared.add_(20)
    second(shared)
    assert torch.equal(
        snapshots[0].rows, torch.tensor([[[3, 1], [13, 11]], [[4, 2], [14, 12]]])
    )
    with pytest.raises(AssertionError, match="not consumed"):
        capture.snapshot(2)
    capture.release_snapshot(snapshots[0].slot)
    next_snapshot = capture.snapshot(2)
    assert next_snapshot.slot == snapshots[0].slot
    assert torch.equal(
        next_snapshot.rows, torch.tensor([[[0, 0], [20, 20]], [[0, 0], [20, 20]]])
    )
    capture.close()


def test_close_unbinds_sources_and_preserves_later_binding(capture_config):
    source = CaptureSource(2)
    capture = IndexerCapturer(source, capture_config)
    bind_indexer_capturer(source, capture)
    with pytest.raises(ValueError, match="already bound"):
        bind_indexer_capturer(source, capture)
    capture.close()
    assert source.indexer_capture is None
    rebound = IndexerCapturer(source, capture_config)
    bind_indexer_capturer(source, rebound)
    assert rebound.namespace == capture.namespace
    capture.close()
    source(torch.tensor([[7, 1]], dtype=torch.int32))
    assert torch.equal(rebound.snapshot(1).rows, torch.tensor([[[7, 1]]]))
    rebound.close()
    rebound.close()
    assert source.indexer_capture is None


@pytest.mark.parametrize(
    "layers,topks", [((2, 2), (2, 2)), ((2, 5), (2, 3)), ((2,), (0,)), ((-1,), (2,))]
)
def test_capturer_rejects_invalid_source_layout(capture_config, layers, topks):
    model = torch.nn.ModuleList(
        [CaptureSource(layer, topk) for layer, topk in zip(layers, topks, strict=True)]
    )
    with pytest.raises(ValueError, match="unique layers and uniform positive topk"):
        IndexerCapturer(model, capture_config)
    assert all(source.indexer_capture is None for source in model)


def test_capturer_rejects_models_without_explicit_capture_support(capture_config):
    with pytest.raises(ValueError, match="No backbone indexer capture sources"):
        IndexerCapturer(torch.nn.Linear(2, 2), capture_config)
