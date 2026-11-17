# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for raw capture binding and snapshot ownership."""

import pytest
import torch
from transformers import LlamaConfig

from vllm.config import DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector
from vllm.model_executor.layers.indexer_capture import (
    IndexerCapturer,
    IndexerCaptureSink,
    IndexerCaptureSource,
    capture_indexer_topk,
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
    assert consumer.indexer_capture is None and mtp.indexer_capture is None
    shared = torch.tensor([[3, -1], [4, 2]], dtype=torch.int32)
    first(shared)
    shared.add_(10)
    second(shared)
    shared.zero_()
    snapshot = capture.snapshot(2)
    assert torch.equal(snapshot, torch.tensor([[[3, -1], [13, 9]], [[4, 2], [14, 12]]]))
    capture.close()


@pytest.mark.parametrize("num_rows", [0, 1, 2])
def test_snapshots_survive_capture_buffer_reuse(capture_config, num_rows):
    source = CaptureSource(2)
    capture = IndexerCapturer(source, capture_config)
    rows = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    source(rows)
    snapshot = capture.snapshot(num_rows)
    source(rows + 10)
    next_snapshot = capture.snapshot(2)

    assert snapshot.shape == (num_rows, 1, 2)
    torch.testing.assert_close(snapshot[:, 0], rows[:num_rows])
    torch.testing.assert_close(next_snapshot[:, 0], rows + 10)
    capture.close()


def test_close_unbinds_sources_and_preserves_later_binding(capture_config):
    source = CaptureSource(2)
    capture = IndexerCapturer(source, capture_config)
    with pytest.raises(ValueError, match="already bound"):
        IndexerCapturer(source, capture_config)
    capture.close()
    assert source.indexer_capture is None
    rebound = IndexerCapturer(source, capture_config)
    assert rebound.namespace == capture.namespace
    capture.close()
    source(torch.tensor([[7, 1]], dtype=torch.int32))
    assert torch.equal(rebound.snapshot(1), torch.tensor([[[7, 1]]]))
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


def test_connector_binds_indexer_at_construction_before_storage(
    capture_config, dist_init
):
    """Early capture works without SHM and connector close removes its sink."""
    capture_config.aux_output_config.enable_return_indexer_topk = True
    source = CaptureSource(2)
    assert source.indexer_capture is None
    connector = AuxOutputWorkerConnector(vllm_config=capture_config, model=source)
    try:
        rows = torch.tensor([[7, 1], [3, -1]], dtype=torch.int32)
        source(rows)
        assert source.indexer_capture is not None
        assert torch.equal(source.indexer_capture.rows[:2], rows)
    finally:
        connector.close()
    assert source.indexer_capture is None
    connector.close()


def test_rocm_monokernel_rejects_indexer_replay_before_loading_layers(capture_config):
    from vllm.models.deepseek_v4.amd.mono_decode import MonoDecodeLayer

    capture_config.aux_output_config.enable_return_indexer_topk = True
    capture_config.kernel_config.monokernel = True
    with pytest.raises(ValueError, match="disable --monokernel to use replay"):
        MonoDecodeLayer.create(None, capture_config)


def test_capture_helper_skips_unbound_sink_without_a_buffer():
    capture_indexer_topk(None, None, 2)


@pytest.mark.parametrize("num_tokens", [1, 3])
def test_capture_helper_copies_active_rows_and_topk_width(num_tokens):
    buffer = torch.tensor(
        [[8, -1, 30, 31], [2, 0, 32, 33], [-1, 4, 34, 35]], dtype=torch.int32
    )
    destination = torch.full((3, 2), -99, dtype=torch.int32)
    sink = IndexerCaptureSink(destination)
    capture_indexer_topk(sink, buffer, num_tokens)
    expected = torch.tensor([[8, -1], [2, 0], [-1, 4]], dtype=torch.int32)
    torch.testing.assert_close(destination[:num_tokens], expected[:num_tokens])
    torch.testing.assert_close(
        destination[num_tokens:], torch.full_like(destination[num_tokens:], -99)
    )


def test_capture_helper_preserves_sink_width_validation():
    sink = IndexerCaptureSink(torch.empty((2, 3), dtype=torch.int32))
    with pytest.raises(ValueError, match="selection width changed"):
        capture_indexer_topk(sink, torch.zeros((2, 2), dtype=torch.int32), 2)


def test_capture_helper_requires_a_buffer_for_a_bound_sink():
    sink = IndexerCaptureSink(torch.empty((2, 2), dtype=torch.int32))
    with pytest.raises(AssertionError):
        capture_indexer_topk(sink, None, 2)


class _RejectingSource(CaptureSource):
    def set_indexer_capture_sink(self, sink):
        if sink is not None:
            raise RuntimeError("sink rejected")
        super().set_indexer_capture_sink(sink)


def test_bind_rolls_back_sinks_installed_before_a_failure(capture_config):
    first = CaptureSource(2)
    rejecting = _RejectingSource(5)
    model = torch.nn.ModuleList([first, rejecting])
    with pytest.raises(RuntimeError, match="sink rejected"):
        IndexerCapturer(model, capture_config)

    assert first.indexer_capture is None
    assert rejecting.indexer_capture is None


def test_capture_helper_writes_at_the_dbo_ubatch_offset(monkeypatch):
    import vllm.model_executor.layers.indexer_capture as indexer_capture

    rows = torch.zeros((8, 2), dtype=torch.int32)
    sink = IndexerCaptureSink(rows)
    indices = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    monkeypatch.setattr(indexer_capture, "dbo_current_ubatch_token_offset", lambda: 5)

    capture_indexer_topk(sink, indices, 3)

    assert rows[5:8].tolist() == indices.tolist()
    assert not rows[:5].any() and not rows[8:].any()


def test_capture_helper_rejects_a_ubatch_offset_beyond_capacity(monkeypatch):
    import vllm.model_executor.layers.indexer_capture as indexer_capture

    sink = IndexerCaptureSink(torch.zeros((4, 2), dtype=torch.int32))
    indices = torch.ones((2, 2), dtype=torch.int32)
    monkeypatch.setattr(indexer_capture, "dbo_current_ubatch_token_offset", lambda: 3)

    with pytest.raises(ValueError, match="exceeds capture token capacity"):
        capture_indexer_topk(sink, indices, 2)
