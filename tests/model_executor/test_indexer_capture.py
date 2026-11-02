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
    bind_indexer_capturer,
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
    snapshots = [capture.snapshot(2) for _ in range(3)]
    shared.zero_()
    first(shared)
    shared.add_(20)
    second(shared)
    assert torch.equal(
        snapshots[0].rows, torch.tensor([[[3, 1], [13, 11]], [[4, 2], [14, 12]]])
    )
    assert capture.snapshot(2).rows.tolist() == [
        [[0, 0], [20, 20]],
        [[0, 0], [20, 20]],
    ]
    capture.close()


def test_snapshot_is_independent_of_the_capture_buffer(capture_config):
    source = CaptureSource(2)
    capture = IndexerCapturer(source, capture_config)
    capture.device_buffer.fill_(7)
    snapshot = capture.snapshot(len(capture.device_buffer))

    capture.device_buffer.fill_(123)

    assert snapshot.rows.tolist() == [[[7, 7]], [[7, 7]]]
    assert snapshot.rows.data_ptr() != capture.device_buffer.data_ptr()
    capture.close()


def test_snapshots_of_different_sizes_stay_independent(capture_config):
    capture = IndexerCapturer(CaptureSource(2), capture_config)
    capture.device_buffer[:2] = torch.arange(2, dtype=torch.int32)[:, None, None]
    small = capture.snapshot(1)
    large = capture.snapshot(2)

    capture.device_buffer.zero_()

    assert small.rows.shape == (1, 1, 2)
    assert large.rows.shape == (2, 1, 2)
    assert small.rows.tolist() == [[[0, 0]]]
    assert large.rows[:, 0, 0].tolist() == [0, 1]
    assert small.rows.data_ptr() != large.rows.data_ptr()
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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_profile_reserves_all_in_flight_indexer_snapshots(
    capture_config, dist_init, enabled, async_scheduling
):
    capture_config.aux_output_config.enable_return_indexer_topk = enabled
    capture_config.scheduler_config.async_scheduling = async_scheduling
    source = CaptureSource(2)
    model = torch.nn.ModuleList([source, CaptureSource(5)])
    connector = AuxOutputWorkerConnector(vllm_config=capture_config, model=model)
    try:
        snapshots = connector.allocate_profile_snapshots()
        if not enabled:
            assert snapshots == []
            return
        assert source.indexer_capture is not None
        capture_rows = source.indexer_capture.rows
        expected_batches = capture_config.max_concurrent_batches + int(async_scheduling)
        assert len(snapshots) == expected_batches
        assert sum(t.numel() * t.element_size() for t in snapshots) == (
            expected_batches
            * len(model)
            * capture_rows.numel()
            * capture_rows.element_size()
        )
        assert len({t.data_ptr() for t in snapshots}) == expected_batches
        for snapshot in snapshots:
            assert snapshot.shape == (len(capture_rows), len(model), source.topk)
            assert snapshot.dtype == capture_rows.dtype
            assert snapshot.device == capture_rows.device
            assert snapshot.data_ptr() != capture_rows.data_ptr()
    finally:
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
    capture = IndexerCapturer(model, capture_config)

    with pytest.raises(RuntimeError, match="sink rejected"):
        bind_indexer_capturer(model, capture)

    assert first.indexer_capture is None
    assert rejecting.indexer_capture is None
