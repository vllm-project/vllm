# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU oracles for the selections consumed by sparse attention adapters."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.distributed.aux_output_connector.profile import (
    AuxOutputProfile,
    IndexerSource,
    decode_indexer,
    validate_shm_budget,
)
from vllm.model_executor.layers.indexer_capture import (
    IndexerCapturer,
    IndexerCaptureSink,
)

pytestmark = pytest.mark.cpu_test


def _sink(source, tokens=3):
    return IndexerCaptureSink(
        source,
        torch.empty((tokens, source.width), dtype=torch.int32),
        torch.zeros(tokens, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    ("semantic", "ratio", "consumers"),
    [
        pytest.param("logical_token_id", 1, (0, 1), id="deepseek-v32-indexcache"),
        pytest.param("logical_token_id", 1, (0,), id="glm-52"),
        pytest.param("compressed_cache_index", 4, (0,), id="deepseek-v4"),
        pytest.param("compressed_cache_index", 2, (0, 2, 3), id="deepseek-v41"),
        pytest.param("logical_token_id", 1, (0,), id="glm-53-no-pool"),
    ],
)
def test_raw_selection_preserves_order_padding_and_source_mapping(
    semantic, ratio, consumers
):
    source = IndexerSource(0, (4,), semantic, ratio=ratio, consumer_layer_ids=consumers)
    consumed = torch.tensor([[8, 2, -1, 4], [0, -1, -1, -1]], dtype=torch.int32)
    sink = _sink(source, 2)
    sink.capture(consumed)
    np.testing.assert_array_equal(
        decode_indexer(sink.rows.numpy(), AuxOutputProfile("weights", (source,))),
        consumed.numpy()[:, None],
    )
    assert sink.valid.all()


def test_pool_encoding_preserves_short_causal_rows_and_unsorted_pools():
    source = IndexerSource(2, (5,), "pool_id_with_validity", "pool_mask", 4, 8)
    sink = _sink(source)
    consumed = torch.tensor(
        [
            [20, 21, 22, 23, 4, 5, 6, 7, 32, 33, -1],
            [0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    sink.capture_pools(consumed)
    profile = AuxOutputProfile("glm-pool", (source,))
    np.testing.assert_array_equal(
        decode_indexer(sink.rows.numpy(), profile), consumed.numpy()[:, None]
    )
    assert profile.fixed_bytes_per_token == 20


def test_qsa_blocks_restore_causal_tail_and_trailing_count():
    source = IndexerSource(
        1,
        (4,),
        "compressed_block_id",
        "qsa_blocks",
        4,
        8,
        trailing_count=True,
    )
    sink = _sink(source)
    sink.capture_qsa(
        torch.tensor([[3, 0], [0, -1], [-1, -1]], dtype=torch.int32),
        torch.tensor([18, 5, 1]),
        torch.tensor([4, 1, 0]),
    )
    expected = np.array(
        [
            [12, 13, 14, 15, 0, 1, 2, 3, 16, 17, 18, 11],
            [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, 6],
            [0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(
        decode_indexer(sink.rows.numpy(), AuxOutputProfile("qsa", (source,))),
        expected[:, None],
    )


def test_step5_prefill_groups_and_decode_flat_rows_have_identical_layout():
    source = IndexerSource(3, (2, 5), "region_id", ratio=8, selection_width=True)
    prefill, decode = _sink(source, 2), _sink(source, 2)
    for sink in (prefill, decode):
        sink.region_local = torch.empty((2, 2, 5), dtype=torch.int32)
        sink.region_replicas = 1
    consumed = torch.tensor(
        [
            [[2, 0, 3], [1, -1, 3]],
            [[0, 3, 4], [-1, 2, 4]],
        ]
    )
    prefill.capture_regions(consumed[:, 0], group=0)
    assert not prefill.valid.any()
    prefill.capture_regions(consumed[:, 1], group=1)
    decode.capture_regions(consumed.reshape(4, 3))
    assert torch.equal(prefill.rows, decode.rows)
    restored = decode_indexer(
        decode.rows.numpy(), AuxOutputProfile("step5", (source,))
    )[:, 0]
    np.testing.assert_array_equal(restored[..., :3], consumed.numpy())
    assert (restored[..., 3] == -1).all()
    assert (restored[..., 4] == 3).all()


def test_profile_identity_includes_codec_semantics_and_consumer_layers():
    source = IndexerSource(0, (4,), "logical_token_id", consumer_layer_ids=(0, 1))
    profile = AuxOutputProfile("weights-a", (source,))
    assert AuxOutputProfile.from_dict(profile.to_dict()) == profile
    for changed in (
        replace(profile, model_identity="weights-b"),
        replace(profile, sources=(replace(source, semantic="region_id"),)),
        replace(profile, sources=(replace(source, consumer_layer_ids=(0, 2)),)),
    ):
        assert profile.profile_id != changed.profile_id
    with pytest.raises(ValueError, match="version"):
        AuxOutputProfile.from_dict({**profile.to_dict(), "codec_version": 2})


def test_capture_snapshots_survive_shared_buffer_reuse(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.indexer_capture.current_platform",
        SimpleNamespace(device_type="cpu"),
    )

    class Source(torch.nn.Module):
        indexer_capture: IndexerCaptureSink | None = None

        def __init__(self, layer):
            super().__init__()
            self.indexer_capture_spec = IndexerSource(layer, (2,), "logical_token_id")

    first, second = Source(0), Source(1)
    model = torch.nn.ModuleList([first, second])
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2),
        max_concurrent_batches=2,
        model_config=SimpleNamespace(
            model="synthetic",
            revision="r1",
            code_revision=None,
            hf_config=SimpleNamespace(_commit_hash="sha"),
            hf_text_config=SimpleNamespace(to_dict=lambda: {"num_hidden_layers": 2}),
            dtype=torch.bfloat16,
            quantization=None,
        ),
    )
    capture = IndexerCapturer(model, config)
    shared = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    capture.begin_step()
    assert first.indexer_capture is not None
    first.indexer_capture.capture(shared)
    shared.add_(10)
    assert second.indexer_capture is not None
    second.indexer_capture.capture(shared)
    snapshot = capture.snapshot(2)
    validity = snapshot.valid
    capture.begin_step()
    shared.zero_()
    first.indexer_capture.capture(shared)
    assert validity.all()
    next_snapshot = capture.snapshot(2)
    assert not next_snapshot.valid[:, 1].any()
    with pytest.raises(AssertionError, match="not consumed"):
        capture.snapshot(2)
    capture.release_snapshot(next_snapshot.slot)
    assert capture.snapshot(2).slot == next_snapshot.slot
    assert torch.equal(snapshot.rows, torch.tensor([[3, 1, 13, 11], [4, 2, 14, 12]]))
    capture.close()
    assert first.indexer_capture is None and second.indexer_capture is None
    capture.close()


def test_budget_covers_every_kv_block_without_silent_truncation():
    assert (
        validate_shm_budget(
            num_blocks=10, scheduler_block_size=16, bytes_per_token=20, max_bytes=None
        )
        == 3200
    )
    with pytest.raises(ValueError, match="Maximum Supported Blocks: 9"):
        validate_shm_budget(
            num_blocks=10,
            scheduler_block_size=16,
            bytes_per_token=20,
            max_bytes=3199,
        )
