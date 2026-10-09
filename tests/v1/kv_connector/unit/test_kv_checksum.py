# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side KV checksums match their definition (per-slice checksums
weighted by layer and head), computed independently here, whatever the TP or
PP sharding, replication, memory layout or kernel block size of the KV
cache."""

import copy
import threading

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_checksum import worker as checksum_worker
from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    MAX_KV_HEADS_PER_LAYER,
    KVChecksumBlocks,
    KVChecksumRecord,
    KVChecksumScheduled,
)
from vllm.distributed.kv_transfer.kv_checksum.worker import (
    KVChecksumOutput,
    KVChecksumWorker,
)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
HEAD_SIZE = 8  # (K + V) * bf16 = 32 bytes = 4 pieces per cell
CELL_BYTES = 4 * HEAD_SIZE
NUM_BLOCKS = 6
# Three blocks covering 40 tokens: the last block (id 4) holds tokens 32..39.
BLOCKS = KVChecksumBlocks(num_tokens=40, groups={0: (0, [3, 1, 4])})


def _name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.attn"


def _spec(heads: int, **kwargs) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=heads,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        **kwargs,
    )


def _config(*groups: KVCacheGroupSpec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=list(groups)
    )


def _group(spec, *layer_names: str, **kwargs) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(list(layer_names or (_name(0),)), spec, **kwargs)


def _worker(config, caches, *, tp_rank=0, tp_size=1, uniform_kv_heads=2):
    return KVChecksumWorker(
        config,
        caches,
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        uniform_kv_heads=uniform_kv_heads,
    )


def _logical_kv(heads: int, seed: int = 0) -> torch.Tensor:
    """Random logical KV bytes of one layer: [blocks, heads, tokens, cell]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        256,
        (NUM_BLOCKS, heads, BLOCK_SIZE, CELL_BYTES),
        generator=gen,
        dtype=torch.uint8,
    )


def _cache(logical: torch.Tensor, *, token_major=False, kernel_split=1):
    """A bf16 [B, H, N, C] cache view holding ``logical``, in a given layout."""
    blocks, heads, tokens, cell = logical.shape
    data = logical.view(blocks, heads, kernel_split, tokens // kernel_split, cell)
    data = data.permute(0, 2, 1, 3, 4).reshape(
        blocks * kernel_split, heads, tokens // kernel_split, cell
    )
    if token_major:
        storage = torch.empty(
            blocks * kernel_split,
            tokens // kernel_split,
            heads,
            cell,
            dtype=torch.uint8,
        )
        view = storage.permute(0, 2, 1, 3)
    else:
        view = torch.empty_like(data)
    view.copy_(data)
    return view.view(torch.bfloat16)


def _reference(
    layers: dict[int, torch.Tensor], blocks: KVChecksumBlocks = BLOCKS
) -> np.ndarray:
    """Checksums from the definition: each (layer, head) slice of a block is
    s = sum_j x_j * (2j + 1) over its 8-byte pieces in token order, and the
    block's checksum is sum m * s with m = 2 * (layer * H + head) + 1.

    ``layers`` maps layer numbers to the KV of all their heads.
    """
    start, block_ids = blocks.groups[0]
    sums = np.zeros(len(block_ids), dtype=np.uint64)
    for number, kv in layers.items():
        pieces = kv.contiguous().view(torch.int64).numpy().view(np.uint64)
        _, heads, tokens, num_pieces = pieces.shape
        head = np.arange(heads, dtype=np.uint64)[:, None, None]
        token = np.arange(tokens, dtype=np.uint64)[None, :, None]
        piece = np.arange(num_pieces, dtype=np.uint64)[None, None, :]
        slice_factor = np.uint64(2) * (
            np.uint64(number * MAX_KV_HEADS_PER_LAYER) + head
        ) + np.uint64(1)
        position = token * np.uint64(num_pieces) + piece
        weights = slice_factor * (np.uint64(2) * position + np.uint64(1))
        for row, block_id in enumerate(block_ids):
            block = pieces[block_id].copy()
            block[:, max(0, blocks.num_tokens - (start + row) * tokens) :, :] = 0
            with np.errstate(over="ignore"):
                sums[row] += (block * weights).sum(dtype=np.uint64)
    return sums


def _send_record(worker: KVChecksumWorker, blocks=BLOCKS) -> KVChecksumRecord:
    worker.bind(KVChecksumScheduled(reqs_to_send={"req": blocks}))
    [record] = worker.post_forward(None, set(), set()).finalize()
    return record


def _send(worker: KVChecksumWorker, blocks=BLOCKS):
    return _send_record(worker, blocks).checksums["req"]


def _recv(worker: KVChecksumWorker, blocks=BLOCKS):
    worker.bind(KVChecksumScheduled(reqs_to_recv={"req": blocks}))
    [record] = worker.post_forward({"req"}, set(), set()).finalize()
    return record.checksums["req"]


@pytest.mark.parametrize("token_major", [False, True])
def test_checksum_matches_definition(token_major):
    logical = _logical_kv(heads=2)
    worker = _worker(
        _config(_group(_spec(2))),
        {_name(0): _cache(logical, token_major=token_major)},
    )
    [group] = _send(worker).values()
    assert group.start == 0 and group.replica_keys == [(1, 0)]
    np.testing.assert_array_equal(group.values[:, 0], _reference({0: logical}))


@pytest.mark.parametrize("token_major", [False, True])
def test_tp_shards_sum_to_unsharded_checksum(token_major):
    """Each TP rank weighs its heads by their global index, so the shards'
    partial checksums add up to the unsharded definition."""
    logical = _logical_kv(heads=4)
    total = np.zeros(3, dtype=np.uint64)
    for rank in range(2):
        cache = _cache(logical[:, 2 * rank : 2 * rank + 2], token_major=token_major)
        worker = _worker(
            _config(_group(_spec(2))),
            {_name(0): cache},
            tp_rank=rank,
            tp_size=2,
            uniform_kv_heads=4,
        )
        [group] = _send(worker).values()
        total += group.values[:, 0]
    np.testing.assert_array_equal(total, _reference({0: logical}))


def test_pp_stages_sum_to_whole_model_checksum():
    """PP stages hold different layers; with model-wide layer numbers their
    partial checksums add up to the whole model's."""
    layers = {0: _logical_kv(heads=2, seed=1), 1: _logical_kv(heads=2, seed=2)}
    total = np.zeros(3, dtype=np.uint64)
    for number, kv in layers.items():
        worker = _worker(
            _config(_group(_spec(2), _name(number))), {_name(number): _cache(kv)}
        )
        total += _send(worker)[0].values[:, 0]
    np.testing.assert_array_equal(total, _reference(layers))


def test_group_fingerprints_sum_over_pp_stages():
    """A group's fingerprint summed over PP stages does not depend on how its
    layers are split, and changes with the layers it holds."""

    def fingerprint(*stages: tuple[int, ...]) -> int:
        total = 0
        for numbers in stages:
            names = [_name(n) for n in numbers]
            caches = {name: _cache(_logical_kv(heads=2)) for name in names}
            worker = _worker(_config(_group(_spec(2), *names)), caches)
            total += _send_record(worker).fingerprints[0]
        return total % 2**64

    assert fingerprint((0, 1)) == fingerprint((0,), (1,))
    assert fingerprint((0, 1)) != fingerprint((0, 2))


def test_gqa_replicas_are_checked_per_replica():
    """With more TP ranks than KV heads, consecutive ranks replicate a head.
    A producer sums only replica 0 (other replicas report the request with no
    checksums); a consumer reports each replica so every copy can be checked
    against the same unsharded value."""
    logical = _logical_kv(heads=2)
    expected = _reference({0: logical})
    workers = [
        _worker(
            _config(_group(_spec(1))),
            {_name(0): _cache(logical[:, rank // 2 : rank // 2 + 1])},
            tp_rank=rank,
            tp_size=4,
            uniform_kv_heads=2,
        )
        for rank in range(4)
    ]

    sent = [_send(worker) for worker in workers]
    assert sent[1] == sent[3] == {}
    assert sent[0][0].replica_keys == sent[2][0].replica_keys == [(2, 0)]
    np.testing.assert_array_equal(
        sent[0][0].values[:, 0] + sent[2][0].values[:, 0], expected
    )

    received = [_recv(worker)[0] for worker in workers]
    assert [c.replica_keys for c in received] == [
        [(2, 0)],
        [(2, 1)],
        [(2, 0)],
        [(2, 1)],
    ]
    for replica in range(2):
        np.testing.assert_array_equal(
            received[replica].values[:, 0] + received[replica + 2].values[:, 0],
            expected,
        )


def test_mla_replicas_are_checked_per_replica():
    """A cache with one TP shard (MLA) is replicated on every rank."""
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=2 * HEAD_SIZE,
        head_size_v=0,
        dtype=torch.bfloat16,
        max_tp_shards=1,
    )
    logical = _logical_kv(heads=1)
    workers = [
        _worker(
            _config(_group(spec)), {_name(0): _cache(logical)}, tp_rank=r, tp_size=2
        )
        for r in range(2)
    ]
    assert _send(workers[1]) == {}
    [group] = _send(workers[0]).values()
    assert group.replica_keys == [(2, 0)]
    for rank, worker in enumerate(workers):
        [received] = _recv(worker).values()
        assert received.replica_keys == [(2, rank)]
        np.testing.assert_array_equal(received.values[:, 0], _reference({0: logical}))


def test_misplaced_tokens_and_layers_change_the_checksum():
    """The same bytes in the wrong place (two tokens or two layers swapped)
    give a different checksum."""
    first, second = _logical_kv(heads=2, seed=1), _logical_kv(heads=2, seed=2)
    group = _group(_spec(2), _name(0), _name(1))

    def checksum(layer0, layer1):
        caches = {_name(0): _cache(layer0), _name(1): _cache(layer1)}
        return _send(_worker(_config(group), caches))[0].values[:, 0]

    baseline = checksum(first, second)
    swapped_tokens = first.clone()
    swapped_tokens[3, :, [0, 1]] = first[3, :, [1, 0]]
    assert checksum(swapped_tokens, second)[0] != baseline[0]
    assert not np.array_equal(checksum(second, first), baseline)


def test_slots_past_num_tokens_are_ignored():
    """Writes past num_tokens in the last block (decode or draft tokens) do not
    change the checksum; writes to a valid slot do."""
    cache = _cache(_logical_kv(heads=2))
    worker = _worker(_config(_group(_spec(2))), {_name(0): cache})
    before = _send(worker)[0].values.copy()

    cache.view(torch.uint8)[4, :, 8:] ^= 0xFF
    np.testing.assert_array_equal(_send(worker)[0].values, before)

    cache.view(torch.uint8)[4, 1, 7, 3] ^= 0x01
    after = _send(worker)[0].values
    np.testing.assert_array_equal(after[:2], before[:2])
    assert not np.array_equal(after[2], before[2])


def test_kernel_block_split_matches_definition():
    logical = _logical_kv(heads=2)
    worker = _worker(
        _config(_group(_spec(2))), {_name(0): _cache(logical, kernel_split=2)}
    )
    np.testing.assert_array_equal(
        _send(worker)[0].values[:, 0], _reference({0: logical})
    )


def test_kv_sharing_layers_counted_once():
    """A KV-sharing layer aliasing its target's cache is counted once, under
    the target's layer number."""
    first, second = _logical_kv(heads=2, seed=1), _logical_kv(heads=2, seed=2)
    shared = _cache(first)
    worker = _worker(
        _config(_group(_spec(2), _name(0), _name(1), _name(2))),
        {_name(0): shared, _name(1): _cache(second), _name(2): shared},
    )
    np.testing.assert_array_equal(
        _send(worker)[0].values[:, 0], _reference({0: first, 1: second})
    )


def test_gathers_are_chunked_without_changing_results(monkeypatch):
    monkeypatch.setattr(checksum_worker, "_MAX_GATHER_BYTES", 1)
    logical = _logical_kv(heads=2)
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(logical)})
    np.testing.assert_array_equal(
        _send(worker)[0].values[:, 0], _reference({0: logical})
    )


_LOGICAL = _logical_kv(heads=2)


def _bf16_cache(blocks: int, tokens: int = BLOCK_SIZE) -> torch.Tensor:
    return torch.zeros(blocks, 2, tokens, 2 * HEAD_SIZE, dtype=torch.bfloat16)


def _misaligned_cache() -> torch.Tensor:
    """A cache view starting 2 bytes into its storage."""
    flat = _bf16_cache(NUM_BLOCKS).flatten()
    return torch.cat([flat[:1], flat])[1:].view_as(_bf16_cache(NUM_BLOCKS))


@pytest.mark.parametrize(
    "group, caches, worker_kwargs",
    [
        (
            _group(
                MambaSpec(
                    block_size=BLOCK_SIZE, shapes=((4,),), dtypes=(torch.float32,)
                )
            ),
            {_name(0): _cache(_LOGICAL)},
            {},
        ),
        (_group(_spec(2, num_head_slots=2)), {_name(0): _cache(_LOGICAL)}, {}),
        (_group(_spec(2), host_resident=True), {_name(0): _cache(_LOGICAL)}, {}),
        # One head per rank: sharded or replicated is unknown without a
        # model-wide KV head count.
        (
            _group(_spec(1)),
            {_name(0): _cache(_LOGICAL[:, :1])},
            {"tp_size": 2, "uniform_kv_heads": None},
        ),
        (_group(_spec(2)), {_name(0): _bf16_cache(NUM_BLOCKS + 1)}, {}),
        (_group(_spec(2)), {_name(0): _misaligned_cache()}, {}),
        (
            _group(_spec(2)),
            {_name(0): _bf16_cache(NUM_BLOCKS, tokens=BLOCK_SIZE // 2)},
            {},
        ),
        (_group(_spec(2), "attn"), {"attn": _cache(_LOGICAL)}, {}),
        (
            _group(_spec(2), _name(0), "mtp.layers.0.attn"),
            {_name(0): _cache(_LOGICAL), "mtp.layers.0.attn": _cache(_LOGICAL)},
            {},
        ),
    ],
    ids=[
        "mamba",
        "packed-head-slots",
        "host-resident",
        "unknown-replication",
        "block-count",
        "misaligned",
        "kernel-blocks-do-not-tile",
        "no-layer-number",
        "duplicate-layer-number",
    ],
)
def test_unsupported_groups_are_skipped(group, caches, worker_kwargs):
    """Unsupported groups are left out while other groups are checksummed."""
    good = _name(9)
    caches = {**caches, good: _cache(_LOGICAL)}
    worker = _worker(_config(group, _group(_spec(2), good)), caches, **worker_kwargs)
    blocks = KVChecksumBlocks(
        num_tokens=40, groups={0: (0, [3, 1, 4]), 1: (0, [3, 1, 4])}
    )
    record = _send_record(worker, blocks)
    assert record.skipped_groups == {0}
    checksums = record.checksums["req"]
    assert set(checksums) == {1}
    # Rank 0 holds the model's first heads, so its part covers only those.
    np.testing.assert_array_equal(checksums[1].values[:, 0], _reference({9: _LOGICAL}))


def test_recv_checksummed_only_when_its_load_finishes():
    """A request to receive is checksummed in the step its load finishes, not
    before, and is dropped if the load fails or the request ends."""
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(_LOGICAL)})
    worker.bind(
        KVChecksumScheduled(reqs_to_recv={"a": BLOCKS, "b": BLOCKS, "c": BLOCKS})
    )
    assert worker.post_forward(None, set(), set()) is None

    worker.bind(None)
    [record] = worker.post_forward({"a", "b", "c"}, {"b"}, {"c"}).finalize()
    assert set(record.checksums) == {"a"}

    worker.bind(None)
    assert worker.post_forward({"b", "c"}, set(), set()) is None


def test_deferred_sends_include_kv_written_before_finalize():
    """With a drafter writing KV after the connector step, the sends are
    checksummed by finalize_sends, so they cover the drafter's writes."""
    logical = _logical_kv(heads=2)
    cache = _cache(logical)
    worker = _worker(_config(_group(_spec(2))), {_name(0): cache})
    worker.bind(KVChecksumScheduled(reqs_to_send={"req": BLOCKS}))
    output = worker.post_forward(None, set(), set(), defer_sends=True)
    assert output is not None

    drafted = _logical_kv(heads=2, seed=7)
    cache.copy_(_cache(drafted))
    worker.finalize_sends()
    [record] = output.finalize()
    np.testing.assert_array_equal(
        record.checksums["req"][0].values[:, 0], _reference({0: drafted})
    )


def test_outputs_serialize_and_aggregate_across_workers():
    """Outputs finalize when serialized for the scheduler (deepcopy goes
    through the same __getstate__ as pickling), and the aggregator merges
    every worker's records."""
    logical = _logical_kv(heads=4)
    outputs = []
    for rank in range(2):
        worker = _worker(
            _config(_group(_spec(2))),
            {_name(0): _cache(logical[:, 2 * rank : 2 * rank + 2])},
            tp_rank=rank,
            tp_size=2,
            uniform_kv_heads=4,
        )
        worker.bind(KVChecksumScheduled(reqs_to_send={"req": BLOCKS}))
        kv_output = KVConnectorOutput(
            kv_checksums=worker.post_forward(None, set(), set())
        )
        assert not kv_output.is_empty()
        outputs.append(
            copy.deepcopy(ModelRunnerOutput.with_kv_conn_output_only(kv_output))
        )

    merged = KVOutputAggregator(expected_finished_count=2).aggregate(outputs)
    checksums = merged.kv_connector_output.kv_checksums
    assert isinstance(checksums, KVChecksumOutput)
    records = checksums.finalize()
    assert sorted(r.tp_rank for r in records) == [0, 1]
    total = sum(r.checksums["req"][0].values[:, 0] for r in records)
    np.testing.assert_array_equal(total, _reference({0: logical}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
def test_copy_is_ordered_after_checksums_for_any_reader():
    """The copy is queued with the checksums, so another thread finalizing
    (as the async output thread does) reads complete values even while the
    main stream is still busy."""
    logical = _logical_kv(heads=2)
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(logical).cuda()})
    main_stream = torch.cuda.Stream()
    torch.cuda.set_stream(main_stream)
    try:
        torch.cuda._sleep(200_000_000)
        worker.bind(KVChecksumScheduled(reqs_to_send={"req": BLOCKS}))
        output = worker.post_forward(None, set(), set())
    finally:
        torch.cuda.set_stream(torch.cuda.default_stream())

    records = []
    reader = threading.Thread(target=lambda: records.extend(output.finalize()))
    reader.start()
    reader.join()
    np.testing.assert_array_equal(
        records[0].checksums["req"][0].values[:, 0], _reference({0: logical})
    )
