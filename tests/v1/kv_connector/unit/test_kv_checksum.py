# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side KV checksums match their definition (per-slice checksums
weighted by layer and head), computed independently here, whatever the TP or
PP sharding, replication, memory layout or kernel block size of the KV
cache."""

import copy
from collections.abc import Iterable
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_checksum import checksum
from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    MAX_KV_HEADS_PER_LAYER,
    KVChecksumBlocks,
    KVChecksumRecord,
    KVChecksumScheduled,
    kv_checksum_platform_supported,
)
from vllm.distributed.kv_transfer.kv_checksum.scheduler import _sum_replicas
from vllm.distributed.kv_transfer.kv_checksum.worker import (
    KVChecksumWorker,
    PendingKVChecksums,
)
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MambaStateSplit,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.worker.utils import allocate_kv_cache

# KV checksums are computed by a Triton kernel, on CUDA, ROCm or XPU.
pytestmark = pytest.mark.skipif(
    not kv_checksum_platform_supported(), reason="needs a GPU with Triton"
)

# KV caches live here; logical KV and the reference checksums stay on the host.
DEVICE = current_platform.device_type

BLOCK_SIZE = 16
HEAD_SIZE = 8  # (K + V) * bf16 = 32 bytes = 4 pieces per cell
CELL_BYTES = 4 * HEAD_SIZE
NUM_BLOCKS = 6
# Three blocks covering 40 tokens: the last block (id 4) holds tokens 32..39.
BLOCKS = KVChecksumBlocks(num_tokens=40, groups={0: (0, [3, 1, 4])})


def _name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.attn"


def _spec(
    heads: int, block_size: int = BLOCK_SIZE, head_size: int = HEAD_SIZE, **kwargs
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=heads,
        head_size=head_size,
        dtype=torch.bfloat16,
        **kwargs,
    )


def _config(*groups: KVCacheGroupSpec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS, kv_cache_tensors=[], kv_cache_groups=list(groups)
    )


def _group(spec, *layer_names: str, **kwargs) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(list(layer_names or (_name(0),)), spec, **kwargs)


def _worker(
    config, caches, *, tp_rank=0, tp_size=1, uniform_kv_heads=2, draft_layers=()
):
    return KVChecksumWorker(
        config,
        caches,
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        uniform_kv_heads=uniform_kv_heads,
        draft_layer_names=frozenset(draft_layers),
    )


def _logical_kv(
    heads: int, seed: int = 0, tokens: int = BLOCK_SIZE, cell_bytes: int = CELL_BYTES
) -> torch.Tensor:
    """Random logical KV bytes of one layer: [blocks, heads, tokens, cell]."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(
        0,
        256,
        (NUM_BLOCKS, heads, tokens, cell_bytes),
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
            device=DEVICE,
        )
        view = storage.permute(0, 2, 1, 3)
    else:
        view = torch.empty_like(data, device=DEVICE)
    view.copy_(data)
    return view.view(torch.bfloat16)


def _reference(
    layers: dict[int, torch.Tensor], blocks: KVChecksumBlocks = BLOCKS
) -> np.ndarray:
    """Checksums from the definition: each (layer, head) slice of a block is
    s = sum_j x_j * (2j + 1) over its 8-byte pieces, numbered in token order
    from the request's first token, and the block's checksum is sum m * s with
    m = 2 * (layer * H + head) + 1.

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
        for row, block_id in enumerate(block_ids):
            first_token = np.uint64((start + row) * tokens)
            position = (first_token + token) * np.uint64(num_pieces) + piece
            weights = slice_factor * (np.uint64(2) * position + np.uint64(1))
            block = pieces[block_id].copy()
            block[:, max(0, blocks.num_tokens - (start + row) * tokens) :, :] = 0
            with np.errstate(over="ignore"):
                sums[row] += (block * weights).sum(dtype=np.uint64)
    return sums


def _results(
    *finished_recving: str, failed: Iterable[str] = ()
) -> KVConnectorTransferResults:
    """A connector's transfer results with these loads finished."""
    return KVConnectorTransferResults(
        finished_recving=set(finished_recving), failed_recving=set(failed)
    )


def _finish(pending: PendingKVChecksums | None) -> KVChecksumRecord:
    """Copy the checksums to host as the model runner does, and wait."""
    assert pending is not None
    record = pending.enqueue_async_cpu_copy()
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    return record


def _send_record(worker: KVChecksumWorker, blocks=BLOCKS) -> KVChecksumRecord:
    worker.bind(KVChecksumScheduled(reqs_to_send={"req": blocks}))
    return _finish(worker.post_forward(_results(), set()))


def _send(worker: KVChecksumWorker, blocks=BLOCKS):
    return _send_record(worker, blocks).checksums["req"]


def _recv(worker: KVChecksumWorker, blocks=BLOCKS):
    worker.bind(KVChecksumScheduled(reqs_to_recv={"req": blocks}))
    record = _finish(worker.post_forward(_results("req"), set()))
    return record.checksums["req"]


@pytest.mark.parametrize(
    "platform_supported,dcp_size,error",
    [(False, 1, "need Triton"), (True, 2, "context parallelism"), (True, 1, None)],
)
def test_unsupported_setups_fail_validation(
    monkeypatch, platform_supported, dcp_size, error
):
    """Enabling KV checksums where they cannot run fails startup rather than
    silently leaving loads unverified."""
    monkeypatch.setattr(
        checksum, "kv_checksum_platform_supported", lambda: platform_supported
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(enable_kv_checksum=True),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size, prefill_context_parallel_size=1
        ),
    )
    if error is None:
        VllmConfig._validate_kv_checksum(config)  # type: ignore[arg-type]
    else:
        with pytest.raises(ValueError, match=error):
            VllmConfig._validate_kv_checksum(config)  # type: ignore[arg-type]


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_only_the_first_worker_passes_on_received_checksums(tp_rank):
    """Every consumer worker's connector receives the producer's checksums;
    the scheduler needs them once."""
    worker = _worker(
        _config(_group(_spec(2))),
        {_name(0): _cache(_logical_kv(heads=2))},
        tp_rank=tp_rank,
        tp_size=2,
    )
    results = _results("req")
    results.kv_checksums = {"req": b"checksums"}
    pending = worker.post_forward(results, set())
    if tp_rank == 0:
        assert _finish(pending).received == {"req": b"checksums"}
    else:
        # Nothing else to report this step.
        assert pending is None


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


@pytest.mark.parametrize("declared", [True, False])
def test_mla_replicas_are_checked_per_replica(declared):
    """A cache with one TP shard (MLA) is replicated on every rank, as its
    spec declares (max_tp_shards=1) or its type implies, whatever the model's
    KV head count (above 1 in Kimi-K3's config)."""
    spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=2 * HEAD_SIZE,
        head_size_v=0,
        dtype=torch.bfloat16,
        max_tp_shards=1 if declared else None,
    )
    logical = _logical_kv(heads=1)
    workers = [
        _worker(
            _config(_group(spec)),
            {_name(0): _cache(logical)},
            tp_rank=r,
            tp_size=2,
            uniform_kv_heads=8,
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


def test_checksums_add_up_across_block_sizes():
    """Blocks of 16 tokens add up to the 32-token blocks holding the same
    tokens, also where the last one is partial, so producers and consumers
    with different block sizes can compare."""
    small = _logical_kv(heads=2)
    large = (
        small.view(NUM_BLOCKS // 2, 2, 2, BLOCK_SIZE, CELL_BYTES)
        .transpose(1, 2)
        .reshape(NUM_BLOCKS // 2, 2, 2 * BLOCK_SIZE, CELL_BYTES)
        .repeat(2, 1, 1, 1)  # As many blocks as the config.
    )
    small_worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(small)})
    large_spec = _spec(2, block_size=2 * BLOCK_SIZE)
    large_worker = _worker(_config(_group(large_spec)), {_name(0): _cache(large)})
    num_tokens = 4 * BLOCK_SIZE + 8
    small_values = _send(
        small_worker, KVChecksumBlocks(num_tokens, {0: (0, [0, 1, 2, 3, 4])})
    )[0].values[:, 0]
    large_values = _send(
        large_worker, KVChecksumBlocks(num_tokens, {0: (0, [0, 1, 2])})
    )[0].values[:, 0]
    np.testing.assert_array_equal(
        np.add.reduceat(small_values, [0, 2, 4]), large_values
    )


def test_requests_and_groups_of_one_step_are_checksummed_together():
    """Loads of several requests and groups finishing in one step, starting
    past the first block and ending in partial blocks, match the definition.
    Each group's rows are read from one shared copy, after a group with an odd
    number of rows."""
    first, second = _logical_kv(heads=2, seed=1), _logical_kv(heads=2, seed=2)
    worker = _worker(
        _config(_group(_spec(2), _name(0)), _group(_spec(2), _name(1))),
        {_name(0): _cache(first), _name(1): _cache(second)},
    )
    reqs = {
        "a": KVChecksumBlocks(
            num_tokens=40, groups={0: (1, [3, 1]), 1: (0, [2, 5, 0])}
        ),
        "b": KVChecksumBlocks(num_tokens=21, groups={0: (1, [2])}),
    }
    worker.bind(KVChecksumScheduled(reqs_to_recv=reqs))
    record = _finish(worker.post_forward(_results(*reqs), set()))
    for req_id, blocks in reqs.items():
        for group_id, (start, block_ids) in blocks.groups.items():
            kv = first if group_id == 0 else second
            group_blocks = KVChecksumBlocks(blocks.num_tokens, {0: (start, block_ids)})
            np.testing.assert_array_equal(
                record.checksums[req_id][group_id].values[:, 0],
                _reference({group_id: kv}, group_blocks),
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


def test_caches_of_one_layer_take_consecutive_slices():
    """Distinct caches with one layer number (e.g. an attention's KV and its
    indexer's) are all covered: the second, in layer name order, takes the
    heads after the first's."""
    kv, index = _logical_kv(heads=2, seed=1), _logical_kv(heads=2, seed=2)
    index_name = "model.layers.0.self_attn.indexer.k_cache"
    worker = _worker(
        _config(_group(_spec(2), index_name, _name(0))),
        {_name(0): _cache(kv), index_name: _cache(index)},
    )
    np.testing.assert_array_equal(
        _send(worker)[0].values[:, 0], _reference({0: torch.cat([kv, index], 1)})
    )


@pytest.mark.parametrize("declared", [True, False])
@pytest.mark.parametrize("tp_size", [2, 4])
def test_replicated_indexer_cache_sums_across_tp(tp_size, declared):
    """As in MiniMax-M3: per layer, attention KV of 2 heads (sharded at TP2,
    replicated at TP4) and an indexer cache of one head held whole by every
    rank, as its spec declares (max_tp_shards=1) or its MLA spec type
    implies. Producer shards sum to the definition with each head counted
    once, and every consumer replica check equals it."""
    kv, index = _logical_kv(heads=2, seed=1), _logical_kv(heads=1, seed=2)
    index_name = "model.layers.0.self_attn.indexer.k_cache"
    index_spec = MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        max_tp_shards=1 if declared else None,
    )
    group = KVCacheGroupSpec(
        [_name(0), index_name],
        UniformTypeKVCacheSpecs(
            block_size=BLOCK_SIZE,
            kv_cache_specs={_name(0): _spec(1), index_name: index_spec},
        ),
    )
    head_replicas = tp_size // 2
    workers = [
        _worker(
            _config(group),
            {
                _name(0): _cache(kv[:, rank // head_replicas :][:, :1]),
                index_name: _cache(index),
            },
            tp_rank=rank,
            tp_size=tp_size,
        )
        for rank in range(tp_size)
    ]
    expected = _reference({0: torch.cat([kv, index], 1)})
    with np.errstate(over="ignore"):
        sent = sum(
            group.values.sum(axis=1, dtype=np.uint64)
            for worker in workers
            for group in _send(worker).values()
        )
    np.testing.assert_array_equal(sent, expected)
    received = [_recv(worker)[0] for worker in workers]
    checks = _sum_replicas(received, len(expected))
    assert len(checks) == tp_size
    for check in checks:
        np.testing.assert_array_equal(check, expected)


def _mamba_name(layer: int) -> str:
    return f"model.layers.{layer}.linear_attn"


# conv (4 x 8 bf16 = 64 bytes) and ssm (2 x 8 fp32 = 64 bytes): 16 pieces,
# then an extra state (5 fp32 = 20 bytes) that is not checksummed.
MAMBA_SHAPES = ((4, 8), (2, 8), (5,))
MAMBA_DTYPES = (torch.bfloat16, torch.float32, torch.float32)
MAMBA_COVERED_BYTES = 128


def _mamba_spec(shapes=MAMBA_SHAPES, dtypes=MAMBA_DTYPES, **kwargs) -> MambaSpec:
    return MambaSpec(block_size=10000, shapes=shapes, dtypes=dtypes, **kwargs)


def _mamba_cache(
    content_bytes: int, page_bytes: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """A Mamba layer's ``[blocks, 1, 1, content]`` int8 cache view over padded
    pages, as bound by the model runner, and the pages it views."""
    gen = torch.Generator().manual_seed(seed)
    pages = torch.randint(
        0, 256, (NUM_BLOCKS, page_bytes), generator=gen, dtype=torch.uint8
    ).to(DEVICE)
    view = pages[:, :content_bytes].view(torch.int8).view(NUM_BLOCKS, 1, 1, -1)
    return view, pages


def _mamba_reference(
    states: dict[int, torch.Tensor],
    block_id: int,
    covered_bytes: int = MAMBA_COVERED_BYTES,
) -> np.uint64:
    """The definition for Mamba state: per (layer, state shard) the slice of
    the covered state bytes, s = sum_j x_j * (2j + 1), weighted by
    m = 2 * (layer * H + shard) + 1. ``states`` maps layer * H + shard to its
    pages."""
    total = np.uint64(0)
    for slice_index, pages in states.items():
        pieces = pages[block_id, :covered_bytes].cpu()
        x = pieces.view(torch.int64).numpy().view(np.uint64)
        j = np.arange(len(x), dtype=np.uint64)
        with np.errstate(over="ignore"):
            s = (x * (np.uint64(2) * j + np.uint64(1))).sum(dtype=np.uint64)
            total += (np.uint64(2) * np.uint64(slice_index) + np.uint64(1)) * s
    return total


MAMBA_BLOCKS = KVChecksumBlocks(num_tokens=39, groups={0: (0, [3])})


_LOGICAL = _logical_kv(heads=2)


def _bf16_cache(blocks: int, tokens: int = BLOCK_SIZE) -> torch.Tensor:
    return torch.zeros(
        blocks, 2, tokens, 2 * HEAD_SIZE, dtype=torch.bfloat16, device=DEVICE
    )


def _misaligned_cache() -> torch.Tensor:
    """A cache view starting 2 bytes into its storage."""
    flat = _bf16_cache(NUM_BLOCKS).flatten()
    return torch.cat([flat[:1], flat])[1:].view_as(_bf16_cache(NUM_BLOCKS))


@pytest.mark.parametrize(
    "group, caches, worker_kwargs",
    [
        (
            _group(_mamba_spec(((3,),), (torch.float16,)), _mamba_name(0)),
            {_mamba_name(0): _mamba_cache(6, 8)[0]},
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
        # Under PP these could be on different stages, numbered apart.
        (
            _group(_spec(2), _name(0), "mtp.layers.0.attn"),
            {_name(0): _cache(_LOGICAL), "mtp.layers.0.attn": _cache(_LOGICAL)},
            {},
        ),
    ],
    ids=[
        "mamba-state-not-whole-pieces",
        "packed-head-slots",
        "host-resident",
        "unknown-replication",
        "block-count",
        "misaligned",
        "kernel-blocks-do-not-tile",
        "no-layer-number",
        "layer-number-of-two-modules",
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


def test_mamba_state_checksum_covers_the_transferred_states():
    """A Mamba layer's state is one slice of its conv and ssm bytes: extra
    states and page padding (never transferred, possibly stale) do not count,
    a flipped state bit does."""
    cache, pages = _mamba_cache(content_bytes=148, page_bytes=160)
    worker = _worker(
        _config(_group(_mamba_spec(), _mamba_name(2))), {_mamba_name(2): cache}
    )
    [group] = _send(worker, MAMBA_BLOCKS).values()
    expected = _mamba_reference({2 * MAX_KV_HEADS_PER_LAYER: pages}, 3)
    assert group.values[0, 0] == expected

    pages[3, MAMBA_COVERED_BYTES:] ^= 0xFF  # Extra state and padding.
    assert _send(worker, MAMBA_BLOCKS)[0].values[0, 0] == expected
    pages[3, MAMBA_COVERED_BYTES - 1] ^= 1
    assert _send(worker, MAMBA_BLOCKS)[0].values[0, 0] != expected


def test_mamba_state_checksum_on_allocated_kv_cache():
    """On the model runner's views of an allocated cache (two Mamba layers
    interleaved per block, padded pages, a nonzero offset), each layer's
    checksum covers its own state bytes, located independently here from the
    tensor's offset and strides."""
    page, layers = 160, [_mamba_name(2), _mamba_name(5)]
    tensor = KVCacheTensor(
        size=8 + 2 * page * NUM_BLOCKS,
        layers=layers,
        layer_stride=page,
        block_stride=2 * page,
        offset=8,
    )
    spec = _mamba_spec(page_size_padded=page)
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[_group(spec, *layers)],
    )
    caches = allocate_kv_cache(config, torch.device(DEVICE), KVCacheLayout.LBHNC)
    raw = torch.empty(0, dtype=torch.uint8, device=DEVICE).set_(
        caches[layers[0]].untyped_storage()
    )
    raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8))

    def pages(layer: int) -> torch.Tensor:
        starts = [
            tensor.offset + layer * tensor.layer_stride + b * tensor.block_stride
            for b in range(NUM_BLOCKS)
        ]
        return torch.stack([raw[start : start + page] for start in starts])

    worker = _worker(config, caches)
    expected = _mamba_reference(
        {2 * MAX_KV_HEADS_PER_LAYER: pages(0), 5 * MAX_KV_HEADS_PER_LAYER: pages(1)},
        3,
    )
    assert _send(worker, MAMBA_BLOCKS)[0].values[0, 0] == expected


@pytest.mark.parametrize("tp_replicated", [False, True])
def test_mamba_state_tp_shards_and_replicas(tp_replicated):
    """TP-sharded state sums its shards (each weighed by its shard index);
    replicated state is checked per replica, the producer sending replica 0."""
    spec = _mamba_spec(tp_replicated=tp_replicated)
    caches = [_mamba_cache(148, 160, seed=rank) for rank in range(2)]
    if tp_replicated:
        caches[1][1].copy_(caches[0][1])
    workers = [
        _worker(
            _config(_group(spec, _mamba_name(2))),
            {_mamba_name(2): caches[rank][0]},
            tp_rank=rank,
            tp_size=2,
        )
        for rank in range(2)
    ]
    first = 2 * MAX_KV_HEADS_PER_LAYER
    if tp_replicated:
        expected = _mamba_reference({first: caches[0][1]}, 3)
        sent = [_send(worker, MAMBA_BLOCKS) for worker in workers]
        assert sent[1] == {} and sent[0][0].values[0, 0] == expected
        for rank, worker in enumerate(workers):
            [received] = _recv(worker, MAMBA_BLOCKS).values()
            assert received.replica_keys == [(2, rank)]
            assert received.values[0, 0] == expected
    else:
        expected = _mamba_reference({first: caches[0][1], first + 1: caches[1][1]}, 3)
        total = sum(_send(worker, MAMBA_BLOCKS)[0].values[0, 0] for worker in workers)
        assert np.uint64(total) == expected


def test_mamba_fingerprint_follows_state_shapes_not_block_size():
    """Producer and consumer compare a Mamba group only if its checksummed
    states have the same shapes (which change with the TP size), whatever
    their block sizes (the sequence length without prefix caching)."""

    def fingerprint(spec: MambaSpec) -> int:
        cache = _mamba_cache(148, 160)[0]
        worker = _worker(_config(_group(spec, _mamba_name(2))), {_mamba_name(2): cache})
        return _send_record(worker, MAMBA_BLOCKS).fingerprints[0]

    base = fingerprint(_mamba_spec())
    assert (
        fingerprint(
            MambaSpec(block_size=4096, shapes=MAMBA_SHAPES, dtypes=MAMBA_DTYPES)
        )
        == base
    )
    assert fingerprint(_mamba_spec(shapes=((4, 8), (2, 8), (6,)))) == base
    assert fingerprint(_mamba_spec(shapes=((2, 16), (2, 8), (5,)))) != base


# A TP-split Mamba2 layer, globally: conv [x, B, C] of 8 rows each by 3 bf16
# columns, and a temporal state of 8 heads of 2 x 2 fp32.
SPLIT_ROWS = 8


def _split_states(fp32_conv: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    conv = torch.randint(-(2**31), 2**31, (3 * SPLIT_ROWS, 3), generator=gen)
    temporal = torch.randint(-(2**31), 2**31, (SPLIT_ROWS, 2, 2), generator=gen)
    temporal = temporal.to(torch.int32).view(torch.float32)
    if fp32_conv:
        return conv.to(torch.int32).view(torch.float32), temporal
    return conv.to(torch.int16).view(torch.bfloat16), temporal


def _split_layer(
    states: tuple[torch.Tensor, torch.Tensor],
    tp_size: int,
    rank: int,
    one_group: bool = False,
    dim_first: bool = True,
) -> tuple[MambaSpec, torch.Tensor]:
    """A rank's spec and ``[blocks, 1, 1, state]`` cache with its shard of
    the global ``states`` in block 3. With ``one_group``, every rank holds B
    and C whole, as with a single Mamba2 group."""
    conv, temporal = states
    rows = SPLIT_ROWS // tp_size
    shard = slice(rank * rows, (rank + 1) * rows)
    x, b, c = conv.split(SPLIT_ROWS)
    if one_group:
        group_rows, replicas = SPLIT_ROWS, tp_size
        local_conv = torch.cat([x[shard], b, c])
    else:
        group_rows, replicas = rows, 1
        local_conv = torch.cat([x[shard], b[shard], c[shard]])
    if not dim_first:
        local_conv = local_conv.T.contiguous()
    local_temporal = temporal[shard].contiguous()
    spec = _mamba_spec(
        shapes=(tuple(local_conv.shape), tuple(local_temporal.shape)),
        dtypes=(local_conv.dtype, local_temporal.dtype),
        state_splits=(
            MambaStateSplit(
                0 if dim_first else 1,
                ((rows, 1), (group_rows, replicas), (group_rows, replicas)),
            ),
            MambaStateSplit(0, ((rows, 1),)),
        ),
    )
    content = torch.cat(
        [t.view(torch.uint8).flatten() for t in (local_conv, local_temporal)]
    )
    cache, pages = _mamba_cache(len(content), len(content) + 8, seed=rank)
    pages[3, : len(content)] = content
    return spec, cache


def _split_reference(states: tuple[torch.Tensor, torch.Tensor]) -> np.uint64:
    """The definition for TP-split Mamba state: one slice per global row (the
    conv rows, then the temporal heads), whose pieces are its elements read
    unsigned."""
    conv, temporal = states
    total = np.uint64(0)
    for row, data in enumerate([*conv, *temporal]):
        unsigned = np.uint16 if data.element_size() == 2 else np.uint32
        x = data.contiguous().view(
            torch.int16 if unsigned == np.uint16 else torch.int32
        )
        x = x.flatten().numpy().view(unsigned).astype(np.uint64)
        j = np.arange(len(x), dtype=np.uint64)
        m = np.uint64(2 * (2 * MAX_KV_HEADS_PER_LAYER + row) + 1)
        with np.errstate(over="ignore"):
            total += m * (x * (np.uint64(2) * j + np.uint64(1))).sum(dtype=np.uint64)
    return total


def _split_workers(
    states, tp_size, **kwargs
) -> tuple[list[KVChecksumWorker], list[torch.Tensor]]:
    workers, caches = [], []
    for rank in range(tp_size):
        spec, cache = _split_layer(states, tp_size, rank, **kwargs)
        caches.append(cache)
        workers.append(
            _worker(
                _config(_group(spec, _mamba_name(2))),
                {_mamba_name(2): cache},
                tp_rank=rank,
                tp_size=tp_size,
            )
        )
    return workers, caches


@pytest.mark.parametrize("fp32_conv", [False, True])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("one_group", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_split_mamba_state_checksum_is_tp_independent(
    tp_size, one_group, dim_first, fp32_conv
):
    """With its TP split known, a Mamba layer's producer checksum (replica 0
    of replicated B and C) is the definition over the global state, whatever
    the TP size, conv layout or conv dtype, so producers and consumers of
    different TP sizes compare it."""
    states = _split_states(fp32_conv)
    workers, _ = _split_workers(
        states, tp_size, one_group=one_group, dim_first=dim_first
    )
    total = np.uint64(0)
    for worker in workers:
        for group in _send(worker, MAMBA_BLOCKS).values():
            with np.errstate(over="ignore"):
                total += group.values.sum(dtype=np.uint64)
    assert total == _split_reference(states)


def test_split_mamba_state_checks_each_replica():
    """Replicated B and C are checked per replica: corrupting one rank's copy
    fails that replica's check only."""
    states = _split_states()
    workers, caches = _split_workers(states, 2, one_group=True)

    def checks():
        groups = [group for w in workers for group in _recv(w, MAMBA_BLOCKS).values()]
        return [int(check[0]) for check in _sum_replicas(groups, 1)]

    expected = int(_split_reference(states))
    assert checks() == [expected, expected]
    # Rank 1's copy of B, after its x rows of 3 bf16.
    caches[1][3, 0, 0, SPLIT_ROWS // 2 * 3 * 2] ^= 1
    first, second = checks()
    assert first == expected and second != expected


def test_split_mamba_fingerprint_is_tp_independent():
    """The fingerprint of a TP-split Mamba layer follows its global shapes:
    equal across TP sizes and conv layouts, different for other conv widths
    (e.g. other speculative token counts)."""

    def fingerprint(states, tp_size, **kwargs) -> int:
        [worker, *_], _ = _split_workers(states, tp_size, **kwargs)
        return _send_record(worker, MAMBA_BLOCKS).fingerprints[0]

    states = _split_states()
    base = fingerprint(states, 1)
    assert fingerprint(states, 2) == base
    assert fingerprint(states, 2, dim_first=False) == base
    wider = (torch.cat([states[0], states[0][:, :1]], dim=1), states[1])
    assert fingerprint(wider, 2) != base


def test_state_splits_cover_state_shapes():
    """The model layers' TP splits partition the split dimension of the state
    shapes they describe, into segments whose global sizes do not depend on
    the TP size; with one Mamba2 group, B and C are whole on every rank."""
    calc = MambaStateShapeCalculator

    def cases(tp_size):
        yield (
            calc.mamba1_state_shape(tp_size, 64, 16, 4),
            calc.mamba1_state_splits(tp_size, 64),
        )
        yield (
            calc.gated_delta_net_state_shape(tp_size, 4, 8, 16, 32, 4),
            calc.gated_delta_net_state_splits(tp_size, 4, 8, 16, 32),
        )
        yield (
            calc.kda_state_shape(tp_size, 8, 16, conv_kernel_size=4),
            calc.kda_state_splits(tp_size, 8, 16),
        )
        yield (
            calc.linear_attention_state_shape(
                num_heads=8, tp_size=tp_size, head_dim=16
            ),
            calc.linear_attention_state_splits(tp_size, 8),
        )
        yield (
            calc.short_conv_state_shape(tp_size, 64, 4),
            calc.short_conv_state_splits(tp_size, 64),
        )
        for n_groups in (1, 4):
            yield (
                calc.mamba2_state_shape(tp_size, 64, n_groups, 8, 8, 16, 4),
                calc.mamba2_state_splits(tp_size, 64, n_groups, 8, 16),
            )

    global_sizes = []
    for tp_size in (1, 2, 4):
        sizes = []
        for shapes, splits in cases(tp_size):
            assert splits is not None
            for shape, split in zip(shapes, splits):
                assert sum(local for local, _ in split.segments) == shape[split.axis]
                sizes.append([local * tp_size // r for local, r in split.segments])
        global_sizes.append(sizes)
    assert global_sizes[0] == global_sizes[1] == global_sizes[2]
    [conv, _] = calc.mamba2_state_splits(2, 64, 1, 8, 16)
    assert conv.segments == ((32, 1), (16, 2), (16, 2))
    [conv, recurrent] = calc.kda_state_splits(2, 8, 16)
    assert conv.segments == ((64, 1), (64, 1), (64, 1))
    assert recurrent == MambaStateSplit(0, ((4, 1),))


def test_recv_checksummed_only_when_its_load_finishes():
    """A request to receive is checksummed in the step its load finishes, not
    before, and is dropped if the load fails or the request ends."""
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(_LOGICAL)})
    worker.bind(
        KVChecksumScheduled(reqs_to_recv={"a": BLOCKS, "b": BLOCKS, "c": BLOCKS})
    )
    assert worker.post_forward(_results(), set()) is None

    worker.bind(None)
    record = _finish(worker.post_forward(_results("a", "b", "c", failed={"b"}), {"c"}))
    assert set(record.checksums) == {"a"}

    worker.bind(None)
    assert worker.post_forward(_results("b", "c"), set()) is None


def test_send_and_finished_load_in_one_step():
    """A step that both finishes a load and finishes a request to send
    reports the checksums of both, in every group."""
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(_LOGICAL)})
    worker.bind(
        KVChecksumScheduled(
            reqs_to_send={"send": BLOCKS}, reqs_to_recv={"recv": BLOCKS}
        )
    )
    record = _finish(worker.post_forward(_results("recv"), set()))
    expected = _reference({0: _LOGICAL})
    for req_id in ("send", "recv"):
        np.testing.assert_array_equal(
            record.checksums[req_id][0].values[:, 0], expected
        )


def test_draft_layers_are_not_checksummed():
    """Drafter layers are left out of their group's checksums and
    fingerprint, and a group of drafter layers only is not checksummed (nor
    reported as skipped)."""
    draft = "drafter.layers.1.self_attn.attn"
    target_only = _worker(_config(_group(_spec(2))), {_name(0): _cache(_LOGICAL)})
    caches = {_name(0): _cache(_LOGICAL), draft: _cache(_logical_kv(2, seed=9))}
    worker = _worker(
        _config(_group(_spec(2), _name(0), draft), _group(_spec(2), draft)),
        caches,
        draft_layers={draft},
    )
    blocks = KVChecksumBlocks(
        num_tokens=BLOCKS.num_tokens, groups={0: BLOCKS.groups[0], 1: BLOCKS.groups[0]}
    )
    record = _send_record(worker, blocks)
    assert set(record.checksums["req"]) == {0}
    np.testing.assert_array_equal(
        record.checksums["req"][0].values[:, 0], _reference({0: _LOGICAL})
    )
    assert record.fingerprints[0] == _send_record(target_only).fingerprints[0]
    assert not record.skipped_groups


def test_outputs_serialize_and_aggregate_across_workers():
    """Workers' records are plain host data that serialize for the scheduler
    (deepcopy goes through the same path as pickling), and the aggregator
    merges every worker's records."""
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
            kv_checksums=[_finish(worker.post_forward(_results(), set()))]
        )
        assert not kv_output.is_empty()
        outputs.append(
            copy.deepcopy(ModelRunnerOutput.with_kv_conn_output_only(kv_output))
        )

    merged = KVOutputAggregator(expected_finished_count=2).aggregate(outputs)
    records = merged.kv_connector_output.kv_checksums
    assert records is not None
    assert sorted(r.tp_rank for r in records) == [0, 1]
    total = sum(r.checksums["req"][0].values[:, 0] for r in records)
    np.testing.assert_array_equal(total, _reference({0: logical}))
