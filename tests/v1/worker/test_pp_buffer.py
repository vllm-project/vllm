# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from vllm.v1.worker.pp.boundary import DynamicPPConfig, PPBoundary, PPBoundaryController
from vllm.v1.worker.pp.buffer import PPSendBuffer
from vllm.v1.worker.pp.state import (
    CONTROL_KEY_PREFIX,
    PPControlChannel,
    PPStateTransfer,
)


def _gloo_worker(rank, rendezvous, first_two_posted):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        controller = PPBoundaryController(DynamicPPConfig(redundant_layers=2), 6, 12)
        torch.manual_seed(7)
        layers = [(torch.randn(8, 8) / 4, torch.randn(8)) for _ in range(12)]

        def execute(value, start, end):
            for weight, bias in layers[start:end]:
                value = torch.tanh(value @ weight + bias)
            return value

        if rank == 0:
            buffer = PPSendBuffer(2)
            output = torch.empty(7, 4, 8)
            buffer.reserve({"hidden_states": output})
            for batch in range(12):
                rows = 3 if batch % 2 else 7
                boundary = 4 + batch % 5
                buffer.acquire()
                output[:rows].copy_(
                    execute(torch.full_like(output[:rows], batch), 0, boundary)
                )

                def send(payload):
                    header = torch.tensor(payload["header"], dtype=torch.int64)
                    # Gloo Work retains the header allocation until completion.
                    return [
                        dist.isend(header, 1),
                        dist.isend(payload["hidden_states"], 1),
                    ]

                buffer.send(
                    {
                        "hidden_states": output[:rows],
                        "header": (batch, boundary, batch, rows),
                    },
                    send,
                )
                output.fill_(-100)
                if batch == 1:
                    first_two_posted.set()
            buffer.drain()
            assert buffer.num_sends == 12
        else:
            assert first_two_posted.wait(30), (
                "producer waited for receive before filling two slots"
            )
            for batch in range(12):
                wire = torch.empty(4, dtype=torch.int64)
                dist.recv(wire, 0)
                batch_id, cut, version, rows = wire.tolist()
                header = controller.accept((batch_id, cut, version), batch)
                value = torch.empty(rows, 4, 8)
                dist.recv(value, 0)
                actual = execute(value, *controller.execution_range(1, header))
                expected = execute(torch.full_like(value, batch), 0, 12)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_delayed_receiver_and_changing_boundaries_preserve_each_layer(tmp_path):
    context = torch.multiprocessing.get_context("spawn")
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", context.Event()),
        nprocs=2,
        join=True,
    )


def test_reserved_storage_rejects_an_unbudgeted_shape():
    buffer = PPSendBuffer(2)
    buffer.reserve({"hidden_states": torch.empty(7, 4, 8)})
    buffer.acquire()
    with pytest.raises(ValueError, match="exceeds its reserved"):
        buffer.send({"hidden_states": torch.empty(8, 4, 8)}, lambda tensors: ())
    buffer.cancel()
    assert buffer.reserved_bytes == 2 * 7 * 4 * 8 * 4


def test_snapshot_separates_output_and_metadata_from_the_next_forward():
    buffer = PPSendBuffer(1)
    output = torch.ones(2, 4, 8)
    header = [0, 4, 0]
    received = []
    buffer.acquire()

    def send(tensors):
        received.append(tensors)
        return ()

    buffer.send({"hidden_states": output, "header": header}, send)
    output.zero_()
    header[1] = 5
    assert received[0]["header"] == [0, 4, 0]
    torch.testing.assert_close(received[0]["hidden_states"], torch.ones_like(output))
    buffer.drain()


def _cache_components():
    return {
        "kv": torch.zeros(48, dtype=torch.int8),
        "compressor": torch.zeros(64, dtype=torch.int8),
        "indexer": torch.zeros(32, dtype=torch.int8),
        "swa": torch.zeros(80, dtype=torch.int8),
    }


def _stateful_layer(value, state, weight, layer, batch):
    kv = state["kv"][:32].view(torch.float64)
    compressor = state["compressor"][:32].view(torch.float64)
    indexer = state["indexer"][:16].view(torch.int64)
    swa = state["swa"][:32].view(torch.float64)
    indexer[0] += 1
    indexer[1] += layer + 1
    value = torch.tanh(
        value @ weight + 0.3 * kv + 0.2 * compressor + 0.1 * swa + 0.05 * indexer[0]
    )
    kv.copy_(value)
    compressor.mul_(0.75).add_(value * 0.25)
    swa.copy_(value.roll(1))
    for key, raw in state.items():
        logical_bytes = 16 if key == "indexer" else 32
        raw[logical_bytes:].fill_((layer + batch + len(key)) % 127)
    return value


def _state_transfer_worker(rank, rendezvous, mode):
    from vllm.distributed.parallel_state import GroupCoordinator

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    pp = GroupCoordinator([[0, 1]], rank, "gloo", False, group_name="pp_state")
    try:
        if mode == "metadata":
            for problem in ("shape", "dtype", "missing", "alias", "bytes", "padding"):
                state = {23: _cache_components(), 24: _cache_components()}
                if rank == 1:
                    if problem == "shape":
                        state[23]["kv"] = torch.zeros((2, 24), dtype=torch.int8)
                    elif problem == "dtype":
                        state[23]["kv"] = torch.zeros(48, dtype=torch.float32)
                    elif problem == "missing":
                        del state[23]["indexer"]
                    elif problem == "alias":
                        state[23]["indexer"] = state[23]["kv"]
                    elif problem == "bytes":
                        state[23]["kv"] = torch.zeros(56, dtype=torch.int8)
                    else:
                        state[23]["kv"] = torch.zeros(64, dtype=torch.int8)[:48]
                with pytest.raises(ValueError, match="PP cache"):
                    PPStateTransfer(pp, torch.cpu, state)
                dist.barrier()

            state = {23: _cache_components(), 24: _cache_components()}
            transfer = PPStateTransfer(pp, torch.cpu, state)
            with pytest.raises(ValueError, match="differs between stages"):
                transfer.move(23, 24 if rank == 0 else 25, 0)
            if rank == 1:
                state[23]["kv"] = torch.zeros(48, dtype=torch.int8)
            with pytest.raises(ValueError, match="allocations changed"):
                transfer.move(23, 24, 0)
            return

        states = {layer: _cache_components() for layer in range(26)}
        reference = {layer: _cache_components() for layer in range(26)}
        generator = torch.Generator().manual_seed(7)
        weights = torch.randn(26, 4, 4, generator=generator, dtype=torch.float64) / 4
        transfer = (
            PPStateTransfer(pp, torch.cpu, states, max_chunk_bytes=16)
            if mode == "handoff"
            else None
        )
        cuts = [23, 23, 24, 24, 23, 24, 23, 24, 23, 23]
        oldcut = cuts[0]
        mismatches = 0
        for batch, cut in enumerate(cuts):
            if transfer is not None:
                before = {
                    layer: {key: raw.clone() for key, raw in parts.items()}
                    for layer, parts in states.items()
                }
                transfer.move(oldcut, cut, batch)
                changed = range(min(oldcut, cut), max(oldcut, cut))
                for layer, parts in states.items():
                    expected = reference[layer] if layer in changed else before[layer]
                    for key, raw in parts.items():
                        torch.testing.assert_close(raw, expected[key], rtol=0, atol=0)
            original = torch.arange(4, dtype=torch.float64) / 8 + batch / 32
            if rank == 0:
                value = original.clone()
                for layer in range(cut):
                    value = _stateful_layer(
                        value, states[layer], weights[layer], layer, batch
                    )
                dist.send(value, dst=1, group=pp.device_group)
            else:
                value = torch.empty(4, dtype=torch.float64)
                dist.recv(value, src=0, group=pp.device_group)
                for layer in range(cut, 26):
                    value = _stateful_layer(
                        value, states[layer], weights[layer], layer, batch
                    )

            expected = original
            for layer in range(26):
                expected = _stateful_layer(
                    expected, reference[layer], weights[layer], layer, batch
                )
            if rank == 1:
                if transfer is not None:
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                else:
                    mismatches += not torch.equal(value, expected)
            oldcut = cut

        if transfer is not None:
            stats = transfer.stats()
            assert stats["moves"] == 6
            assert stats["bytes"] == 6 * (48 + 64 + 32 + 80)
            assert stats["chunks"] == 6 * (3 + 4 + 2 + 5)
            assert stats["last_move"]["layers"] == [23]
        elif rank == 1:
            assert mismatches > 0, "Control without handoff must lose decode history"
    finally:
        pp.destroy()
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["handoff", "no_handoff", "metadata"])
def test_cache_handoff_preserves_stateful_decode_on_real_gloo(tmp_path, mode):
    """History and raw padding survive repeated 23↔24 ownership changes."""
    torch.multiprocessing.spawn(
        _state_transfer_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", mode),
        nprocs=2,
        join=True,
    )


def _control_channel_worker(rank, rendezvous, first_two_posted, consumer_gate):
    from vllm.distributed.parallel_state import GroupCoordinator

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    pp = GroupCoordinator([[0, 1]], rank, "gloo", False, group_name="pp_control")
    try:
        store = pp.cpu_group.get_group_store()
        control = PPControlChannel(store)
        if rank == 0:
            buffer = PPSendBuffer(2)
            output = torch.empty(4, dtype=torch.int64)
            buffer.reserve({"hidden_states": output})
            for batch in range(12):
                buffer.acquire()
                control.send(PPBoundary(batch, 23 + batch % 2, batch))
                output.fill_(batch)

                def send(value):
                    handle = dist.isend(
                        value["hidden_states"], 1, group=pp.device_group
                    )
                    assert handle is not None
                    return [handle]

                buffer.send({"hidden_states": output}, send)
                output.fill_(-1)
                if batch == 1:
                    # The parent has not opened the consumer gate yet.
                    assert store.check(
                        [f"{CONTROL_KEY_PREFIX}{index}" for index in (0, 1)]
                    )
                    first_two_posted.set()
            buffer.drain()
        else:
            assert consumer_gate.wait(30), "consumer gate was not opened"
            for batch in range(12):
                header = control.receive(batch)
                assert header == PPBoundary(batch, 23 + batch % 2, batch)
                assert not store.check([f"{CONTROL_KEY_PREFIX}{batch}"])
                output = torch.empty(4, dtype=torch.int64)
                dist.recv(output, 0, group=pp.device_group)
                torch.testing.assert_close(
                    output, torch.full_like(output, batch), rtol=0, atol=0
                )
        dist.barrier()
        assert all(
            not store.check([f"{CONTROL_KEY_PREFIX}{batch}"]) for batch in range(12)
        ), "Consumed control headers must not accumulate in the store"
    finally:
        pp.destroy()
        dist.destroy_process_group()


def test_control_store_publishes_two_batches_before_consumer_on_real_gloo(tmp_path):
    """The activation ring bounds headers without a per-batch CPU rendezvous."""
    context = torch.multiprocessing.get_context("spawn")
    first_two_posted, consumer_gate = context.Event(), context.Event()
    workers = torch.multiprocessing.spawn(
        _control_channel_worker,
        args=(
            f"file://{tmp_path / 'rendezvous'}",
            first_two_posted,
            consumer_gate,
        ),
        nprocs=2,
        join=False,
    )
    try:
        posted_before_consumer = first_two_posted.wait(30)
    finally:
        consumer_gate.set()
        while not workers.join():
            pass
    assert posted_before_consumer, (
        "producer waited for the consumer before publishing two headers"
    )


def test_control_file_store_rejects_duplicate_and_out_of_order_headers(tmp_path):
    store = dist.FileStore(str(tmp_path / "control"), 1)
    producer, consumer = PPControlChannel(store), PPControlChannel(store)
    first = PPBoundary(0, 23, 1)
    producer.send(first)
    with pytest.raises(ValueError, match="already published"):
        PPControlChannel(store).send(first)
    with pytest.raises(ValueError, match="out of order"):
        producer.send(first)
    assert consumer.receive(0) == first
    assert not store.check([f"{CONTROL_KEY_PREFIX}0"])
    with pytest.raises(ValueError, match="out of order"):
        producer.send(first)
    with pytest.raises(ValueError, match="out of order"):
        consumer.receive(0)
    with pytest.raises(ValueError, match="out of order"):
        producer.send(PPBoundary(2, 23, 1))
    with pytest.raises(ValueError, match="version"):
        producer.send(PPBoundary(1, 23, 0))
    with pytest.raises(ValueError, match="version"):
        producer.send(PPBoundary(1, 24, 1))
    second = PPBoundary(1, 24, 2)
    producer.send(second)
    assert consumer.receive(1) == second
    assert not store.check([f"{CONTROL_KEY_PREFIX}1"])


def _prefix_cache_views():
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
        create_kv_cache_views,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    spec = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float64,
        page_size_padded=128,
    )
    names = ["model.layers.23.attn", "model.layers.24.attn"]
    size = 16 * 2 * spec.page_size_bytes
    tensor = KVCacheTensor(size, names, spec.page_size_bytes, 2 * spec.page_size_bytes)
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[KVCacheGroupSpec(names, spec)],
        hash_block_size=4,
        cache_hit_alignment_tokens=4,
    )
    raw = torch.zeros(size, dtype=torch.int8)
    views = create_kv_cache_views(raw, spec, 16, KVCacheLayout.BLHNC, tensor)
    return config, dict(zip(names, views)), raw


def test_isolated_movable_pages_keep_ids_padding_and_copy_on_write():
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker.pp.cache import isolate_pp_cache
    from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

    config, caches, pooled = _prefix_cache_views()
    untouched = caches["model.layers.24.attn"]
    state, description = isolate_pp_cache(
        config,
        caches,
        {"model.layers.23.attn": (23, "compressed")},
        KVCacheLayout.BLHNC,
        [4],
        torch.device("cpu"),
    )
    isolated = state[23]["compressed"]
    assert isolated.nbytes == 16 * 128
    assert isolated.data_ptr() != pooled.data_ptr()
    isolated.view(16, 128)[5].fill_(37)
    copy_kv_cache_blocks_inplace(caches.values(), 16, [KVCacheBlockCopy(5, 7)])
    torch.testing.assert_close(
        isolated.view(16, 128)[7], isolated.view(16, 128)[5], rtol=0, atol=0
    )
    assert caches["model.layers.24.attn"] is untouched
    assert not torch.count_nonzero(untouched)
    assert description[23]["compressed"][0][:3] == (
        0,
        repr(config.kv_cache_groups[0].kv_cache_spec),
        16,
    )


def _prefix_handoff_worker(rank, rendezvous):
    from vllm import SamplingParams
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.request import Request
    from vllm.v1.worker.pp.cache import isolate_pp_cache

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    pp = GroupCoordinator([[0, 1]], rank, "gloo", False, group_name="prefix")
    try:
        config, caches, _ = _prefix_cache_views()
        names = {name: (23 + i, "compressed") for i, name in enumerate(caches)}
        state, description = isolate_pp_cache(
            config, caches, names, KVCacheLayout.BLHNC, [4], torch.device("cpu")
        )
        # Reject matching byte counts with different physical block meanings.
        bad = dict(description)
        if rank:
            original = description[23]["compressed"]
            bad[23] = {"compressed": ((99, *original[0][1:]), *original[1:])}
        with pytest.raises(ValueError, match="block layout"):
            PPStateTransfer(pp, torch.cpu, state, cache_layout=bad)
        transfer = PPStateTransfer(pp, torch.cpu, state, cache_layout=description)
        manager = KVCacheManager(
            config,
            max_model_len=64,
            enable_caching=True,
            hash_block_size=4,
            scheduler_block_size=4,
        )
        init_none_hash(sha256)

        def request(name, tokens):
            return Request(
                request_id=name,
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_hasher=get_request_block_hasher(4, sha256),
            )

        tokens = list(range(9))
        original = request("warm", tokens)
        manager.allocate_slots(original, len(tokens))
        blocks = manager.get_blocks(original.request_id).get_block_ids()[0]
        original.num_computed_tokens = len(tokens)
        manager.cache_blocks(original, len(tokens))
        for layer, owner in ((23, 1), (24, 1)):
            if rank == owner:
                for logical, block in enumerate(blocks):
                    caches[f"model.layers.{layer}.attn"][block].fill_(logical + layer)
                    # State padding is part of the handoff, too.
                    state[layer]["compressed"].view(16, 128)[block, 64:].fill_(67)
        manager.free(original)
        assert all(block.ref_cnt == 0 for block in manager.block_pool.blocks[1:])
        transfer.move(23, 24, 0)
        shared = []
        for name in ("shared-a", "shared-b"):
            hit = request(name, tokens[:8] + [99])
            shared.append(hit)
            computed, count, _ = manager.get_computed_blocks(hit)
            assert count == 8
            ids = computed.get_block_ids()[0]
            assert ids == blocks[:2]
            manager.allocate_slots(hit, 1, count, computed)
            if rank == 0:
                for logical, block in enumerate(ids):
                    assert torch.all(
                        caches["model.layers.23.attn"][block] == logical + 23
                    )
                    assert torch.all(
                        state[23]["compressed"].view(16, 128)[block, 64:] == 67
                    )
        # Updates by the new owner must return with the layer on a later move.
        if rank == 0:
            caches["model.layers.23.attn"][blocks[0]].fill_(91)
        transfer.move(24, 23, 1)
        if rank == 1:
            assert torch.all(caches["model.layers.23.attn"][blocks[0]] == 91)
        for hit in shared:
            manager.free(hit)
        assert manager.reset_prefix_cache()
        _, count, _ = manager.get_computed_blocks(request("old", tokens))
        assert count == 0

        # Fill the entire pool so previously cached physical IDs are reused.
        replacement = request("replacement", list(range(100, 160)))
        assert manager.allocate_slots(replacement, 60) is not None
        reused = manager.get_blocks(replacement.request_id).get_block_ids()[0]
        assert set(blocks) <= set(reused)
        if rank == 1:
            for logical, block in enumerate(reused):
                caches["model.layers.23.attn"][block].fill_(logical + 100)
        replacement.num_computed_tokens = 60
        manager.cache_blocks(replacement, 60)
        manager.free(replacement)
        transfer.move(23, 24, 2)
        new_hit = request("new-hit", list(range(100, 160)))
        computed, count, _ = manager.get_computed_blocks(new_hit)
        assert count == 56
        if rank == 0:
            for logical, block in enumerate(computed.get_block_ids()[0]):
                assert torch.all(caches["model.layers.23.attn"][block] == logical + 100)
        _, count, _ = manager.get_computed_blocks(request("old-again", tokens))
        assert count == 0
        assert transfer.stats()["moves"] == 3
    finally:
        pp.destroy()
        dist.destroy_process_group()


def test_completed_and_shared_prefixes_survive_bidirectional_handoff(tmp_path):
    """Real prefix-manager hits read bytes retained after their request ended."""
    torch.multiprocessing.spawn(
        _prefix_handoff_worker,
        args=(f"file://{tmp_path / 'prefix'}",),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        ((2, 23, 1), "expected batch"),
        ((1, 23, 0), "version"),
        ((1, 24, 1), "version"),
        ((1, 0, 2), "boundary is invalid"),
        ((1, 23, "2"), "Invalid dynamic PP batch header"),
    ],
)
def test_control_file_store_rejects_malformed_peer_epochs(tmp_path, wire, message):
    store = dist.FileStore(str(tmp_path / "control"), 1)
    producer, consumer = PPControlChannel(store), PPControlChannel(store)
    producer.send(PPBoundary(0, 23, 1))
    consumer.receive(0)
    key = f"{CONTROL_KEY_PREFIX}1"
    store.set(key, json.dumps(wire))
    with pytest.raises(ValueError, match=message):
        consumer.receive(1)
    assert store.check([key]), "Rejected headers must not advance consumption"
