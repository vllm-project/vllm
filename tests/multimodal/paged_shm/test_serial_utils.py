# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
import threading

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalSharedField,
)
from vllm.multimodal.paged_shm.client import PagedShmClient
from vllm.multimodal.paged_shm.serial_utils import (
    encode_item,
    read_decoded_from_blocks,
    write_encoded_to_blocks,
)
from vllm.multimodal.paged_shm.server import PagedShmServerProc
from vllm.multimodal.paged_shm.types import ShmItem, ShmWriteRequest
from vllm.multimodal.processing.processor import (
    PromptUpdateDetails,
    ResolvedPromptUpdate,
    UpdateMode,
)
from vllm.utils import random_uuid
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def server():
    server = PagedShmServerProc(size=1024 * 1024, block_size=4096, debug=True)
    server.start()
    yield server
    server.shutdown()


@pytest.fixture(scope="function")
def client(server):
    c = PagedShmClient(address=server.address, pin=False)
    c.debug_cleanup()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_uuid() -> str:
    return f"test-{random_uuid()}"


def _field_data_equal(elem1: MultiModalFieldElem, elem2: MultiModalFieldElem) -> bool:
    """Compare only the tensor data of two MultiModalFieldElem objects."""
    return torch.equal(elem1.data, elem2.data)


def _kwargs_item_equal(
    item1: MultiModalKwargsItem, item2: MultiModalKwargsItem
) -> bool:
    """Compare two MultiModalKwargsItem by their data (ignore field config)."""
    if set(item1.keys()) != set(item2.keys()):
        return False
    return all(_field_data_equal(item1[key], item2[key]) for key in item1)


def _compare_items(item1: ShmItem, item2: ShmItem) -> bool:
    """Compare two ShmItem objects, focusing on data, ignoring field config."""
    if not _kwargs_item_equal(item1.kwargs_item, item2.kwargs_item):
        return False
    if len(item1.prompt_updates) != len(item2.prompt_updates):
        return False
    for u1, u2 in zip(item1.prompt_updates, item2.prompt_updates):
        if u1.modality != u2.modality:
            return False
        if u1.item_idx != u2.item_idx:
            return False
        if u1.mode != u2.mode:
            return False
        if isinstance(u1.target, list) and isinstance(u2.target, list):
            if u1.target != u2.target:
                return False
        elif isinstance(u1.target, list) or isinstance(u2.target, list):
            return False
        if u1.content != u2.content:
            return False
    return True


def _make_test_item() -> ShmItem:
    """Create a representative ShmItem for testing."""
    e1 = MultiModalFieldElem(
        torch.randn(10, 10, dtype=torch.float32),
        MultiModalBatchedField(),
    )
    e2 = MultiModalFieldElem(
        torch.randint(0, 256, (5, 5), dtype=torch.int64),
        MultiModalFlatField(slices=[slice(1, 3), slice(2, 4)], dim=0),
    )
    e3 = MultiModalFieldElem(
        torch.zeros(100, dtype=torch.bfloat16),
        MultiModalSharedField(batch_size=2),
    )
    kwargs_item = MultiModalKwargsItem(
        {
            "img1": e1,
            "img2": e2,
            "txt": e3,
        }
    )
    update = ResolvedPromptUpdate(
        modality="image",
        item_idx=0,
        mode=UpdateMode.INSERT,
        target=[1, 2, 3],
        content=PromptUpdateDetails.from_seq([4, 5, 6]),
    )
    return ShmItem(kwargs_item=kwargs_item, prompt_updates=[update])


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestEncode:
    """Tests for encode_item function."""

    def test_encode_item_basic(self):
        block_size = 4096
        encoder = MsgpackEncoder(size_threshold=block_size)
        item = _make_test_item()
        meta_block_data, chunks, chunk_lengths = encode_item(item, block_size, encoder)

        assert len(meta_block_data) == block_size

        num_chunks = struct.unpack("<I", meta_block_data[:4])[0]
        assert num_chunks == len(chunks) == len(chunk_lengths)

        offset = 4
        for length in chunk_lengths:
            stored_len = struct.unpack("<I", meta_block_data[offset : offset + 4])[0]
            assert stored_len == length
            offset += 4

        assert sum(chunk_lengths) == sum(len(c) for c in chunks)

    def test_encode_item_empty(self):
        block_size = 4096
        encoder = MsgpackEncoder(size_threshold=block_size)
        item = ShmItem(kwargs_item=MultiModalKwargsItem({}), prompt_updates=[])
        meta_block_data, chunks, chunk_lengths = encode_item(item, block_size, encoder)

        assert len(meta_block_data) == block_size
        num_chunks = struct.unpack("<I", meta_block_data[:4])[0]
        assert num_chunks == len(chunks) == len(chunk_lengths)
        assert sum(chunk_lengths) == sum(len(c) for c in chunks)


class TestIntegration:
    """Integration tests using client to allocate blocks,
    but serial_utils only uses storage."""

    def _write_and_get_token(self, client, item, encoder):
        storage = client._storage
        block_size = storage.block_size
        meta_block_data, chunks, _ = encode_item(item, block_size, encoder)
        total_blocks = 1 + sum((len(c) + block_size - 1) // block_size for c in chunks)
        mm_hash = _unique_uuid()
        req = ShmWriteRequest(
            uuid=mm_hash,
            size=total_blocks * block_size,
            use_cache=True,
            generate_read_token=True,
        )
        alloc = client.open_write([req], timeout=5.0)[0]
        write_encoded_to_blocks(storage, meta_block_data, chunks, alloc.blocks)
        client.close_write(mm_hash)
        return alloc.read_token

    def test_write_read_roundtrip(self, client):
        encoder = MsgpackEncoder(size_threshold=client._block_size)
        decoder = MsgpackDecoder(ShmItem)
        original_item = _make_test_item()

        token = self._write_and_get_token(client, original_item, encoder)

        read_alloc = client.open_read(token, timeout=5.0)
        decoded_kwargs, decoded_updates = read_decoded_from_blocks(
            client._storage, read_alloc.blocks, client._block_size, decoder
        )
        decoded_item = ShmItem(
            kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
        )
        assert _compare_items(original_item, decoded_item)
        client.close_read(token)

    def test_write_read_large_data(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(ShmItem)

        large_tensor = torch.randn(4096 * 2 + 100, dtype=torch.float32)
        e1 = MultiModalFieldElem(large_tensor, MultiModalBatchedField())
        kwargs_item = MultiModalKwargsItem({"x": e1})
        original_item = ShmItem(kwargs_item=kwargs_item, prompt_updates=[])

        token = self._write_and_get_token(client, original_item, encoder)

        read_alloc = client.open_read(token, timeout=5.0)
        decoded_kwargs, decoded_updates = read_decoded_from_blocks(
            client._storage, read_alloc.blocks, block_size, decoder
        )
        decoded_item = ShmItem(
            kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
        )
        assert _compare_items(original_item, decoded_item)
        client.close_read(token)

    def test_write_read_with_prompt_updates(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(ShmItem)

        update = ResolvedPromptUpdate(
            modality="audio",
            item_idx=5,
            mode=UpdateMode.REPLACE,
            target=[10, 20, 30],
            content=PromptUpdateDetails.from_seq([100, 200, 300]),
        )
        kwargs_item = MultiModalKwargsItem(
            {
                "a": MultiModalFieldElem(
                    torch.tensor([1, 2, 3]), MultiModalBatchedField()
                )
            }
        )
        original_item = ShmItem(kwargs_item=kwargs_item, prompt_updates=[update])

        token = self._write_and_get_token(client, original_item, encoder)

        read_alloc = client.open_read(token, timeout=5.0)
        decoded_kwargs, decoded_updates = read_decoded_from_blocks(
            client._storage, read_alloc.blocks, block_size, decoder
        )
        assert len(decoded_updates) == 1
        du = decoded_updates[0]
        assert du.modality == update.modality
        assert du.item_idx == update.item_idx
        assert du.mode == update.mode
        assert du.target == update.target
        assert du.content == update.content

        decoded_item = ShmItem(
            kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
        )
        assert _compare_items(original_item, decoded_item)
        client.close_read(token)

    def test_concurrent_readers(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(ShmItem)
        original_item = _make_test_item()

        token = self._write_and_get_token(client, original_item, encoder)

        results = []
        errors = []

        def reader():
            try:
                read_alloc = client.open_read(token, timeout=5.0)
                decoded_kwargs, decoded_updates = read_decoded_from_blocks(
                    client._storage, read_alloc.blocks, block_size, decoder
                )
                decoded_item = ShmItem(
                    kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
                )
                results.append(_compare_items(original_item, decoded_item))
                # Do NOT close token here; will be closed by main thread
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results), f"Some readers got corrupted data: results={results}"
        assert not errors, f"Reader errors: {errors}"

        # Close token once after all readers finish
        client.close_read(token)

    def test_read_token_reusable(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(ShmItem)
        original_item = _make_test_item()

        token = self._write_and_get_token(client, original_item, encoder)

        # Read twice without closing in between
        for _ in range(2):
            read_alloc = client.open_read(token, timeout=5.0)
            decoded_kwargs, decoded_updates = read_decoded_from_blocks(
                client._storage, read_alloc.blocks, block_size, decoder
            )
            decoded_item = ShmItem(
                kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
            )
            assert _compare_items(original_item, decoded_item)
            # Do not close here

        # Close once after all reads
        client.close_read(token)

    def test_read_nonexistent_blocks_fails(self, client):
        decoder = MsgpackDecoder(ShmItem)
        with pytest.raises(ValueError):
            read_decoded_from_blocks(client._storage, [], client._block_size, decoder)

    def test_write_empty_blocks_fails(self, client):
        with pytest.raises(ValueError):
            write_encoded_to_blocks(client._storage, b"", (), [])


class TestErrorHandling:
    """Tests for error conditions."""

    def test_insufficient_blocks_raises(self, client):
        storage = client._storage
        block_size = storage.block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        item = _make_test_item()
        meta_block_data, chunks, _ = encode_item(item, block_size, encoder)
        with pytest.raises(ValueError, match="Insufficient blocks"):
            write_encoded_to_blocks(storage, meta_block_data, chunks, [0])

    def test_decode_corrupted_meta(self, client):
        storage = client._storage
        block_size = storage.block_size

        # Write meta: num_chunks=1, length=block_size*10 (too large)
        meta_data = struct.pack("<II", 1, block_size * 10)
        meta_block = bytearray(block_size)
        meta_block[: len(meta_data)] = meta_data
        storage.write(bytes(meta_block), [0])

        # Write dummy data to block 1 (so we have at least one data block)
        dummy = b"\x00" * block_size
        storage.write(dummy, [1])

        decoder = MsgpackDecoder(ShmItem)
        # read_decoded_from_blocks will see that chunk length needs 10 blocks
        # but we only have 1 data block,
        # so it raises ValueError with "Insufficient data blocks"
        with pytest.raises(ValueError, match="Insufficient data blocks"):
            read_decoded_from_blocks(storage, [0, 1], block_size, decoder)


@pytest.mark.slow
class TestStress:
    """Stress tests."""

    def test_stress_many_writes_reads(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(ShmItem)
        num_items = 20

        tokens = []
        for i in range(num_items):
            tensor = torch.full((10,), i, dtype=torch.float32)
            e = MultiModalFieldElem(tensor, MultiModalBatchedField())
            kwargs_item = MultiModalKwargsItem({"v": e})
            item = ShmItem(kwargs_item=kwargs_item, prompt_updates=[])
            meta, chunks, _ = encode_item(item, block_size, encoder)
            total_blocks = 1 + sum(
                (len(c) + block_size - 1) // block_size for c in chunks
            )
            mm_hash = _unique_uuid()
            req = ShmWriteRequest(
                uuid=mm_hash,
                size=total_blocks * block_size,
                use_cache=True,
                generate_read_token=True,
            )
            alloc = client.open_write([req], timeout=5.0)[0]
            write_encoded_to_blocks(client._storage, meta, chunks, alloc.blocks)
            client.close_write(mm_hash)
            tokens.append((alloc.read_token, i))

        for token, expected_idx in tokens:
            read_alloc = client.open_read(token, timeout=5.0)
            decoded_kwargs, _ = read_decoded_from_blocks(
                client._storage, read_alloc.blocks, block_size, decoder
            )
            data_dict = decoded_kwargs.data
            tensor = data_dict["v"].data
            assert torch.equal(
                tensor, torch.full((10,), expected_idx, dtype=torch.float32)
            )
            client.close_read(token)
