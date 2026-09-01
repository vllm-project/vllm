# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
import threading

import numpy as np
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
from vllm.multimodal.paged_shm.types import (
    PagedShmCacheOutItem,
    ShmWriteRequest,
)
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
    server = PagedShmServerProc(size=1024 * 1024 * 10, block_size=4096, debug=True)
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


def _compare_items(item1: PagedShmCacheOutItem, item2: PagedShmCacheOutItem) -> bool:
    """Compare two PagedShmCacheOutItem objects, focusing on data."""
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


def _make_test_item() -> PagedShmCacheOutItem:
    # Large tensor (~256KB) to trigger multi-chunk, but within 1MB server limit
    large_tensor = torch.randn(256, 256, dtype=torch.float32)  # ~256KB
    e1 = MultiModalFieldElem(large_tensor, MultiModalBatchedField())
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
    return PagedShmCacheOutItem(kwargs_item=kwargs_item, prompt_updates=[update])


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestEncode:
    """Tests for encode_item function."""

    def test_encode_item_basic(self):
        encoder = MsgpackEncoder(size_threshold=4096)
        item = _make_test_item()
        result = encode_item((item.kwargs_item, item.prompt_updates), encoder)
        assert result is not None, "Expected multi-chunk encoding"
        chunks, lengths = result

        # First chunk is the metadata chunk
        meta = chunks[0]
        assert isinstance(meta, bytes)

        # Parse 8-byte header
        meta_size = struct.unpack("<I", meta[:4])[0]
        total_chunks = struct.unpack("<I", meta[4:8])[0]
        assert total_chunks == len(chunks)

        # Parse per-data-chunk metadata entries
        offset = 8
        stored_lengths = []
        for _ in range(total_chunks - 1):  # data chunks
            stored_len = struct.unpack("<I", meta[offset : offset + 4])[0]
            offset += 4
            flag = struct.unpack("<B", meta[offset : offset + 1])[0]
            offset += 1
            stored_lengths.append(stored_len)
            assert flag in (0, 1)

        # Verify metadata chunk size
        assert meta_size == len(meta)

        # Verify that chunk lengths match metadata
        assert lengths[0] == meta_size
        for i in range(1, total_chunks):
            assert lengths[i] == stored_lengths[i - 1]

        # Verify actual sizes
        for i, ch in enumerate(chunks):
            actual_size = (
                ch.nbytes if isinstance(ch, (torch.Tensor, np.ndarray)) else len(ch)
            )
            assert actual_size == lengths[i]

    def test_encode_item_empty(self):
        encoder = MsgpackEncoder(size_threshold=4096)
        result = encode_item((MultiModalKwargsItem({}), []), encoder)
        # Empty item may result in one chunk or None; either is acceptable.
        if result is not None:
            chunks, _ = result
            assert len(chunks) == 1

    def test_encode_item_single_chunk(self):
        small_tensor = torch.randn(10, 10)
        e = MultiModalFieldElem(small_tensor, MultiModalBatchedField())
        kwargs = MultiModalKwargsItem({"x": e})
        encoder = MsgpackEncoder(size_threshold=4096)
        result = encode_item((kwargs, []), encoder)
        # Single chunk (no large tensor) should return None.
        assert result is None, "Should return None for single chunk"


class TestIntegration:
    """Integration tests using client to allocate blocks."""

    def _write_and_get_token(
        self, client, item: PagedShmCacheOutItem, encoder, timeout=10.0
    ):
        storage = client._storage
        block_size = storage.block_size

        result = encode_item((item.kwargs_item, item.prompt_updates), encoder)
        if result is None:
            raise RuntimeError("Test item did not produce multi-chunk encoding")
        chunks, lengths = result

        total_blocks = 0
        for length in lengths:
            total_blocks += (length + block_size - 1) // block_size

        mm_hash = _unique_uuid()
        req = ShmWriteRequest(
            uuid=mm_hash,
            size=total_blocks * block_size,
            use_cache=True,
            generate_read_token=True,
        )
        alloc = client.open_write([req], timeout=timeout)[0]
        write_encoded_to_blocks(storage, chunks, alloc.blocks)
        client.close_write(mm_hash)
        return alloc.read_token

    def test_write_read_roundtrip(self, client):
        encoder = MsgpackEncoder(size_threshold=client._block_size)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)
        original_item = _make_test_item()

        token = self._write_and_get_token(client, original_item, encoder)

        read_alloc = client.open_read(token, timeout=5.0)
        decoded_kwargs, decoded_updates = read_decoded_from_blocks(
            client._storage, read_alloc.blocks, client._block_size, decoder
        )
        decoded_item = PagedShmCacheOutItem(
            kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
        )
        assert _compare_items(original_item, decoded_item)
        client.close_read(token)

    def test_write_read_large_data(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)

        large_tensor = torch.randn(4096 + 100, dtype=torch.float32)  # ~16KB
        e1 = MultiModalFieldElem(large_tensor, MultiModalBatchedField())
        kwargs_item = MultiModalKwargsItem({"x": e1})
        original_item = PagedShmCacheOutItem(kwargs_item=kwargs_item, prompt_updates=[])

        token = self._write_and_get_token(client, original_item, encoder)

        read_alloc = client.open_read(token, timeout=5.0)
        decoded_kwargs, decoded_updates = read_decoded_from_blocks(
            client._storage, read_alloc.blocks, block_size, decoder
        )
        decoded_item = PagedShmCacheOutItem(
            kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
        )
        assert _compare_items(original_item, decoded_item)
        client.close_read(token)

    def test_write_read_with_prompt_updates(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)

        update = ResolvedPromptUpdate(
            modality="audio",
            item_idx=5,
            mode=UpdateMode.REPLACE,
            target=[10, 20, 30],
            content=PromptUpdateDetails.from_seq([100, 200, 300]),
        )
        large_tensor = torch.randn(256, 256, dtype=torch.float32)  # ~256KB
        e = MultiModalFieldElem(large_tensor, MultiModalBatchedField())
        kwargs_item = MultiModalKwargsItem({"a": e})
        original_item = PagedShmCacheOutItem(
            kwargs_item=kwargs_item, prompt_updates=[update]
        )

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

        decoded_item = PagedShmCacheOutItem(
            kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
        )
        assert _compare_items(original_item, decoded_item)
        client.close_read(token)

    def test_concurrent_readers(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)
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
                decoded_item = PagedShmCacheOutItem(
                    kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
                )
                results.append(_compare_items(original_item, decoded_item))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results), f"Some readers got corrupted data: results={results}"
        assert not errors, f"Reader errors: {errors}"
        client.close_read(token)

    def test_read_token_reusable(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)
        original_item = _make_test_item()

        token = self._write_and_get_token(client, original_item, encoder)

        for _ in range(2):
            read_alloc = client.open_read(token, timeout=5.0)
            decoded_kwargs, decoded_updates = read_decoded_from_blocks(
                client._storage, read_alloc.blocks, block_size, decoder
            )
            decoded_item = PagedShmCacheOutItem(
                kwargs_item=decoded_kwargs, prompt_updates=decoded_updates
            )
            assert _compare_items(original_item, decoded_item)

        client.close_read(token)

    def test_read_nonexistent_blocks_fails(self, client):
        decoder = MsgpackDecoder(PagedShmCacheOutItem)
        with pytest.raises(ValueError, match="Blocks list cannot be empty"):
            read_decoded_from_blocks(client._storage, [], client._block_size, decoder)

    def test_write_empty_blocks_fails(self, client):
        storage = client._storage
        chunks = [b"dummy"]
        with pytest.raises(ValueError, match="Blocks list cannot be empty"):
            write_encoded_to_blocks(storage, chunks, [])

    def test_type_flag_restoration(self, client):
        """Test that type flags correctly restore bytes vs tensor."""
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)

        large_tensor = torch.randn(1024, dtype=torch.float32)
        small_tensor = torch.randn(10, 10, dtype=torch.float32)
        e1 = MultiModalFieldElem(large_tensor, MultiModalBatchedField())
        e2 = MultiModalFieldElem(
            small_tensor, MultiModalFlatField(slices=[slice(1, 3)], dim=0)
        )
        kwargs_item = MultiModalKwargsItem({"t1": e1, "t2": e2})
        original_item = PagedShmCacheOutItem(kwargs_item=kwargs_item, prompt_updates=[])

        result = encode_item(
            (original_item.kwargs_item, original_item.prompt_updates), encoder
        )
        assert result is not None
        chunks, lengths = result

        total_blocks = sum(
            (length + block_size - 1) // block_size for length in lengths
        )
        mm_hash = _unique_uuid()
        req = ShmWriteRequest(
            uuid=mm_hash,
            size=total_blocks * block_size,
            use_cache=True,
            generate_read_token=True,
        )
        alloc = client.open_write([req], timeout=10.0)[0]
        write_encoded_to_blocks(client._storage, chunks, alloc.blocks)
        client.close_write(mm_hash)

        read_alloc = client.open_read(alloc.read_token, timeout=5.0)
        decoded_kwargs, _ = read_decoded_from_blocks(
            client._storage, read_alloc.blocks, block_size, decoder
        )
        client.close_read(alloc.read_token)

        decoded_item = PagedShmCacheOutItem(
            kwargs_item=decoded_kwargs, prompt_updates=[]
        )
        assert _compare_items(original_item, decoded_item)

    def test_metadata_chunk_cross_blocks(self, client):
        """
        Test that metadata chunk can span multiple blocks when it's large.
        Create many medium-sized tensor chunks to force a large metadata chunk.
        Each tensor is slightly above the encoder's size_threshold, so each
        becomes a separate data chunk.
        """
        block_size = client._block_size
        threshold = 1024
        encoder = MsgpackEncoder(size_threshold=threshold)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)

        # Create many data chunks, enough to make metadata chunk > block_size
        num_tensors = (
            1000  # This many tensors, but encoder may produce more/less data chunks
        )
        data_dict = {}
        for i in range(num_tensors):
            data_dict[f"t{i}"] = MultiModalFieldElem(
                torch.randn(257, dtype=torch.float32), MultiModalBatchedField()
            )
        kwargs_item = MultiModalKwargsItem(data_dict)
        original_item = PagedShmCacheOutItem(kwargs_item=kwargs_item, prompt_updates=[])

        result = encode_item(
            (original_item.kwargs_item, original_item.prompt_updates), encoder
        )
        assert result is not None
        chunks, lengths = result

        # Verify metadata chunk size > block_size
        meta_chunk = chunks[0]
        assert len(meta_chunk) > block_size, (
            f"Metadata chunk size {len(meta_chunk)} "
            f"should exceed block size {block_size}"
        )

        # Parse header to verify total_chunks == len(chunks)
        meta_size = struct.unpack("<I", meta_chunk[:4])[0]
        total_chunks = struct.unpack("<I", meta_chunk[4:8])[0]
        assert total_chunks == len(chunks)
        assert meta_size == len(meta_chunk)

        # Calculate total blocks needed
        total_blocks = sum(
            (length + block_size - 1) // block_size for length in lengths
        )

        # Write and read back
        mm_hash = _unique_uuid()
        req = ShmWriteRequest(
            uuid=mm_hash,
            size=total_blocks * block_size,
            use_cache=True,
            generate_read_token=True,
        )
        alloc = client.open_write([req], timeout=10.0)[0]
        write_encoded_to_blocks(client._storage, chunks, alloc.blocks)
        client.close_write(mm_hash)

        read_alloc = client.open_read(alloc.read_token, timeout=5.0)
        decoded_kwargs, _ = read_decoded_from_blocks(
            client._storage, read_alloc.blocks, block_size, decoder
        )
        client.close_read(alloc.read_token)

        decoded_item = PagedShmCacheOutItem(
            kwargs_item=decoded_kwargs, prompt_updates=[]
        )
        assert _compare_items(original_item, decoded_item), (
            "Data mismatch for large metadata chunk"
        )


class TestErrorHandling:
    """Tests for error conditions."""

    def test_insufficient_blocks_raises(self, client):
        storage = client._storage
        block_size = storage.block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        item = _make_test_item()
        result = encode_item((item.kwargs_item, item.prompt_updates), encoder)
        assert result is not None
        chunks, _ = result

        with pytest.raises(ValueError, match="Not enough blocks for chunk"):
            write_encoded_to_blocks(storage, chunks, [0])

    def test_decode_insufficient_blocks_raises(self, client):
        storage = client._storage
        block_size = storage.block_size

        # Construct a metadata chunk with one data chunk whose length requires
        # more blocks than available.
        data_length = 10000  # > block_size, needs multiple blocks
        meta_body = struct.pack("<I", data_length) + struct.pack("<B", 0)  # type 0
        meta_size = 8 + len(meta_body)
        total_chunks = 2  # metadata + 1 data
        header = struct.pack("<I", meta_size) + struct.pack("<I", total_chunks)
        meta_chunk = header + meta_body

        # Write the metadata chunk into block 0 (it fits in one block)
        storage.write(meta_chunk, [0])

        # Attempt to read with only block 0 available.
        # The metadata chunk can be read, but the data chunk requires more blocks.
        with pytest.raises(ValueError, match="Insufficient blocks for data chunk"):
            read_decoded_from_blocks(
                storage, [0], block_size, MsgpackDecoder(PagedShmCacheOutItem)
            )


@pytest.mark.slow
class TestStress:
    """Stress tests."""

    def test_stress_many_writes_reads(self, client):
        block_size = client._block_size
        encoder = MsgpackEncoder(size_threshold=block_size)
        decoder = MsgpackDecoder(PagedShmCacheOutItem)
        num_items = 20

        tokens = []
        for i in range(num_items):
            tensor = torch.full((256, 256), i, dtype=torch.float32)
            e = MultiModalFieldElem(tensor, MultiModalBatchedField())
            kwargs_item = MultiModalKwargsItem({"v": e})
            item = PagedShmCacheOutItem(kwargs_item=kwargs_item, prompt_updates=[])
            result = encode_item((item.kwargs_item, item.prompt_updates), encoder)
            assert result is not None
            chunks, lengths = result
            total_blocks = sum(
                (length + block_size - 1) // block_size for length in lengths
            )
            mm_hash = _unique_uuid()
            req = ShmWriteRequest(
                uuid=mm_hash,
                size=total_blocks * block_size,
                use_cache=True,
                generate_read_token=True,
            )
            alloc = client.open_write([req], timeout=10.0)[0]
            write_encoded_to_blocks(client._storage, chunks, alloc.blocks)
            client.close_write(mm_hash)
            tokens.append((alloc.read_token, i))

        for token, expected_idx in tokens:
            read_alloc = client.open_read(token, timeout=5.0)
            decoded_kwargs, _ = read_decoded_from_blocks(
                client._storage, read_alloc.blocks, block_size, decoder
            )
            data_dict = decoded_kwargs.data
            tensor = data_dict["v"].data
            expected = torch.full((256, 256), expected_idx, dtype=torch.float32)
            assert torch.equal(tensor, expected)
            client.close_read(token)
