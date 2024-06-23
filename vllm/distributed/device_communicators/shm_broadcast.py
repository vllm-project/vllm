import itertools
import pickle
import sys
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Generator, Iterable, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL

logger = init_logger(__name__)


class _MetadataBuffer:

    def __init__(self, buffer):
        self.buffer = buffer
        self.n_reader = len(buffer) - 9
        self.written_flag = buffer[0]
        self.read_count = sum(buffer[1:self.n_reader + 1])
        self.start = int.from_bytes(
            buffer[self.n_reader + 1:self.n_reader + 5], sys.byteorder)
        self.end = int.from_bytes(buffer[self.n_reader + 5:self.n_reader + 9],
                                  sys.byteorder)

    def mark_ready_to_read(self):
        # NOTE: order is important here
        # (1) set the written flag to 0
        # (2) set the read flags to 0
        # (3) set the written flag to 1
        # otherwise, the readers may think they already read the block

        # mark the block as not written
        self.buffer[0] = 0
        for i in range(1, self.n_reader + 1):
            # set read flag to 0, meaning it is not read yet
            self.buffer[i] = 0
        # mark the block as written
        self.buffer[0] = 1

    def mark_read(self, reader_rank):
        self.buffer[reader_rank + 1] = 1

    def write_start_end(self):
        self.buffer[self.n_reader + 1:self.n_reader + 5] = self.start.to_bytes(
            4, sys.byteorder)
        self.buffer[self.n_reader + 5:self.n_reader + 9] = self.end.to_bytes(
            4, sys.byteorder)

    def can_write(self):
        # this block is either
        # (1) not written
        # (2) read by all readers
        return not self.written_flag or self.read_count == self.n_reader

    def can_read(self, reader_rank):
        return self.written_flag and not self.buffer[reader_rank + 1]


class ShmRingBuffer:

    def __init__(self,
                 n_reader: int,
                 max_bytes: int = 10 << 20,
                 max_chunks: int = 1000,
                 name: Optional[str] = None):
        """
        A shared memory ring buffer implementation for broadcast communication.
        Essentially, it is a queue where only one will `enqueue` and multiple
        will `dequeue`. The max size of each item, together with the max number
        of items that can be stored in the buffer are known in advance.
        In this case, we don't need to synchronize the access to
         the buffer.
        
        Buffer memory layout:
                  data                                 metadata
                    |                                      |
                    | [current_start, current_end)         | (current_idx)
                    v                                      v
        +-------------------------------+----------------------------------------+
        | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
        +-------------------------------+----------------------------------------+
        |         max_bytes             | max_chunks x (1 + n_reader + 8) bytes  |

        metadata memory layout:
            the first bytes are flags (set to 0 by default):
                the first byte is the written flag
                the following bytes are reader flags.
            the next 8 bytes are two int32 values, indicating the start (inclusive) and end (exclusive) of the data in the chunk.
        +--------------+--------------+--------------+-----+--------------+--------------+--------------+
        | written_flag | reader0_flag | reader1_flag | ... | readerN_flag | start_offset | end_offset   |
        +--------------+--------------+--------------+-----+--------------+--------------+--------------+

        The state of metadata (excluding the 8 bytes) is as follows:

        (case 1) 0???...???: the block is not written yet, cannot read, can write
        (case 2) 1000...000: the block is just written, can read, cannot write
        (case 3) 1???...???: the block is written and read by some readers, can read if not read, cannot write
        (case 4) 1111...111: the block is written and read by all readers, cannot read, can write

        State transition for readers:

        When a reader finds a block that it can read (case 2 or 3), it can yield the block for caller to read.
        Only after the caller finishes reading the block, the reader can mark the block as read.
        Readers only mark the block as read (from 0 to 1), the writer marks the block as ready to read (from 1 to 0).

        State transition for writer:

        When the writer writes to a block (case 1 or 4), it can yield the block for caller to write. 
        After the caller finishes writing the block, the writer first resets the written flag to 0, converting either case
        to case 1. Then it can reset the reader flags to 0, and mark the block as written (from 0 to 1).
        NOTE: the order is important here, first reset the reader flags (so that we are still in case 1), then mark the block as written. The state transition is atomic. If we do it in the reverse order, it will go through case 3 and then back to case 2, and readers might read the intermediate case 3, which is not correct.

        During creation, `name` is None and the buffer is created. We can pass the
        created object to other processes by pickling it. The other processes will
        get the name of the shared memory and open it, so that they can access the
        same shared memory buffer.
        """# noqa

        self.n_reader = n_reader
        self.flags_size = 1 + n_reader
        self.metadata_size = self.flags_size + 8
        self.max_bytes = max_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = self.max_bytes + self.metadata_size * self.max_chunks  # noqa
        self.data_offset = 0
        self.metadata_offset = self.max_bytes

        if name is None:
            # we are creating a buffer
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes_of_buffer)
            # initialize the metadata section to 0
            with memoryview(self.shared_memory.buf[self.metadata_offset:]
                            ) as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            # we are opening an existing buffer
            self.is_creator = False
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch("multiprocessing.resource_tracker.register",
                       lambda *args, **kwargs: None):
                self.shared_memory = shared_memory.SharedMemory(name=name)
            assert self.shared_memory.size == self.total_bytes_of_buffer

    def __reduce__(self):
        return (
            self.__class__,
            (self.n_reader, self.max_bytes, self.max_chunks,
             self.shared_memory.name),
        )

    def __del__(self):
        self.shared_memory.close()
        if self.is_creator:
            self.shared_memory.unlink()

    @contextmanager
    def get_data(self, start: int, end: int):
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf

    @contextmanager
    def get_metadata(
            self, current_idx: int) -> Generator[_MetadataBuffer, None, None]:
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield _MetadataBuffer(buf)


class ShmRingBufferIO:

    def __init__(self, buffer: ShmRingBuffer, reader_rank: int):
        self.buffer = buffer
        self.reader_rank = reader_rank
        self._is_writer = self.reader_rank == -1
        self._is_reader = not self._is_writer
        if self._is_reader:
            assert 0 <= self.reader_rank < buffer.n_reader, \
                (f"Invalid reader rank {self.reader_rank} for buffer"
                f" created with {buffer.n_reader} readers")
        """
        `current_idx` of the writer points to the next enqueue position.
        `current_idx` of the readers points to the next dequeue position.
        i.e.:
        for reader, current_idx is the next block to read
        for writer, current_idx is the next block to write.
        """ # noqa
        self.current_idx = 0


        """
        `low_watermark` and `high_watermark` are used for the writer to know the available space in the buffer:
        - if `low_watermark` == `high_watermark`, it is ambiguous whether the buffer is full or empty.
            We reserve the case for the buffer is empty. In this case, we can safely reset the
            `low_watermark` and `high_watermark` to 0. `[0, max_bytes - 1)` are available for writing.
        - if `low_watermark` < `high_watermark`, both the range `[high_watermark, max_bytes - 1)`
            and `[0, low_watermark -1)` are available for writing.
        0           low_watermark       high_watermark    max_bytes
                          |<--      data    -->|
        |<-- can write -->|<-- cannot write -->|<-- can write -->|

        - if `low_watermark` > `high_watermark`, the range `[high_watermark, low_watermark - 1)`
            is available for writing.
        0              high_watermark     low_watermark       max_bytes
        |<--    data      -->|                 |<--       data   -->|
        |<-- cannot write -->|<-- can write -->|<-- cannot write -->|

        The following fields are only for writer.
        `start_recycle_idx` points to the bottom of the queue

        invariant:
        just after finish writing,
        self.buffer.get_metadata(self.current_idx).end == self.high_watermark
        self.buffer.get_metadata(self.start_recycle_idx).start == self.low_watermark
        """ # noqa
        self.start_recycle_idx = 0
        self.low_watermark = 0
        self.high_watermark = 0

    def recycle_memory(self):
        # essentially this function sets `low_watermark`

        recycle_range: Iterable
        recycle_range = tuple()
        if self.start_recycle_idx < self.current_idx:
            recycle_range = range(self.start_recycle_idx, self.current_idx)
        elif self.start_recycle_idx > self.current_idx:
            recycle_range = itertools.chain(
                range(self.start_recycle_idx, self.buffer.max_chunks),
                range(0, self.current_idx))
        for i in recycle_range:
            with self.buffer.get_metadata(i) as metadata_buffer:
                if metadata_buffer.can_write():
                    self.start_recycle_idx = (self.start_recycle_idx +
                                              1) % self.buffer.max_chunks
                    self.low_watermark = metadata_buffer.end
                else:
                    self.low_watermark = metadata_buffer.start
                    break
        if self.start_recycle_idx == self.current_idx:
            # the current block can also be recycled
            self.low_watermark = 0
            self.high_watermark = 0

    @contextmanager
    def acquire_write(self, n_bytes: int = 0):
        assert self._is_writer, "Only writers can acquire write"
        assert 0 < n_bytes < self.buffer.max_bytes, \
            f"bytes acquired for write should be in (0, {self.buffer.max_bytes})"  # noqa
        start_time = time.time()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                if not metadata_buffer.can_write():
                    # for writers, `self.current_idx` is the next block to write
                    # if this block is not ready to write,
                    # we need to wait until it is read by all readers

                    # wait for a while (0.1 us)
                    time.sleep(1e-7)

                    # if we wait for a long time, we should warn the user
                    if time.time(
                    ) - start_time > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:  # noqa
                        logger.warning(
                            "No available block found in %s second. ",
                            VLLM_RINGBUFFER_WARNING_INTERVAL)
                        n_warning += 1

                    continue

                # the block is ready to write now

                # recycle memory
                self.recycle_memory()

                # find enough space to write
                start_loc = None
                if self.low_watermark == self.high_watermark:
                    # the buffer is empty
                    # reset to 0
                    start_loc = 0
                    self.low_watermark = 0
                    self.high_watermark = n_bytes
                elif self.low_watermark < self.high_watermark:
                    right_space = self.buffer.max_bytes - 1 - self.high_watermark  # noqa
                    left_space = self.low_watermark - 1
                    if left_space < n_bytes and right_space < n_bytes:
                        # not enough space
                        # wait for a while (0.1 us)
                        # readers may read the block and then
                        # we can free the space
                        time.sleep(1e-7)
                        continue
                    if right_space >= n_bytes:
                        start_loc = self.high_watermark
                        self.high_watermark += n_bytes
                    else:
                        start_loc = 0
                        self.high_watermark = n_bytes
                else:
                    space = self.low_watermark - self.high_watermark - 1
                    if space < n_bytes:
                        # not enough space
                        # wait for a while (0.1 us)
                        # readers may read the block and then
                        # we can free the space
                        time.sleep(1e-7)
                        continue
                    start_loc = self.high_watermark
                    self.high_watermark += n_bytes

                metadata_buffer.start = start_loc
                metadata_buffer.end = start_loc + n_bytes
                metadata_buffer.write_start_end()

                # let caller write to the buffer
                with self.buffer.get_data(metadata_buffer.start,
                                          metadata_buffer.end) as buf:
                    yield buf

                # caller has written to the buffer
                metadata_buffer.mark_ready_to_read()
                self.current_idx = (self.current_idx +
                                    1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(self):
        assert self._is_reader, "Only readers can acquire read"
        start_time = time.time()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                if not metadata_buffer.can_read(self.reader_rank):
                    # for readers, `self.current_idx` is the next block to read
                    # if this block is not ready,
                    # we need to wait until it is written

                    # wait for a while (0.1 us)
                    time.sleep(1e-7)

                    # if we wait for a long time, we should warn the user
                    if time.time(
                    ) - start_time > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:  # noqa
                        logger.warning(
                            "No available block found in %s second. ",
                            VLLM_RINGBUFFER_WARNING_INTERVAL)
                        n_warning += 1

                    continue

                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer.get_data(metadata_buffer.start,
                                          metadata_buffer.end) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer.mark_read(self.reader_rank)
                self.current_idx = (self.current_idx +
                                    1) % self.buffer.max_chunks
                break

    def enqueue(self, obj):
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        n_bytes = len(serialized_obj)
        with self.acquire_write(n_bytes) as buf:
            buf[:] = serialized_obj

    def dequeue(self):
        assert self._is_reader, "Only readers can dequeue"
        with self.acquire_read() as buf:
            # no need to know the size of serialized object
            # pickle format itself contains the size information internally
            # see https://docs.python.org/3/library/pickle.html
            obj = pickle.loads(buf)
        return obj

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        else:
            return self.dequeue()

    def create_from_process_group(
        pg: ProcessGroup,
        writer_rank=0,
        max_bytes=10 << 20,
        max_chunks=1000,
    ) -> "ShmRingBufferIO":
        group_rank = dist.get_rank(pg)
        group_world_size = dist.get_world_size(pg)
        ranks_inside_group = list(range(group_world_size))
        global_ranks = dist.get_process_group_ranks(pg)
        n_reader = group_world_size - 1
        buffer: ShmRingBuffer
        if group_rank == writer_rank:
            buffer = ShmRingBuffer(n_reader, max_bytes, max_chunks)
            dist.broadcast_object_list([buffer],
                                       src=global_ranks[writer_rank],
                                       group=pg)
            return ShmRingBufferIO(buffer, -1)
        else:
            recv = [None]
            dist.broadcast_object_list(recv,
                                       src=global_ranks[writer_rank],
                                       group=pg)
            buffer = recv[0]  # type: ignore
            rest_ranks = [r for r in ranks_inside_group if r != writer_rank]
            return ShmRingBufferIO(buffer, rest_ranks.index(group_rank))
