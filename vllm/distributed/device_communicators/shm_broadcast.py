import pickle
import time
import torch
from contextlib import contextmanager
from multiprocessing import shared_memory
from unittest.mock import patch
from typing import Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class ShmRingBuffer:

    def __init__(self,
                 n_reader: int,
                 max_chunk_bytes: int,
                 max_chunks: int,
                 name: Optional[str] = None):
        """
        A shared memory ring buffer implementation for broadcast communication.
        It is optimized for the case where there is one writer and multiple
         readers. In this case, we don't need to synchronize the access to
         the buffer.
        
        Buffer memory layout:
                  data                                 metadata
                    |                                      |
                    | (current_idx)                        | (current_idx)
                    v                                      v
        +-------------------------------+----------------------------------------+
        | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
        +-------------------------------+----------------------------------------+
        | max_chunks x max_chunk_bytes  | max_chunks x (1 + n_reader) bytes      |

        During creation, `name` is None and the buffer is created. We can pass the
        created object to other processes by pickling it. The other processes will
        get the name of the shared memory and open it, so that they can access the
        same shared memory buffer.
        """# noqa
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (self.max_chunk_bytes +
                                      self.metadata_size) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

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
            with memoryview(self.shared_memory.buf[self.metadata_offset:]
                            ) as metadata_buffer:
                tensor = torch.frombuffer(metadata_buffer, dtype=torch.uint8)
                assert torch.all(tensor == 0)

    def __reduce__(self):
        return (
            self.__class__,
            (self.n_reader, self.max_chunk_bytes, self.max_chunks,
             self.shared_memory.name),
        )

    def __del__(self):
        self.shared_memory.close()
        if self.is_creator:
            self.shared_memory.unlink()

    @contextmanager
    def get_data(self, current_idx: int):
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf


class ShmRingBufferIO:
    # seconds to wait before warning about a potential blocking call
    WARNING_INTERVAL = 60

    def __init__(self, buffer: ShmRingBuffer, reader_rank: int):
        self.buffer = buffer
        self.reader_rank = reader_rank
        self._is_writer = self.reader_rank == -1
        self._is_reader = not self._is_writer
        if self._is_reader:
            assert self.reader_rank >= 0 and self.reader_rank < buffer.n_reader, \
                f"Invalid reader rank {self.reader_rank} for buffer created with {buffer.n_reader} readers"
        self.current_idx = 0

    @contextmanager
    def acquire_write(self):
        assert self._is_writer, "Only writers can acquire write"
        start_index = self.current_idx
        start_time = time.time()
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:
                    # this block is written and not read by all readers
                    # try to write to the next block
                    self.current_idx = (self.current_idx +
                                        1) % self.buffer.max_chunks
                    if self.current_idx == start_index:
                        # no empty block found
                        if time.time() - start_time > self.WARNING_INTERVAL:
                            logger.warning(
                                "No available block found in %s second. ",
                                self.WARNING_INTERVAL)
                        # wait for a while (0.1 us)
                        time.sleep(1e-7)
                    continue
                # found a block that is either
                # (1) not written
                # (2) read by all readers
                # let caller write to the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has written to the buffer
                # reset the state
                metadata_buffer[0] = 1
                for i in range(1, self.buffer.n_reader + 1):
                    metadata_buffer[i] = 0
                break

    @contextmanager
    def acquire_read(self):
        assert self._is_reader, "Only readers can acquire read"
        start_index = self.current_idx
        start_time = time.time()
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_flag = metadata_buffer[self.reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    # this block is either
                    # (1) not written
                    # (2) already read by this reader
                    # try to read the next block
                    self.current_idx = (self.current_idx +
                                        1) % self.buffer.max_chunks
                    if self.current_idx == start_index:
                        # no block found
                        if time.time() - start_time > self.WARNING_INTERVAL:
                            logger.warning(
                                "No available block found in %s second. ",
                                self.WARNING_INTERVAL)
                        # wait for a while (0.1 us)
                        time.sleep(1e-7)
                    continue
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.reader_rank + 1] = 1
                break

    def broadcast_object(self, obj=None):
        if self._is_writer:
            serialized_obj = pickle.dumps(obj,
                                          protocol=pickle.HIGHEST_PROTOCOL)
            if len(serialized_obj) > self.buffer.max_chunk_bytes:
                raise RuntimeError(
                    f"{len(serialized_obj)=} larger than the allowed value "
                    f"{self.buffer.max_chunk_bytes},"
                    "Please increase the max_chunk_bytes parameter.")
            with self.acquire_write() as buf:
                buf[:len(serialized_obj)] = serialized_obj
            return obj
        else:
            with self.acquire_read() as buf:
                # no need to know the size of serialized object
                # pickle format itself contains the size information internally
                # see https://docs.python.org/3/library/pickle.html
                obj = pickle.loads(buf)
            return obj


def create_shm_ringbuffer_io(pg: ProcessGroup,
                             max_chunk_bytes,
                             max_chunks,
                             writer_rank=0):
    group_rank = dist.get_rank(pg)
    group_world_size = dist.get_world_size(pg)
    global_ranks = dist.get_process_group_ranks(pg)
    n_reader = group_world_size - 1
    buffer: ShmRingBuffer
    if group_rank == writer_rank:
        buffer = ShmRingBuffer(n_reader, max_chunk_bytes, max_chunks)
        dist.broadcast_object_list([buffer], src=global_ranks[0])
        return ShmRingBufferIO(buffer, -1)
    else:
        recv = [None]
        dist.broadcast_object_list(recv, src=global_ranks[0])
        buffer = recv[0]  # type: ignore
        rest_ranks = [r for r in global_ranks if r != writer_rank]
        return ShmRingBufferIO(buffer, rest_ranks.index(group_rank))
