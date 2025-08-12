import pickle
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL

logger = init_logger(__name__)


class ShmRingBuffer:

    def __init__(self,
                 n_reader: int,
                 max_chunk_bytes: int,
                 max_chunks: int,
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
                    | (current_idx)                        | (current_idx)
                    v                                      v
        +-------------------------------+----------------------------------------+
        | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
        +-------------------------------+----------------------------------------+
        | max_chunks x max_chunk_bytes  | max_chunks x (1 + n_reader) bytes      |

        metadata memory layout: each byte is a flag, the first byte is the written
        flag, and the rest are reader flags. The flags are set to 0 by default.
        +--------------+--------------+--------------+-----+--------------+
        | written_flag | reader0_flag | reader1_flag | ... | readerN_flag |
        +--------------+--------------+--------------+-----+--------------+

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

    def __init__(self, buffer: ShmRingBuffer, reader_rank: int):
        self.buffer = buffer
        self.reader_rank = reader_rank
        self._is_writer = self.reader_rank == -1
        self._is_reader = not self._is_writer
        if self._is_reader:
            assert 0 <= self.reader_rank < buffer.n_reader, \
                (f"Invalid reader rank {self.reader_rank} for buffer"
                f" created with {buffer.n_reader} readers")
        self.current_idx = 0

    @contextmanager
    def acquire_write(self):
        assert self._is_writer, "Only writers can acquire write"
        start_index = self.current_idx
        start_time = time.time()
        n_warning = 1
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
                        if time.time(
                        ) - start_time > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:  # noqa
                            logger.warning(
                                "No available block found in %s second. ",
                                VLLM_RINGBUFFER_WARNING_INTERVAL)
                            n_warning += 1
                        # wait for a while (0.1 us)
                        time.sleep(1e-7)
                    continue
                # found a block that is either
                # (1) not written
                # (2) read by all readers

                # mark the block as not written
                metadata_buffer[0] = 0
                # let caller write to the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has written to the buffer
                # mark the block as written
                metadata_buffer[0] = 1
                for i in range(1, self.buffer.n_reader + 1):
                    # set read flag to 0, meaning it is not read yet
                    metadata_buffer[i] = 0
                break

    @contextmanager
    def acquire_read(self):
        assert self._is_reader, "Only readers can acquire read"
        start_index = self.current_idx
        start_time = time.time()
        n_warning = 1
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
                        if time.time(
                        ) - start_time > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:  # noqa
                            logger.warning(
                                "No available block found in %s second. ",
                                VLLM_RINGBUFFER_WARNING_INTERVAL)
                            n_warning += 1
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

    def enqueue(self, obj):
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if len(serialized_obj) > self.buffer.max_chunk_bytes:
            raise RuntimeError(
                f"{len(serialized_obj)=} larger than the allowed value "
                f"{self.buffer.max_chunk_bytes},"
                "Please increase the max_chunk_bytes parameter.")
        with self.acquire_write() as buf:
            buf[:len(serialized_obj)] = serialized_obj

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

    def create_from_process_group(pg: ProcessGroup,
                                  max_chunk_bytes,
                                  max_chunks,
                                  writer_rank=0) -> "ShmRingBufferIO":
        group_rank = dist.get_rank(pg)
        group_world_size = dist.get_world_size(pg)
        ranks_inside_group = list(range(group_world_size))
        global_ranks = dist.get_process_group_ranks(pg)
        n_reader = group_world_size - 1
        buffer: ShmRingBuffer
        if group_rank == writer_rank:
            buffer = ShmRingBuffer(n_reader, max_chunk_bytes, max_chunks)
            dist.broadcast_object_list([buffer], src=global_ranks[writer_rank])
            dist.barrier(pg)
            return ShmRingBufferIO(buffer, -1)
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=global_ranks[writer_rank])
            dist.barrier(pg)
            buffer = recv[0]  # type: ignore
            rest_ranks = [r for r in ranks_inside_group if r != writer_rank]
            return ShmRingBufferIO(buffer, rest_ranks.index(group_rank))
