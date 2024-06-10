import time
import multiprocessing
from multiprocessing import shared_memory, resource_tracker
from multiprocessing import Array, Value
import contextlib
from contextlib import contextmanager
from torch.distributed import ProcessGroup
import torch
import torch.distributed as dist
import pickle


class ShmRingBuffer:

    def __init__(self, pg: ProcessGroup, max_chunk_bytes, max_chunks):
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)
        global_ranks = dist.get_process_group_ranks(pg)
        self.is_writer = self.rank == 0
        self.is_reader = not self.is_writer
        self.current_idx = 0
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        total_bytes = (self.max_chunk_bytes +
                       self.world_size) * self.max_chunks
        self.buffer_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if self.is_writer:
            self.shared_memory = shared_memory.SharedMemory(create=True,
                                                            size=total_bytes)
            # initialize the buffer to 0
            for i in range(total_bytes):
                self.shared_memory.buf[i] = 0
            dist.broadcast_object_list([self.shared_memory.name],
                                       src=global_ranks[0])
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=global_ranks[0])
            name = recv[0]
            self.shared_memory = shared_memory.SharedMemory(name=name)
            resource_tracker.unregister(self.shared_memory._name,
                                        "shared_memory") # noqa

    @property
    def buffer(self):
        start = self.buffer_offset + self.current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        return memoryview(self.shared_memory.buf[start:end])

    @property
    def metadata(self):
        start = self.metadata_offset + self.current_idx * self.world_size
        end = start + self.world_size
        return memoryview(self.shared_memory.buf[start:end])

    @contextmanager
    def acquire_write(self):
        assert self.is_writer, "Only writers can acquire write"
        while True:
            with self.metadata as buffer:
                read_count = sum(buffer[1:])
                if buffer[0] and read_count != self.world_size - 1:
                    # this block is written and not read by all readers
                    # try to write to the next block
                    self.current_idx += 1
                    self.current_idx %= self.max_chunks
                    continue
                # found a block that is (1) not written or (2) read by all readers
                # let caller write to the buffer
                with self.buffer as buf:
                    yield buf

                # caller has written to the buffer
                # reset the state
                buffer[0] = 1
                for i in range(1, self.world_size):
                    buffer[i] = 0
                break

    @contextmanager
    def acquire_read(self):
        assert self.is_reader, "Only readers can acquire read"
        while True:
            with self.metadata as buffer:
                read_flag = buffer[self.rank]
                if not buffer[0] or read_flag:
                    # this block is (1) not written or (2) already read by this reader
                    # try to read the next block
                    self.current_idx += 1
                    self.current_idx %= self.max_chunks
                    continue
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                buffer[self.rank] = 1
                break

    def broadcast_object(self, obj=None):
        if self.is_writer:
            serialized_obj = pickle.dumps(obj)
            with self.acquire_write() as buf:
                buf[:len(serialized_obj)] = serialized_obj
            return obj
        else:
            with self.acquire_read() as buf:
                obj = pickle.loads(buf)
            return obj

    def __del__(self):
        if self.is_writer:
            self.shared_memory.close()
            self.shared_memory.unlink()
        else:
            self.shared_memory.close()
