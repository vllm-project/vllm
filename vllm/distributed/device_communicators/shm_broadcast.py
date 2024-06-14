import pickle
import time
from contextlib import contextmanager
from multiprocessing import shared_memory
from unittest.mock import patch

import torch.distributed as dist
from torch.distributed import ProcessGroup


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
        self.data_offset = 0
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
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch("multiprocessing.resource_tracker.register",
                       lambda *args, **kwargs: None):
                self.shared_memory = shared_memory.SharedMemory(name=name)

    @property
    def data(self):
        start = self.data_offset + self.current_idx * self.max_chunk_bytes
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
        start_index = self.current_idx
        while True:
            with self.metadata as metadata_buffer:
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.world_size - 1:
                    # this block is written and not read by all readers
                    # try to write to the next block
                    self.current_idx = (self.current_idx + 1) % self.max_chunks
                    if self.current_idx == start_index:
                        # no empty block found
                        # wait for a while (0.1 us)
                        time.sleep(1e-7)
                    continue
                # found a block that is either
                # (1) not written
                # (2) read by all readers
                # let caller write to the buffer
                with self.data as buf:
                    yield buf

                # caller has written to the buffer
                # reset the state
                metadata_buffer[0] = 1
                for i in range(1, self.world_size):
                    metadata_buffer[i] = 0
                break

    @contextmanager
    def acquire_read(self):
        assert self.is_reader, "Only readers can acquire read"
        start_index = self.current_idx
        while True:
            with self.metadata as metadata_buffer:
                read_flag = metadata_buffer[self.rank]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    # this block is either
                    # (1) not written
                    # (2) already read by this reader
                    # try to read the next block
                    self.current_idx = (self.current_idx + 1) % self.max_chunks
                    if self.current_idx == start_index:
                        # no block found
                        # wait for a while (0.1 us)
                        time.sleep(1e-7)
                    continue
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.data as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.rank] = 1
                break

    def broadcast_object(self, obj=None):
        if self.is_writer:
            serialized_obj = pickle.dumps(obj,
                                          protocol=pickle.HIGHEST_PROTOCOL)
            if len(serialized_obj) > self.max_chunk_bytes:
                raise RuntimeError(
                    f"{len(serialized_obj)=} larger than the allowed value "
                    f"{self.max_chunk_bytes},"
                    "Please increase the max_chunk_bytes parameter.")
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
