import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import List, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from zmq import SUB, SUBSCRIBE, XPUB, XPUB_VERBOSE, Context  # type: ignore

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils import get_ip, get_open_port

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL

# time to wait if the queue is full or empty
# if we sleep for too short, it will consume too much CPU
# if we sleep for too long, it will slow down the writer/reader
# 0.1 us is a good balance
RINGBUFFER_SLEEP_INTERVAL = 1e-7

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

        The state of metadata is as follows:

        (case 1) 0???...???: the block is not written yet, cannot read, can write
        (case 2) 1000...000: the block is just written, can read, cannot write
        (case 3) 1???...???: the block is written and read by some readers, can read if not read, cannot write
        (case 4) 1111...111: the block is written and read by all readers, cannot read, can write

        State transition for readers:

        When a reader finds a block that it can read (case 2 or 3), it can yield the block for caller to read.
        Only after the caller finishes reading the block, the reader can mark the block as read.
        Readers only mark the block as read (from 0 to 1), the writer marks the block as ready to read (from 1 to 0).

        State transition for writer:

        When the writer writes to a block (case 1 or 4), it first resets the written flag to 0, converting either case
        to case 1. Then it can yield the block for caller to write. After the caller finishes writing the block, the writer
        can reset the reader flags to 0, and mark the block as written (from 0 to 1).
        NOTE: the order is important here, first reset the reader flags (so that we are still in case 1), then mark the block as written. The state transition is atomic. If we do it in the reverse order, it will go through case 3 and then back to case 2, and readers might read the intermediate case 3, which is not correct.

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
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    assert (
                        self.shared_memory.size == self.total_bytes_of_buffer)
                except FileNotFoundError:
                    # we might deserialize the object in a different node
                    # in this case, this object is not used,
                    # and we should suppress the error
                    pass

    def __reduce__(self):
        return (
            self.__class__,
            (self.n_reader, self.max_chunk_bytes, self.max_chunks,
             self.shared_memory.name),
        )

    def __del__(self):
        if hasattr(self, "shared_memory"):
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


@dataclass
class Handle:
    connect_ip: str
    local_reader_ranks: List[int] = field(default_factory=list)

    buffer: Optional[ShmRingBuffer] = None
    local_subscribe_port: Optional[int] = None
    remote_subscribe_port: Optional[int] = None


class MessageQueue:

    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: Optional[List[int]] = None,
        max_chunk_bytes: int = 1024 * 1024 * 10,
        max_chunks: int = 10,
        connect_ip: Optional[str] = None,
    ):
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader

        if connect_ip is None:
            connect_ip = get_ip() if n_remote_reader > 0 else "127.0.0.1"

        context = Context()

        if n_local_reader > 0:
            # for local readers, we will:
            # 1. create a shared memory ring buffer to communicate small data
            # 2. create a publish-subscribe socket to communicate large data
            self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes,
                                        max_chunks)

            # XPUB is very similar to PUB,
            # except that it can receive subscription messages
            # to confirm the number of subscribers
            self.local_socket = context.socket(XPUB)
            # set the verbose option so that we can receive every subscription
            # message. otherwise, we will only receive the first subscription
            # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
            self.local_socket.setsockopt(XPUB_VERBOSE, True)
            local_subscribe_port = get_open_port()
            self.local_socket.bind(f"tcp://*:{local_subscribe_port}")

            self.current_idx = 0

        else:
            self.buffer = None  # type: ignore
            local_subscribe_port = None
            self.local_socket = None
            self.current_idx = -1

        if n_remote_reader > 0:
            # for remote readers, we will:
            # create a publish-subscribe socket to communicate large data
            self.remote_socket = context.socket(XPUB)
            self.remote_socket.setsockopt(XPUB_VERBOSE, True)
            remote_subscribe_port = get_open_port()
            self.remote_socket.bind(f"tcp://*:{remote_subscribe_port}")

        else:
            remote_subscribe_port = None
            self.remote_socket = None

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        # rank does not matter for remote readers
        self._is_remote_reader = False

        self.handle = Handle(
            connect_ip=connect_ip,
            local_reader_ranks=local_reader_ranks,
            buffer=self.buffer,
            local_subscribe_port=local_subscribe_port,
            remote_subscribe_port=remote_subscribe_port,
        )

        logger.info("vLLM message queue communication handle: %s", self.handle)

    def export_handle(self) -> Handle:
        return self.handle

    @staticmethod
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False

        context = Context()

        if rank in handle.local_reader_ranks:
            assert handle.buffer is not None
            self.buffer = handle.buffer
            self.current_idx = 0
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            self.local_socket = context.socket(SUB)
            self.local_socket.setsockopt_string(SUBSCRIBE, "")
            self.local_socket.connect(
                f"tcp://{handle.connect_ip}:{handle.local_subscribe_port}")

            self.remote_socket = None
        else:
            self.buffer = None  # type: ignore
            self.current_idx = -1
            self.local_reader_rank = -1
            self._is_local_reader = False
            self._is_remote_reader = True

            self.local_socket = None

            self.remote_socket = context.socket(SUB)
            self.remote_socket.setsockopt_string(SUBSCRIBE, "")
            self.remote_socket.connect(
                f"tcp://{handle.connect_ip}:{handle.remote_subscribe_port}")

        return self

    def wait_until_ready(self):
        """This is a collective operation. All processes (including the
        readers and the writer) should call this function.
        """
        if self._is_writer:
            # wait for all readers to connect

            # local readers
            for i in range(self.n_local_reader):
                # wait for subscription messages from all local readers
                self.local_socket.recv()
            if self.n_local_reader > 0:
                # send a message to all local readers
                # to make sure the publish channel is working
                self.local_socket.send(b"READY")

            # remote readers
            for i in range(self.n_remote_reader):
                # wait for subscription messages from all remote readers
                self.remote_socket.recv()
            if self.n_remote_reader > 0:
                # send a message to all remote readers
                # to make sure the publish channel is working
                self.remote_socket.send(b"READY")
        elif self._is_local_reader:
            # wait for the writer to send a message
            recv = self.local_socket.recv()
            assert recv == b"READY"
        elif self._is_remote_reader:
            # wait for the writer to send a message
            recv = self.remote_socket.recv()
            assert recv == b"READY"

    @contextmanager
    def acquire_write(self):
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:
                    # this block is written and not read by all readers
                    # for writers, `self.current_idx` is the next block to write
                    # if this block is not ready to write,
                    # we need to wait until it is read by all readers

                    # wait for a while
                    time.sleep(RINGBUFFER_SLEEP_INTERVAL)

                    # if we wait for a long time, we should warn the user
                    if (time.monotonic() - start_time >
                            VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning):
                        logger.warning(
                            "No available block found in %s second. ",
                            VLLM_RINGBUFFER_WARNING_INTERVAL)
                        n_warning += 1

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
                # NOTE: order is important here
                # first set the read flags to 0
                # then set the written flag to 1
                # otherwise, the readers may think they already read the block
                for i in range(1, self.buffer.n_reader + 1):
                    # set read flag to 0, meaning it is not read yet
                    metadata_buffer[i] = 0
                # mark the block as written
                metadata_buffer[0] = 1
                self.current_idx = (self.current_idx +
                                    1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(self):
        assert self._is_local_reader, "Only readers can acquire read"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    # this block is either
                    # (1) not written
                    # (2) already read by this reader

                    # for readers, `self.current_idx` is the next block to read
                    # if this block is not ready,
                    # we need to wait until it is written

                    # wait for a while
                    time.sleep(RINGBUFFER_SLEEP_INTERVAL)

                    # if we wait for a long time, we should warn the user
                    if (time.monotonic() - start_time >
                            VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning):
                        logger.warning(
                            "No available block found in %s second. ",
                            VLLM_RINGBUFFER_WARNING_INTERVAL)
                        n_warning += 1

                    continue
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.local_reader_rank + 1] = 1
                self.current_idx = (self.current_idx +
                                    1) % self.buffer.max_chunks
                break

    def enqueue(self, obj):
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.n_local_reader > 0:
            if len(serialized_obj) >= self.buffer.max_chunk_bytes:
                with self.acquire_write() as buf:
                    buf[0] = 1  # overflow
                self.local_socket.send(serialized_obj)
            else:
                with self.acquire_write() as buf:
                    buf[0] = 0  # not overflow
                    buf[1:len(serialized_obj) + 1] = serialized_obj
        if self.n_remote_reader > 0:
            self.remote_socket.send(serialized_obj)

    def dequeue(self):
        if self._is_local_reader:
            with self.acquire_read() as buf:
                overflow = buf[0] == 1
                if not overflow:
                    # no need to know the size of serialized object
                    # pickle format contains the size information internally
                    # see https://docs.python.org/3/library/pickle.html
                    obj = pickle.loads(buf[1:])
            if overflow:
                recv = self.local_socket.recv()
                obj = pickle.loads(recv)
        elif self._is_remote_reader:
            recv = self.remote_socket.recv()
            obj = pickle.loads(recv)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        else:
            return self.dequeue()

    @staticmethod
    def create_from_process_group(pg: ProcessGroup,
                                  max_chunk_bytes,
                                  max_chunks,
                                  writer_rank=0) -> "MessageQueue":
        group_rank = dist.get_rank(pg)
        group_world_size = dist.get_world_size(pg)
        global_ranks = dist.get_process_group_ranks(pg)

        from vllm.distributed.parallel_state import in_the_same_node_as
        status = in_the_same_node_as(pg, source_rank=writer_rank)
        same_node_ranks = [i for i, s in enumerate(status) if s]
        n_reader = group_world_size - 1
        n_local_reader = len(same_node_ranks) - 1
        local_reader_ranks = [i for i in same_node_ranks if i != writer_rank]
        buffer_io: MessageQueue
        if group_rank == writer_rank:
            buffer_io = MessageQueue(
                n_reader=n_reader,
                n_local_reader=n_local_reader,
                local_reader_ranks=local_reader_ranks,
                max_chunk_bytes=max_chunk_bytes,
                max_chunks=max_chunks,
            )
            handle = buffer_io.export_handle()
            dist.broadcast_object_list([handle],
                                       src=global_ranks[writer_rank],
                                       group=pg)
        else:
            recv = [None]
            dist.broadcast_object_list(recv,
                                       src=global_ranks[writer_rank],
                                       group=pg)
            handle = recv[0]  # type: ignore
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)
        buffer_io.wait_until_ready()
        return buffer_io
