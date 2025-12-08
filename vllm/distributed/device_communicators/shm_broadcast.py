# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pickle import PickleBuffer
from threading import Event
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
import zmq
from torch.distributed import ProcessGroup
from zmq import (  # type: ignore
    IPV6,  # type: ignore
    SUB,
    SUBSCRIBE,
    XPUB,
    XPUB_VERBOSE,
    Context,
)

import vllm.envs as envs
from vllm.distributed.utils import StatelessProcessGroup, sched_yield
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    get_ip,
    get_open_port,
    get_open_zmq_ipc_path,
    is_valid_ipv6_address,
)

if TYPE_CHECKING:
    from _typeshed import SizedBuffer

VLLM_RINGBUFFER_WARNING_INTERVAL = envs.VLLM_RINGBUFFER_WARNING_INTERVAL

from_bytes_big = functools.partial(int.from_bytes, byteorder="big")


def to_bytes_big(value: int, size: int) -> bytes:
    return value.to_bytes(size, byteorder="big")


logger = init_logger(__name__)


def long_wait_time_msg(threshold: int) -> str:
    return (
        "No available shared memory broadcast block found "
        f"in {threshold} seconds. This typically happens "
        "when some processes are hanging or doing some "
        "time-consuming work (e.g. compilation, "
        "weight/kv cache quantization)."
    )


class SpinTimer:
    def record_activity(self):
        pass

    def spin(self):
        sched_yield()


class SpinSleepTimer(SpinTimer):
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when vllm does nothing. This would lead to more
    CPU thermal headroom when a request eventually comes, especially when
    multiple GPUs are connected as each GPU would otherwise pin one thread at
    100% CPU usage.

    The simplest solution is to reduce polling frequency when there is no
    activity for a certain period of time.
    """

    def __init__(self, busy_loop_s: float = 3.0, wait_sleep_s: float = 0.1):
        self.last_activity = time.monotonic()
        self.busy_loop_s = busy_loop_s
        self.wait_sleep_s = wait_sleep_s

    def record_activity(self):
        self.last_activity = time.monotonic()

    def spin(self):
        curr_time = time.monotonic()
        if curr_time >= self.last_activity + self.busy_loop_s:
            time.sleep(self.wait_sleep_s)
        else:
            sched_yield()


class ShmRingBuffer:
    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: str | None = None,
    ):
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
        """  # noqa
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (
            self.max_chunk_bytes + self.metadata_size
        ) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if name is None:
            # we are creating a buffer
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes_of_buffer
            )
            # initialize the metadata section to 0
            with self.shared_memory.buf[self.metadata_offset :] as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            # we are opening an existing buffer
            self.is_creator = False
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    # See https://docs.python.org/3/library/multiprocessing.shared_memory.html # noqa
                    # Some platforms allocate memory based on page size,
                    # so the shared memory block size may be larger or equal
                    # to the requested size. The size parameter is ignored
                    # when attaching to an existing block.
                    assert self.shared_memory.size >= self.total_bytes_of_buffer
                except FileNotFoundError:
                    # we might deserialize the object in a different node
                    # in this case, this object is not used,
                    # and we should suppress the error
                    pass

    def handle(self):
        return (
            self.n_reader,
            self.max_chunk_bytes,
            self.max_chunks,
            self.shared_memory.name,
        )

    def __reduce__(self):
        return (
            self.__class__,
            self.handle(),
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
        with self.shared_memory.buf[start:end] as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with self.shared_memory.buf[start:end] as buf:
            yield buf


@dataclass
class Handle:
    local_reader_ranks: list[int] = field(default_factory=list)

    buffer_handle: tuple[int, int, int, str] | None = None
    local_subscribe_addr: str | None = None
    remote_subscribe_addr: str | None = None
    remote_addr_ipv6: bool = False


class MessageQueue:
    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: list[int] | None = None,
        # Default of 24MiB chosen to be large enough to accommodate grammar
        # bitmask tensors for large batches (1024 requests).
        max_chunk_bytes: int = 1024 * 1024 * 24,
        max_chunks: int = 10,
        connect_ip: str | None = None,
    ):
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader

        context = Context()

        if n_local_reader > 0:
            # for local readers, we will:
            # 1. create a shared memory ring buffer to communicate small data
            # 2. create a publish-subscribe socket to communicate large data
            self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes, max_chunks)

            # XPUB is very similar to PUB,
            # except that it can receive subscription messages
            # to confirm the number of subscribers
            self.local_socket = context.socket(XPUB)
            # set the verbose option so that we can receive every subscription
            # message. otherwise, we will only receive the first subscription
            # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
            self.local_socket.setsockopt(XPUB_VERBOSE, True)
            local_subscribe_addr = get_open_zmq_ipc_path()
            logger.debug("Binding to %s", local_subscribe_addr)
            self.local_socket.bind(local_subscribe_addr)

            self.current_idx = 0
        else:
            self.buffer = None  # type: ignore
            local_subscribe_addr = None
            self.local_socket = None
            self.current_idx = -1

        remote_addr_ipv6 = False
        if n_remote_reader > 0:
            # for remote readers, we will:
            # create a publish-subscribe socket to communicate large data
            if not connect_ip:
                connect_ip = get_ip()
            self.remote_socket = context.socket(XPUB)
            self.remote_socket.setsockopt(XPUB_VERBOSE, True)
            remote_subscribe_port = get_open_port()
            if is_valid_ipv6_address(connect_ip):
                self.remote_socket.setsockopt(IPV6, 1)
                remote_addr_ipv6 = True
                connect_ip = f"[{connect_ip}]"
            socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
            self.remote_socket.bind(socket_addr)
            remote_subscribe_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"
        else:
            remote_subscribe_addr = None
            self.remote_socket = None

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        # rank does not matter for remote readers
        self._is_remote_reader = False
        self._read_spin_timer = SpinTimer()

        self.handle = Handle(
            local_reader_ranks=local_reader_ranks,
            buffer_handle=self.buffer.handle() if self.buffer is not None else None,
            local_subscribe_addr=local_subscribe_addr,
            remote_subscribe_addr=remote_subscribe_addr,
            remote_addr_ipv6=remote_addr_ipv6,
        )

        logger.debug("vLLM message queue communication handle: %s", self.handle)

    def export_handle(self) -> Handle:
        return self.handle

    @staticmethod
    def create_from_handle(handle: Handle, rank) -> "MessageQueue":
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False

        context = Context()

        if rank in handle.local_reader_ranks:
            assert handle.buffer_handle is not None
            self.buffer = ShmRingBuffer(*handle.buffer_handle)
            self.current_idx = 0
            self.local_reader_rank = handle.local_reader_ranks.index(rank)
            self._is_local_reader = True
            self._is_remote_reader = False

            self.local_socket = context.socket(SUB)
            self.local_socket.setsockopt_string(SUBSCRIBE, "")
            socket_addr = handle.local_subscribe_addr
            logger.debug("Connecting to %s", socket_addr)
            self.local_socket.connect(socket_addr)

            self.remote_socket = None

            self._read_spin_timer = (
                SpinSleepTimer() if envs.VLLM_SLEEP_WHEN_IDLE else SpinTimer()
            )
        else:
            self.buffer = None  # type: ignore
            self.current_idx = -1
            self.local_reader_rank = -1
            self._is_local_reader = False
            self._is_remote_reader = True

            self.local_socket = None

            self.remote_socket = context.socket(SUB)
            self.remote_socket.setsockopt_string(SUBSCRIBE, "")
            if handle.remote_addr_ipv6:
                self.remote_socket.setsockopt(IPV6, 1)
            socket_addr = handle.remote_subscribe_addr
            logger.debug("Connecting to %s", socket_addr)
            self.remote_socket.connect(socket_addr)

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
    def acquire_write(self, timeout: float | None = None):
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

                    # Release the processor to other threads
                    sched_yield()

                    # if we time out, raise an exception
                    elapsed = time.monotonic() - start_time
                    if timeout is not None and elapsed > timeout:
                        raise TimeoutError

                    # if we wait for a long time, log a message
                    if elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning:
                        logger.info(
                            long_wait_time_msg(VLLM_RINGBUFFER_WARNING_INTERVAL)
                        )
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
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(
        self,
        timeout: float | None = None,
        cancel: Event | None = None,
        indefinite: bool = False,
    ):
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

                    # Release the processor to other threads
                    self._read_spin_timer.spin()

                    if cancel is not None and cancel.is_set():
                        raise RuntimeError("cancelled")

                    # if we time out, raise an exception
                    elapsed = time.monotonic() - start_time
                    if timeout is not None and elapsed > timeout:
                        raise TimeoutError

                    # if we wait for a long time, log a message
                    if not indefinite and (
                        elapsed > VLLM_RINGBUFFER_WARNING_INTERVAL * n_warning
                    ):
                        logger.info(
                            long_wait_time_msg(VLLM_RINGBUFFER_WARNING_INTERVAL)
                        )
                        n_warning += 1

                    continue
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.local_reader_rank + 1] = 1
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks

                self._read_spin_timer.record_activity()
                break

    def enqueue(self, obj, timeout: float | None = None):
        """Write to message queue with optional timeout (in seconds)"""
        assert self._is_writer, "Only writers can enqueue"
        all_buffers: list[SizedBuffer] = [b""]
        total_bytes = 6  # 2 bytes for oob buffer count, 4 for main buffer size

        def oob_callback(buf: PickleBuffer) -> bool:
            raw_buf = buf.raw()
            if len(raw_buf) < 1024 * 1024:
                # In-line buffers smaller than 1MiB.
                return True
            all_buffers.append(raw_buf)
            nonlocal total_bytes
            total_bytes += len(raw_buf) + 4
            return False

        all_buffers[0] = pickle.dumps(
            obj, protocol=pickle.HIGHEST_PROTOCOL, buffer_callback=oob_callback
        )
        if self.n_local_reader > 0:
            if total_bytes + len(all_buffers[0]) >= self.buffer.max_chunk_bytes:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 1  # overflow
                self.local_socket.send_multipart(all_buffers, copy=False)
            else:
                # Byte 0: 0
                # Bytes 1-2: Count of buffers
                # Then each buffer follows, preceded by 4 bytes containing its length:
                # [4 byte int L][L bytes of buffer content] ...
                with self.acquire_write(timeout) as buf:
                    buf[0] = 0  # not overflow
                    offset = 3
                    buf[1:offset] = to_bytes_big(len(all_buffers), 2)  # oob buf count
                    for buffer in all_buffers:
                        buf_len = len(buffer)
                        # prepend each buffer with 4 bytes containing its size.
                        buf_offset = offset + 4
                        buf[offset:buf_offset] = to_bytes_big(buf_len, 4)
                        buf[buf_offset : (offset := buf_offset + buf_len)] = buffer

        if self.n_remote_reader > 0:
            self.remote_socket.send_multipart(all_buffers, copy=False)

    def dequeue(
        self,
        timeout: float | None = None,
        cancel: Event | None = None,
        indefinite: bool = False,
    ):
        """Read from message queue with optional timeout (in seconds)"""
        if self._is_local_reader:
            with self.acquire_read(timeout, cancel, indefinite) as buf:
                overflow = buf[0] == 1
                if not overflow:
                    offset = 3
                    buf_count = from_bytes_big(buf[1:offset])
                    all_buffers = []
                    for i in range(buf_count):
                        buf_offset = offset + 4
                        buf_len = from_bytes_big(buf[offset:buf_offset])
                        offset = buf_offset + buf_len
                        all_buffers.append(buf[buf_offset:offset])
                    obj = pickle.loads(all_buffers[0], buffers=all_buffers[1:])
            if overflow:
                obj = MessageQueue.recv(self.local_socket, timeout)
        elif self._is_remote_reader:
            obj = MessageQueue.recv(self.remote_socket, timeout)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj

    @staticmethod
    def recv(socket: zmq.Socket, timeout: float | None) -> Any:
        timeout_ms = None if timeout is None else int(timeout * 1000)
        if not socket.poll(timeout=timeout_ms):
            raise TimeoutError
        recv, *recv_oob = socket.recv_multipart(copy=False)
        return pickle.loads(recv, buffers=recv_oob)

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        return self.dequeue()

    @staticmethod
    def create_from_process_group_single_reader(
        pg: ProcessGroup,
        max_chunk_bytes,
        max_chunks,
        reader_rank: int = 0,
        blocking: bool = False,
    ) -> tuple["MessageQueue", list[Handle]]:
        """
        Creates a MessageQueue for a process group with a single reader.

        This method is designed for scenarios where only one process (the reader)
        will consume messages, and all other processes are writers. It sets up
        the shared memory buffer and communication handles accordingly, and
        gathers the handles from all processes to the reader.

        Args:
            pg (ProcessGroup): The torch distributed process group.
            max_chunk_bytes (int): Maximum size in bytes for each chunk in the buffer.
            max_chunks (int): Maximum number of chunks in the buffer.
            reader_rank (int, optional): The global rank that will act as the reader.
                Defaults to 0.
            blocking (bool, optional): If True, blocks until all processes are ready.
                Defaults to False.

        Returns:
            tuple[MessageQueue, list[Handle]]:
            The MessageQueue instance for the calling process,
            and a list of handles (only non-empty for the reader process).
        """
        local_size = current_platform.device_count()
        rank = dist.get_rank()
        same_node = rank // local_size == reader_rank // local_size
        buffer_io = MessageQueue(
            n_reader=1,
            n_local_reader=1 if same_node else 0,
            max_chunk_bytes=max_chunk_bytes,
            max_chunks=max_chunks,
        )
        handle = buffer_io.export_handle()
        handles = [None] * dist.get_world_size(pg) if rank == reader_rank else None
        dist.gather_object(handle, handles, dst=reader_rank, group=pg)
        if blocking:
            buffer_io.wait_until_ready()
        return buffer_io, cast(list[Handle], handles or [])

    @staticmethod
    def create_from_process_group(
        pg: ProcessGroup | StatelessProcessGroup,
        max_chunk_bytes,
        max_chunks,
        writer_rank: int = 0,
        external_writer_handle=None,
        blocking: bool = True,
    ) -> "MessageQueue":
        """
        Creates a MessageQueue for a distributed process group with one writer and
        multiple readers.

        This method is designed for scenarios where one process (the writer) sends
        messages, and all other processes (the readers) receive messages. It sets up
        the shared memory buffer and socket communication handles accordingly, and
        broadcasts the handle from the writer to all readers.

        Args:
            pg (ProcessGroup | StatelessProcessGroup): The torch distributed process
                group.
            max_chunk_bytes (int): Maximum size in bytes for each chunk in the buffer.
            max_chunks (int): Maximum number of chunks in the buffer.
            writer_rank (int, optional): The global rank that will act as the writer.
                Defaults to 0.
            external_writer_handle (Handle, optional): Used when there is a handle
                from an external Message Queue. If provided, use this handle to init
                PG writer message queue instead of creating a new one. Defaults to None.
            blocking (bool, optional): If True, blocks until all processes are ready.
                Defaults to True.

        Returns:
            MessageQueue: The MessageQueue instance for the calling process.

        """
        if isinstance(pg, ProcessGroup):
            group_rank = dist.get_rank(pg)
            group_world_size = dist.get_world_size(pg)
            global_ranks = dist.get_process_group_ranks(pg)
        else:
            group_rank = pg.rank
            group_world_size = pg.world_size
            global_ranks = list(range(pg.world_size))
        from vllm.distributed.parallel_state import in_the_same_node_as

        status = in_the_same_node_as(pg, source_rank=writer_rank)
        if group_rank == writer_rank:
            if external_writer_handle is not None:
                buffer_io = MessageQueue.create_from_handle(
                    external_writer_handle, group_rank
                )
            else:
                same_node_ranks = [i for i, s in enumerate(status) if s]
                n_reader = group_world_size - 1
                n_local_reader = len(same_node_ranks) - 1
                local_reader_ranks = [i for i in same_node_ranks if i != writer_rank]
                buffer_io = MessageQueue(
                    n_reader=n_reader,
                    n_local_reader=n_local_reader,
                    local_reader_ranks=local_reader_ranks,
                    max_chunk_bytes=max_chunk_bytes,
                    max_chunks=max_chunks,
                )
            handle = buffer_io.export_handle()
            if isinstance(pg, ProcessGroup):
                dist.broadcast_object_list(
                    [handle], src=global_ranks[writer_rank], group=pg
                )
            else:
                pg.broadcast_obj(handle, writer_rank)
        else:
            if isinstance(pg, ProcessGroup):
                recv = [None]
                dist.broadcast_object_list(
                    recv, src=global_ranks[writer_rank], group=pg
                )
                handle = recv[0]  # type: ignore
            else:
                handle = pg.broadcast_obj(None, writer_rank)
            buffer_io = MessageQueue.create_from_handle(handle, group_rank)
        if blocking:
            buffer_io.wait_until_ready()
        return buffer_io
