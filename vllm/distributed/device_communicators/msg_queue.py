import pickle

from torch import distributed as dist
from torch.distributed import ProcessGroup
from zmq import (
    PUB,  # type: ignore
    RCVHWM,  # type: ignore
    REP,  # type: ignore
    REQ,  # type: ignore
    SNDHWM,  # type: ignore
    SUB,  # type: ignore
    SUBSCRIBE,  # type: ignore
    Context,
)  # type: ignore

from vllm.utils import get_ip, get_open_port


class Publisher:

    def __init__(self, n_reader: int):
        self.n_reader = n_reader
        context = Context()
        self.socket = context.socket(PUB)
        self.port = get_open_port()
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt(SNDHWM, 1000)

        self.sync_socket = context.socket(REP)
        self.sync_port = get_open_port()
        self.sync_socket.bind(f"tcp://*:{self.sync_port}")

        self.ip = get_ip()

    def export_handle(self):
        return (self.n_reader, self.ip, self.port, self.sync_port)

    def wait_for_ready(self):
        # wait for all readers to connect
        for i in range(self.n_reader):
            recv = self.sync_socket.recv()
            assert recv == b"READY"
            self.sync_socket.send(b"READY")
        self.socket.send(b"READY")

    def enqueue(self, obj):
        self.socket.send(pickle.dumps(obj))

    def __del__(self):
        self.socket.close()
        self.sync_socket.close()


class Subscriber:

    def __init__(self, handle):
        self.n_reader, ip, port, sync_port = handle
        context = Context()
        self.socket = context.socket(SUB)
        self.socket.setsockopt_string(SUBSCRIBE, "")
        self.socket.setsockopt(RCVHWM, 1000)
        self.socket.connect(f"tcp://{ip}:{port}")

        self.sync_socket = context.socket(REQ)
        self.sync_socket.connect(f"tcp://{ip}:{sync_port}")

    def wait_for_ready(self):
        self.sync_socket.send(b"READY")
        recv = self.sync_socket.recv()
        assert recv == b"READY"

        recv = self.socket.recv()
        assert recv == b"READY"

    def dequeue(self):
        return pickle.loads(self.socket.recv())

    def __del__(self):
        self.socket.close()
        self.sync_socket.close()


class PublishSubscribeMsgQueue:

    def __init__(self, n_reader: int, reader_rank: int, handle):
        self.reader_rank = reader_rank
        self._is_writer = self.reader_rank == -1
        self._is_reader = not self._is_writer
        if self._is_reader:
            assert 0 <= self.reader_rank < n_reader, \
                (f"Invalid reader rank {self.reader_rank} for "
                f"{n_reader} readers")
        if self._is_writer:
            self.publisher = handle
        else:
            assert handle is not None, "handle must be provided for readers"
            self.subscriber = Subscriber(handle)

    def wait_for_ready(self):
        if self._is_writer:
            self.publisher.wait_for_ready()
        else:
            self.subscriber.wait_for_ready()

    def enqueue(self, obj):
        assert self._is_writer, "enqueue is only allowed for writer"
        self.publisher.enqueue(obj)

    def dequeue(self):
        assert self._is_reader, "dequeue is only allowed for readers"
        return self.subscriber.dequeue()

    def broadcast_object(self, obj=None):
        if self._is_writer:
            self.enqueue(obj)
            return obj
        else:
            return self.dequeue()

    def create_from_process_group(pg: ProcessGroup,
                                  writer_rank=0) -> "PublishSubscribeMsgQueue":
        group_rank = dist.get_rank(pg)
        group_world_size = dist.get_world_size(pg)
        ranks_inside_group = list(range(group_world_size))
        global_ranks = dist.get_process_group_ranks(pg)
        n_reader = group_world_size - 1
        if group_rank == writer_rank:
            buffer = Publisher(n_reader)
            handle = buffer.export_handle()
            dist.broadcast_object_list([handle], src=global_ranks[writer_rank])
            queue = PublishSubscribeMsgQueue(n_reader, -1, buffer)
            queue.wait_for_ready()
            return queue
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=global_ranks[writer_rank])
            handle = recv[0]  # type: ignore
            rest_ranks = [r for r in ranks_inside_group if r != writer_rank]
            queue = PublishSubscribeMsgQueue(n_reader,
                                             rest_ranks.index(group_rank),
                                             handle)
            queue.wait_for_ready()
            return queue
