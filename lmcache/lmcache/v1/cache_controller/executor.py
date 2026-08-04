# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Union
import asyncio
import uuid

# Third Party
import msgspec
import zmq.asyncio

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.message import (  # noqa: E501
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    ClearWorkerMsg,
    CompressMsg,
    CompressRetMsg,
    CompressWorkerMsg,
    DecompressMsg,
    DecompressRetMsg,
    DecompressWorkerMsg,
    ErrorMsg,
    HealthMsg,
    HealthRetMsg,
    HealthWorkerMsg,
    HealthWorkerRetMsg,
    MoveMsg,
    MoveRetMsg,
    MoveWorkerMsg,
    Msg,
    MsgBase,
    PinMsg,
    PinRetMsg,
    PinWorkerMsg,
)

logger = init_logger(__name__)


# NOTE (Jiayi): `LMCacheClusterExecutor` might need to be in different processes
# in the future for the sake of performance.
# NOTE (Jiayi): Also, consider scaling up the number of cluster executors
# in the future.
# TODO (Jiayi): need better error handling
class LMCacheClusterExecutor:
    """
    LMCache Cluster Executor class to handle the execution of cache operations.
    """

    def __init__(self, reg_controller):
        """
        Initialize the LMCache Executor with a cache instance.

        :param lmcache_instance_id: lmcache_instance_id
        """
        self.reg_controller = reg_controller

    async def clear(self, msg: ClearMsg) -> Union[ClearRetMsg, ErrorMsg]:
        """
        Execute a clear cache operation with error handling.
        """
        instance_id = msg.instance_id
        location = msg.location

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None
        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)
            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for instance {instance_id}"
                    )
                )
            sockets.append(socket)

            # TODO(Jiayi): Need a way to track event_id -> worker_event_id mapping
            # Also, we need to track worker_event_id status
            worker_event_id = f"Worker{worker_id}{msg.event_id}"
            serialized_msg = msgspec.msgpack.encode(
                ClearWorkerMsg(
                    worker_event_id=worker_event_id,
                    location=location,
                )
            )
            serialized_msgs.append(serialized_msg)
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for i, serialized_result in enumerate(serialized_results):
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens cleared should be the same across all workers."
        )

        return ClearRetMsg(event_id=msg.event_id, num_tokens=num_tokens_list[0])

    async def pin(self, msg: PinMsg) -> Union[PinRetMsg, ErrorMsg]:
        """
        Execute a pin cache operation with error handling.
        """
        instance_id = msg.instance_id
        tokens = msg.tokens
        location = msg.location

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None
        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)
            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for instance {instance_id}"
                    )
                )
            sockets.append(socket)

            # TODO(Jiayi): Need a way to track event_id -> worker_event_id mapping
            # Also, we need to track worker_event_id status
            worker_event_id = f"Worker{worker_id}{msg.event_id}"
            serialized_msg = msgspec.msgpack.encode(
                PinWorkerMsg(
                    worker_event_id=worker_event_id,
                    tokens=tokens,
                    location=location,
                )
            )
            serialized_msgs.append(serialized_msg)
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for i, serialized_result in enumerate(serialized_results):
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens pinned should be the same across all workers."
        )

        return PinRetMsg(event_id=msg.event_id, num_tokens=num_tokens_list[0])

    async def compress(self, msg: CompressMsg) -> Union[CompressRetMsg, ErrorMsg]:
        """
        Execute a compress operation with error handling.
        """
        event_id = msg.event_id
        instance_id = msg.instance_id
        method = msg.method
        location = msg.location
        tokens = msg.tokens

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None

        # TODO(Jiayi): Currently, we do not support PP or heterogeneous TP.
        # NOTE(Jiayi): The TP ranks are already sorted in registration_controller.

        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)

            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for "
                        f"instance {instance_id} or "
                    )
                )
            sockets.append(socket)

            worker_event_id = f"CompressWorker{worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                CompressWorkerMsg(
                    worker_event_id=worker_event_id,
                    method=method,
                    location=location,
                    tokens=tokens,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending compress operation to worker ({instance_id}, {worker_id})"
            )
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for serialized_result in serialized_results:
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens compressed should be the same across all workers."
        )

        return CompressRetMsg(
            event_id=event_id,
            num_tokens=num_tokens_list[0],
        )

    async def decompress(self, msg: DecompressMsg) -> Union[DecompressRetMsg, ErrorMsg]:
        """
        Execute a decompress operation with error handling.
        """
        event_id = msg.event_id
        instance_id = msg.instance_id
        method = msg.method
        location = msg.location
        tokens = msg.tokens

        worker_ids = self.reg_controller.get_workers(instance_id)
        assert worker_ids is not None

        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)

            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for "
                        f"instance {instance_id} or "
                    )
                )
            sockets.append(socket)

            worker_event_id = f"DecompressWorker{worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                DecompressWorkerMsg(
                    worker_event_id=worker_event_id,
                    method=method,
                    location=location,
                    tokens=tokens,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending decompress operation to worker ({instance_id}, {worker_id})"
            )
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for serialized_result in serialized_results:
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens decompressed should be the same across all workers."
        )

        return DecompressRetMsg(
            event_id=event_id,
            num_tokens=num_tokens_list[0],
        )

    async def move(self, msg: MoveMsg) -> Union[MoveRetMsg, ErrorMsg]:
        """
        Execute a move cache operation with error handling.
        """
        # NOTE(Jiayi): Currently we assume the transfer is push-based.
        src_instance_id = msg.old_position[0]
        dst_instance_id = msg.new_position[0]

        src_worker_ids = self.reg_controller.get_workers(src_instance_id)
        assert src_worker_ids is not None
        dst_worker_ids = self.reg_controller.get_workers(dst_instance_id)
        assert dst_worker_ids is not None

        # TODO(Jiayi): Currently, we do not support PP or heterogeneous TP.
        # NOTE(Jiayi): The TP ranks are already sorted in registration_controller.

        sockets = []
        serialized_msgs = []
        for src_worker_id, dst_worker_id in zip(
            src_worker_ids, dst_worker_ids, strict=False
        ):
            socket = self.reg_controller.get_socket(src_instance_id, src_worker_id)
            dst_url = self.reg_controller.get_peer_init_url(
                dst_instance_id, dst_worker_id
            )

            if socket is None or dst_url is None:
                return ErrorMsg(
                    error=(
                        f"Src worker {src_worker_id} not registered for "
                        f"instance {src_instance_id} or "
                        f"dst worker {dst_worker_id} not registered for "
                        f"instance {dst_instance_id} or P2P is not enabled."
                    )
                )
            sockets.append(socket)

            worker_event_id = f"MoveWorker{src_worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                MoveWorkerMsg(
                    worker_event_id=worker_event_id,
                    old_position=msg.old_position[1],
                    new_position=(dst_url, msg.new_position[1]),
                    tokens=msg.tokens,
                    copy=msg.copy,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending move operation to worker ({src_instance_id}, {src_worker_id})"
            )
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        num_tokens_list = []
        for serialized_result in serialized_results:
            result = msgspec.msgpack.decode(serialized_result, type=Msg)
            num_tokens_list.append(result.num_tokens)

        # TODO(Jiayi): Need to ensure cache consistency across workers.
        assert len(set(num_tokens_list)) == 1, (
            "The number of tokens moved should be the same across all workers."
        )

        return MoveRetMsg(
            event_id=msg.event_id,
            num_tokens=num_tokens_list[0],
        )

    async def health(self, msg: HealthMsg) -> Union[HealthRetMsg, ErrorMsg]:
        """
        Execute a compress operation with error handling.
        """
        instance_id = msg.instance_id

        worker_ids = self.reg_controller.get_workers(instance_id)
        if worker_ids is None:
            return ErrorMsg(error=f"No workers found for instance {instance_id}")

        # TODO(Jiayi): Currently, we do not support PP or heterogeneous TP.
        # NOTE(Jiayi): The TP ranks are already sorted in registration_controller.

        sockets = []
        serialized_msgs = []
        for worker_id in worker_ids:
            socket = self.reg_controller.get_socket(instance_id, worker_id)

            if socket is None:
                return ErrorMsg(
                    error=(
                        f"Worker {worker_id} not registered for "
                        f"instance {instance_id} or socket not found"
                    )
                )
            sockets.append(socket)

            worker_event_id = f"HealthWorker{worker_id}{str(uuid.uuid4())}"
            serialized_msg = msgspec.msgpack.encode(
                HealthWorkerMsg(
                    worker_event_id=worker_event_id,
                )
            )
            serialized_msgs.append(serialized_msg)
            logger.debug(
                f"Sending health check operation to worker ({instance_id}, {worker_id})"
            )

        # Collect results from all workers
        serialized_results = await self.execute_workers(
            sockets=sockets,
            serialized_msgs=serialized_msgs,
        )

        # Process results
        error_codes = {}
        for i, serialized_result in enumerate(serialized_results):
            try:
                result = msgspec.msgpack.decode(serialized_result, type=Msg)
                if isinstance(result, HealthWorkerRetMsg):
                    error_codes[worker_ids[i]] = result.error_code
                elif isinstance(result, ErrorMsg):
                    error_codes[worker_ids[i]] = -1001  # Worker returned error
                else:
                    error_codes[worker_ids[i]] = -1002  # Unexpected response
            except Exception as e:
                logger.error(
                    f"Failed to parse health response from worker "
                    f"{worker_ids[i]}: {str(e)}"
                )
                error_codes[worker_ids[i]] = -1003  # Failed to parse response

        return HealthRetMsg(
            event_id=msg.event_id,
            error_codes=error_codes,
        )

    async def check_finish(
        self, msg: CheckFinishMsg
    ) -> Union[CheckFinishRetMsg, ErrorMsg]:
        raise NotImplementedError

    # TODO(Jiayi): need to make the types more specific
    async def execute(self, operation: str, msg: MsgBase) -> MsgBase:
        """
        Execute a cache operation with error handling.

        :param operation: The operation to execute
        (e.g., 'clear').
        :param msg: The message containing the operation details.
        :return: The result of the operation or an error message.
        """
        try:
            method = getattr(self, operation)
            return await method(msg)
        except AttributeError:
            return ErrorMsg(error=f"Operation '{operation}' is not supported.")
        except Exception as e:
            return ErrorMsg(error=str(e))

    async def execute_workers(
        self,
        sockets: list[zmq.asyncio.Socket],
        serialized_msgs: list[bytes],
    ) -> list[bytes]:
        """
        Execute a list of serialized messages on the given sockets.
        :param sockets: The list of sockets to send the messages to.
        :param serialized_msgs: The list of serialized messages to send.
        :return: A list of serialized results received from the sockets.
        """
        tasks = []
        for socket, serialized_msg in zip(sockets, serialized_msgs, strict=False):

            async def send_and_receive(s, msg):
                await s.send(msg)
                return await s.recv()

            tasks.append(send_and_receive(socket, serialized_msg))

        serialized_results = await asyncio.gather(*tasks)
        return serialized_results
