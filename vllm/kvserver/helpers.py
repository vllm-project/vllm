# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import zmq

from vllm.kvserver.protocol import (KVServerCmd, KVServerOffloadFinished,
                                    decode_cmd, decode_payload)
from vllm.logger import init_logger

logger = init_logger(__name__)


def scheduler_process_response(socket: zmq.Socket,
                               finished_offloads: list[str],
                               finished_onloads: list[str]):
    """A non-blocking function to process the offload/onload 
    finished responses from the server.

    Newly finished offload/onload requests are appended to
    the finished_offloads and finished_onloads lists.

    Args:
        socket (zmq.Socket): The zmq dealer socket in scheduler
    """
    while True:
        try:
            msg = socket.recv_multipart(flags=zmq.NOBLOCK)
            cmd = decode_cmd(msg[0])
            payload = decode_payload(cmd, msg[1])
            match cmd:
                case KVServerCmd.OFFLOAD_FINISHED:
                    assert isinstance(payload, KVServerOffloadFinished)
                    logger.debug(
                        "Offload finished for request_id=%s, success=%s",
                        payload.request_id, payload.success)
                    if payload.success:
                        finished_offloads.append(payload.request_id)

                case _:
                    logger.warning("Received unexpected command: %s", cmd)
        except zmq.Again:
            break
        except zmq.ZMQError as e:
            logger.error("ZMQError when receiving offload response: %s", e)
            break
