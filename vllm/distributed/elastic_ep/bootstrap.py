# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm import envs
from vllm.utils.network_utils import get_tcp_uri, make_zmq_socket

if TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs


class ExternalElasticEPLaunch(msgspec.Struct):
    data_parallel_size: int
    scale_up: bool
    identity: tuple[str, str] | None = None


class ExternalElasticEPBootstrapServer:
    """Reserve automatic launches in one existing startup or scale handshake."""

    def __init__(
        self, data_parallel_size: int, ranks: Collection[int], *, scale_up: bool
    ):
        self.data_parallel_size = data_parallel_size
        self.ranks = frozenset(ranks)
        self.scale_up = scale_up
        self.token = uuid.uuid4().hex
        self.claims: dict[int, str] = {}

    def discover(self, rank: int, launch_id: str) -> ExternalElasticEPLaunch:
        if type(rank) is not int or rank not in self.ranks:
            raise ValueError(f"DP rank {rank} is not an expected bootstrap participant")
        if not isinstance(launch_id, str) or not launch_id:
            raise ValueError("Automatic bootstrap requires a launch identity")
        if rank in self.claims and self.claims[rank] != launch_id:
            raise ValueError(f"DP rank {rank} is already reserved by another launch")
        self.claims[rank] = launch_id
        return ExternalElasticEPLaunch(
            self.data_parallel_size, self.scale_up, (self.token, launch_id)
        )

    def validate(self, rank: int, identity: list[str] | None) -> None:
        expected = self.claims.get(rank)
        if identity is None and expected is None:
            return  # Preserve explicitly configured legacy launches.
        if expected is None or identity != [self.token, expected]:
            raise ValueError("Engine launch does not match this bootstrap handshake")

    def reply(
        self, socket: zmq.Socket, identity: bytes, message: dict[str, Any]
    ) -> None:
        try:
            launch = self.discover(
                message.get("rank", -1), message.get("launch_id", "")
            )
            payload = msgspec.msgpack.encode(launch)
        except ValueError as error:
            payload = msgspec.msgpack.encode({"error": str(error)})
        socket.send_multipart((identity, payload))


def prepare_external_elastic_ep_launch(args: "EngineArgs") -> ExternalElasticEPLaunch:
    """Resolve initial or append startup before constructing the engine config.

    The declared DP size is the initial world size. Nonzero ranks discover
    their actual size through rank zero's existing handshake. This neither
    commits membership nor makes the process eligible to serve.
    """
    if (
        not args.enable_elastic_ep
        or not args.enable_expert_parallel
        or not args.enable_eplb
        or args.data_parallel_rank is None
        or args.data_parallel_hybrid_lb
        or args.data_parallel_start_rank is not None
        or args.data_parallel_multi_port_external_lb
        or args.data_parallel_backend != "mp"
        or args.data_parallel_size_local not in (None, 1)
        or args._api_process_count != 1
    ):
        raise ValueError(
            "Automatic Elastic EP bootstrap requires external MP DP with an "
            "explicit rank, one local DP rank and API server, EP and EPLB enabled"
        )
    initial_size = args.data_parallel_size
    maximum_size = args.elastic_ep_max_dp_size
    rank = args.data_parallel_rank
    if initial_size <= 1 or maximum_size is None or maximum_size < initial_size:
        raise ValueError("Automatic bootstrap requires 1 < initial DP size <= max DP")
    if not 0 <= rank < min(maximum_size, 2**16):
        raise ValueError("Automatic bootstrap rank must be below max DP size")
    if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
        raise ValueError("Automatic bootstrap owns VLLM_ELASTIC_EP_SCALE_UP_LAUNCH")
    address = args.data_parallel_address or args.master_addr or "127.0.0.1"
    port = (
        args.data_parallel_rpc_port
        if args.data_parallel_rpc_port is not None
        else 29550
    )
    if not 1 <= port <= 65535:
        raise ValueError("Automatic bootstrap requires a valid DP RPC port")
    if rank == 0:
        return ExternalElasticEPLaunch(initial_size, False)
    timeout = envs.VLLM_ENGINE_READY_TIMEOUT_S
    if timeout <= 0:
        raise ValueError("Automatic bootstrap requires a positive engine-ready timeout")
    launch_id = uuid.uuid4().hex
    with (
        zmq.Context() as ctx,
        make_zmq_socket(
            ctx,
            get_tcp_uri(address, port),
            zmq.DEALER,
            bind=False,
            identity=launch_id.encode(),
            linger=0,
        ) as socket,
    ):
        socket.send(
            msgspec.msgpack.encode(
                {"status": "BOOTSTRAP", "rank": rank, "launch_id": launch_id}
            )
        )
        if not socket.poll(timeout * 1000):
            raise TimeoutError(
                "No matching initial or scale-up handshake is available; "
                "automatic bootstrap does not recover an existing member"
            )
        payload = socket.recv()
    document = msgspec.msgpack.decode(payload)
    if "error" in document:
        raise ValueError(document["error"])
    launch = msgspec.msgpack.decode(payload, type=ExternalElasticEPLaunch)
    if (
        launch.identity is None
        or launch.identity[1] != launch_id
        or not rank < launch.data_parallel_size <= maximum_size
        or not launch.scale_up
        and (rank >= initial_size or launch.data_parallel_size != initial_size)
    ):
        raise ValueError(
            "Bootstrap handshake does not match the declared initial world"
        )
    return launch
