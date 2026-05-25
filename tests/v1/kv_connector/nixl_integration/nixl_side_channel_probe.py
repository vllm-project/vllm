# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe a NIXL side-channel socket for handshake metadata readiness."""

import argparse
import ipaddress

import msgspec
import zmq

GET_META_MSG = b"get_meta_msg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--timeout-ms", default=1000, type=int)
    return parser.parse_args()


def make_zmq_path(host: str, port: int) -> str:
    try:
        if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
            return f"tcp://[{host}]:{port}"
    except ValueError:
        pass
    return f"tcp://{host}:{port}"


def main() -> None:
    args = parse_args()
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    try:
        sock.connect(make_zmq_path(args.host, args.port))
        sock.send(msgspec.msgpack.encode((GET_META_MSG, args.rank)))
        sock.recv()
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
