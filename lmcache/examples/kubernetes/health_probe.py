#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# This script implements a health check probe for the LMCache server.
# It performs the following functions:
# 1. Sends a health check request to the LMCache server
# 2. Verifies the server's response indicates successful operation
# 3. Serves as a Kubernetes liveness probe to monitor server health
# 4. Helps ensure the server is running and responsive

# Standard
import socket
import sys

# Third Party
import torch

# First Party
from lmcache.v1.protocol import (
    CacheEngineKey,
    ClientCommand,
    ClientMetaMessage,
    MemoryFormat,
    ServerMetaMessage,
    ServerReturnCode,
)


def main():
    if len(sys.argv) != 3:
        print("Usage: health_probe.py <host> <port>", file=sys.stderr)
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    try:
        # Create connection with timeout and ensure proper cleanup with context manager
        with socket.create_connection((host, port), timeout=5) as s:
            # Create and send health check message
            msg = ClientMetaMessage(
                ClientCommand.HEALTH,
                key=CacheEngineKey(
                    model_name="",
                    world_size=0,
                    worker_id=0,
                    chunk_hash=0,
                    dtype=torch.float16,
                ),
                length=0,
                fmt=MemoryFormat(1),
                dtype=torch.float16,
                shape=torch.Size((0, 0, 0, 0)),
            )
            s.sendall(msg.serialize())

            # Receive and parse response
            resp = s.recv(ServerMetaMessage.packlength())
            if not resp:
                print("No response received from server", file=sys.stderr)
                sys.exit(1)

            # Parse the response message
            meta = ServerMetaMessage.deserialize(resp)

            # Check if server responded with success
            if meta.code == ServerReturnCode.SUCCESS:
                sys.exit(0)
            else:
                print(f"Server returned error code: {meta.code}", file=sys.stderr)
                sys.exit(1)

    except socket.timeout:
        print("Connection timed out", file=sys.stderr)
        sys.exit(1)
    except ConnectionRefusedError:
        print("Connection refused - server may not be running", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during health check: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
