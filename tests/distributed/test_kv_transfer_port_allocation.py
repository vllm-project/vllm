# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for KV transfer port allocation race condition fix (issue #28636)."""

import socket
from concurrent.futures import ThreadPoolExecutor

import pytest
import zmq


@pytest.mark.skip_global_cleanup
class TestP2pNcclEnginePortAllocation:
    """Test suite for P2pNcclEngine port allocation with socket holding."""

    def test_socket_holding_import(self):
        """Test that we can import the socket holding function."""
        from vllm.utils.network_utils import get_and_hold_open_port

        assert get_and_hold_open_port is not None

    def test_socket_holding_prevents_port_reuse(self):
        """Test that the held socket prevents other processes from using the port.

        This is the core test for the race condition fix from issue #28636.
        """
        from vllm.utils.network_utils import get_and_hold_open_port

        # Allocate a port with socket holding
        sock1, port1 = get_and_hold_open_port()
        try:
            # Try to bind another socket to the same port - should fail
            s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(OSError) as exc_info:
                s2.bind(("", port1))
            s2.close()
            # EADDRINUSE: 48 on macOS, 98 on Linux
            assert exc_info.value.errno in (48, 98)
        finally:
            sock1.close()

    def test_concurrent_port_allocation(self):
        """Test that concurrent get_and_hold_open_port calls allocate unique ports."""
        from vllm.utils.network_utils import get_and_hold_open_port

        num_allocations = 10

        def allocate_port():
            return get_and_hold_open_port()

        with ThreadPoolExecutor(max_workers=num_allocations) as executor:
            futures = [executor.submit(allocate_port) for _ in range(num_allocations)]
            results = [f.result() for f in futures]

        try:
            ports = [port for _, port in results]
            # All ports should be unique
            assert len(set(ports)) == num_allocations
        finally:
            # Clean up sockets
            for sock, _ in results:
                sock.close()

    def test_multiple_zmq_sockets_use_unique_ports(self):
        """Test that multiple ZMQ sockets get unique ports via socket holding.

        This simulates the P2pNcclEngine pattern without requiring full instantiation.
        """
        from vllm.utils.network_utils import get_and_hold_open_port

        # Simulate two P2pNcclEngine instances (prefill and decode servers)
        contexts = []
        sockets = []
        held_sockets = []
        ports = []

        try:
            for i in range(2):
                # Allocate port with socket holding (what P2pNcclEngine now does)
                held_socket, port = get_and_hold_open_port()
                held_sockets.append(held_socket)
                ports.append(port)

                # Bind ZMQ socket to the allocated port
                ctx = zmq.Context()
                zmq_socket = ctx.socket(zmq.ROUTER)
                zmq_socket.bind(f"tcp://127.0.0.1:{port}")
                contexts.append(ctx)
                sockets.append(zmq_socket)

                # Close held socket after ZMQ binds (as per our fix)
                held_socket.close()

            # Both ZMQ sockets should have different ports
            assert ports[0] != ports[1], (
                f"ZMQ sockets should have unique ports, but both got: "
                f"{ports[0]} and {ports[1]}"
            )

        finally:
            # Clean up
            for sock in sockets:
                sock.close()
            for ctx in contexts:
                ctx.term()
            for held_sock in held_sockets:
                if not held_sock._closed:
                    held_sock.close()

    def test_zmq_port_conflict_scenario(self):
        """Simulate the exact scenario from issue #28636.

        Two ZMQ sockets trying to bind to the same port should fail.
        """
        from vllm.utils.network_utils import get_and_hold_open_port

        # Simulate prefill server allocating a port
        prefill_held_socket, prefill_port = get_and_hold_open_port()

        try:
            # Create ZMQ context and socket for "prefill server"
            prefill_ctx = zmq.Context()
            prefill_zmq_socket = prefill_ctx.socket(zmq.ROUTER)
            prefill_zmq_socket.bind(f"tcp://127.0.0.1:{prefill_port}")

            # Close the held socket after ZMQ binds (as per our fix)
            prefill_held_socket.close()

            # Now simulate decode server trying to allocate its own port
            # It should get a DIFFERENT port
            decode_held_socket, decode_port = get_and_hold_open_port()

            try:
                # The decode server should get a different port
                assert decode_port != prefill_port, (
                    f"Decode server got same port as prefill server: {decode_port}"
                )

                # Decode server can successfully bind
                decode_ctx = zmq.Context()
                decode_zmq_socket = decode_ctx.socket(zmq.ROUTER)
                decode_zmq_socket.bind(f"tcp://127.0.0.1:{decode_port}")

                # Close the decode held socket
                decode_held_socket.close()

                # Both servers should be running on different ports
                assert prefill_port != decode_port

                # Clean up
                decode_zmq_socket.close()
                decode_ctx.term()

            finally:
                if not decode_held_socket._closed:
                    decode_held_socket.close()

            # Clean up
            prefill_zmq_socket.close()
            prefill_ctx.term()

        finally:
            if not prefill_held_socket._closed:
                prefill_held_socket.close()
