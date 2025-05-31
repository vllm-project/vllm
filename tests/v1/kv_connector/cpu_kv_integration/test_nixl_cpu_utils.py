# SPDX-License-Identifier: Apache-2.0
import time

import pytest
import torch
import torch.multiprocessing as mp

import vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils as utils
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
    DestinationSpec, NixlCPUReceiver, NixlCPUSender, RingBufferAllocator,
    SourceSpec)

try:
    #from nixl._api import nixl_agent as NixlWrapper
    import importlib
    spec = importlib.util.find_spec("nixl._api")
    if spec is None:
        raise ImportError("NIXL is not available")
    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False


def run_receiver(buffer_config, host, base_port, rank, ready_event,
                 stop_event):
    """Process function for running the receiver."""
    try:
        # Mock tensor_model_parallel_rank for this process
        utils.get_tensor_model_parallel_rank = lambda: rank

        # Create ring buffer allocator
        allocator = utils.RingBufferAllocator(
            size=buffer_config['buffer_size'],
            align_to=buffer_config['nixl_page_size'])

        # Create and start receiver
        receiver = NixlCPUReceiver(
            allocator=allocator,
            nixl_page_size=buffer_config['nixl_page_size'])
        receiver.start_handshake_listener(host, base_port)

        # Signal receiver is ready
        ready_event.set()

        # Wait for stop signal
        stop_event.wait()

        # Cleanup
        receiver.stop_handshake_listener()

    except Exception as e:
        print(f"Receiver process error: {e}")
        raise


def run_sender(buffer_config, host, base_port, rank, receiver_ready_event):
    """Process function for running the sender."""
    try:
        # Mock tensor_model_parallel_rank for this process
        utils.get_tensor_model_parallel_rank = lambda: rank

        # Create ring buffer allocator
        allocator = utils.RingBufferAllocator(
            size=buffer_config['buffer_size'],
            align_to=buffer_config['nixl_page_size'])

        # Wait for receiver to be ready
        receiver_ready_event.wait()

        # Create sender and perform handshake
        sender = NixlCPUSender(buffer_size=buffer_config['buffer_size'],
                               buffer_ptr=allocator.get_buffer_ptr(),
                               nixl_page_size=buffer_config['nixl_page_size'])

        dest_spec = DestinationSpec(rank=rank, host=host, base_port=base_port)
        sender._nixl_handshake(dest_spec)

        # Verify handshake results
        assert dest_spec.get_id() in sender._remote_agents
        assert sender._remote_agents[dest_spec.get_id()] is not None
        peer_name = sender._remote_agents[dest_spec.get_id()]
        assert sender._remote_xfer_handlers[peer_name] is not None

        return True
    except Exception as e:
        print(f"Sender process error: {e}")
        raise


def run_receiver_with_progress(buffer_config,
                               host,
                               base_port,
                               rank,
                               ready_event,
                               stop_event,
                               progress_interval=0.001):
    """Process function for running the receiver with progress loop."""
    try:
        # Mock tensor_model_parallel_rank for this process
        utils.get_tensor_model_parallel_rank = lambda: rank

        # Create ring buffer allocator
        allocator = utils.RingBufferAllocator(
            size=buffer_config['buffer_size'],
            align_to=buffer_config['nixl_page_size'])
        allocator._buffer.fill_(0)

        # Create and start receiver
        receiver = NixlCPUReceiver(
            allocator=allocator,
            nixl_page_size=buffer_config['nixl_page_size'])
        receiver.start_handshake_listener(host, base_port)

        # Signal receiver is ready
        ready_event.set()

        # Run progress loop until stop signal
        while not receiver.get_finished():
            receiver.progress()
            time.sleep(progress_interval)

        finished = receiver.get_finished(clear=True)
        assert len(finished) == 1
        source_spec, vaddr = finished[0]
        paddr = allocator.virtual_to_physical(vaddr)

        # Check if the numbers are all correct (should be uint8 all 1)
        num_elements = source_spec.get_size()
        should_1 = allocator._buffer[paddr:paddr + num_elements]
        should_0_a = allocator._buffer[:paddr]
        should_0_b = allocator._buffer[paddr + num_elements:]
        assert (should_1 == 1).all(), "Buffer data mismatch"
        if len(should_0_a) > 0:
            assert (should_0_a == 0).all(), "Buffer data mismatch"
        if len(should_0_b) > 0:
            assert (should_0_b == 0).all(), "Buffer data mismatch"

        while not stop_event.is_set():
            receiver.progress()
            time.sleep(progress_interval)

        # Cleanup
        receiver.stop_handshake_listener()

    except Exception as e:
        print(f"Receiver process error: {e}")
        raise


def run_sender_with_protocol(buffer_config, host, base_port, rank,
                             receiver_ready_event, success_event):
    """Process function for running the sender with protocol communication."""
    try:
        # Mock tensor_model_parallel_rank for this process
        utils.get_tensor_model_parallel_rank = lambda: rank

        # Create ring buffer allocator
        allocator = utils.RingBufferAllocator(
            size=buffer_config['buffer_size'],
            align_to=buffer_config['nixl_page_size'])

        # Wait for receiver to be ready
        receiver_ready_event.wait()

        # Create sender
        sender = NixlCPUSender(buffer_size=buffer_config['buffer_size'],
                               buffer_ptr=allocator.get_buffer_ptr(),
                               nixl_page_size=buffer_config['nixl_page_size'])

        # Create destination spec and perform handshake
        dest_spec = DestinationSpec(rank=rank, host=host, base_port=base_port)
        sender._nixl_handshake(dest_spec)

        # Create source spec and prepare send
        source_spec = SourceSpec(
            request_id="test_request",
            layer_id=0,
            start=0,
            stop=16,  # Assuming we want to send 16 tokens
            shape=(2, 1, 16, 8, 128),  # Example shape 
            dtype_str="bfloat16",  # Example dtype
            num_all_tokens=16,
        )

        # Prepare send and wait for completion
        uid = sender.prepare_send(source_spec, dest_spec)

        max_retries = 100
        retry_count = 0
        remote_agent = None

        while retry_count < max_retries:
            remote_agent, receiver_paddr = \
                    sender.check_and_remove_prepared_send(uid)
            if remote_agent is not None:
                break
            time.sleep(0.1)
            retry_count += 1

        assert remote_agent is not None, "Failed to get remote agent"
        assert receiver_paddr != -1, "Failed to get receiver virtual address"

        # Test the real send
        vaddr, buffer = allocator.allocate(source_spec.get_size())
        paddr = allocator.virtual_to_physical(vaddr)

        buffer.fill_(1)  # Fill with dummy data

        handle = sender.send(paddr, receiver_paddr, source_spec.get_size(),
                             uid, dest_spec)

        while not sender.is_send_finished(handle):
            time.sleep(0.1)
        print("Send completed successfully")

        if remote_agent is not None:
            success_event.set()

    except Exception as e:
        print(f"Sender process error: {e}")
        raise


@pytest.mark.skipif(not NIXL_AVAILABLE, reason="NIXL is not available")
class TestNixlCPUUtils:
    """Test cases for NixlCPUSender and NixlCPUReceiver."""

    @classmethod
    def setup_class(cls):
        """Set up the test class."""
        pass

    @pytest.fixture
    def buffer_config(self):
        """Common buffer configuration for tests."""
        buffer_size = 1 << 20  # 1MB
        torch_buffer = torch.zeros(buffer_size,
                                   dtype=torch.uint8,
                                   device='cpu')

        return {
            'buffer_size': buffer_size,
            'buffer_ptr': torch_buffer.data_ptr(),
            'nixl_page_size': 4096  # Standard page size
        }

    def test_sender_creation(self, buffer_config):
        """Test creation of NixlCPUSender."""
        sender = NixlCPUSender(buffer_size=buffer_config['buffer_size'],
                               buffer_ptr=buffer_config['buffer_ptr'],
                               nixl_page_size=buffer_config['nixl_page_size'])

        # Verify internal state
        assert sender._buffer_size == buffer_config['buffer_size']
        assert sender._buffer_ptr == buffer_config['buffer_ptr']
        assert sender._nixl_page_size == buffer_config['nixl_page_size']
        assert isinstance(sender._remote_agents, dict)

        # Verify NIXL initialization
        assert sender._nixl_wrapper is not None
        assert sender._reg_dlist is not None
        assert sender._local_xfer_dlist is not None

    def test_receiver_creation(self, buffer_config):
        """Test creation of NixlCPUReceiver."""
        # Create ring buffer allocator
        allocator = RingBufferAllocator(
            size=buffer_config['buffer_size'],
            align_to=buffer_config['nixl_page_size'])

        receiver = NixlCPUReceiver(
            allocator=allocator,
            nixl_page_size=buffer_config['nixl_page_size'])

        # Verify internal state
        assert receiver._buffer_size == buffer_config['buffer_size']
        assert receiver._buffer_ptr == allocator.get_buffer_ptr()
        assert receiver._nixl_page_size == buffer_config['nixl_page_size']
        assert isinstance(receiver._inflight_requests, dict)
        assert isinstance(receiver._inflight_request_vaddr, dict)
        assert receiver._allocator is allocator

        # Verify NIXL initialization
        assert receiver._nixl_wrapper is not None
        assert receiver._reg_dlist is not None
        assert receiver._local_xfer_dlist is not None

    def test_nixl_handshake_multiprocess(self, buffer_config):
        """Test NIXL handshake between sender and receiver in separate 
        processes.
        """
        # Setup test parameters
        test_host = "127.0.0.1"
        test_base_port = 50051
        test_rank = 0

        old_start_method = mp.get_start_method(allow_none=True)
        mp.set_start_method("spawn", force=True)

        # Create events for process synchronization
        receiver_ready = mp.Event()
        stop_receiver = mp.Event()

        # Start receiver process
        receiver_process = mp.Process(target=run_receiver,
                                      args=(buffer_config, test_host,
                                            test_base_port, test_rank,
                                            receiver_ready, stop_receiver))
        receiver_process.start()

        # Start sender process
        sender_process = mp.Process(target=run_sender,
                                    args=(buffer_config, test_host,
                                          test_base_port, test_rank,
                                          receiver_ready))
        sender_process.start()

        try:
            # Wait for processes to complete
            sender_process.join(timeout=20)
            assert sender_process.exitcode == 0, "Sender process failed"

        finally:
            # Cleanup
            stop_receiver.set()
            receiver_process.join(timeout=5)

            # Force terminate if processes haven't exited
            if receiver_process.is_alive():
                receiver_process.terminate()
            if sender_process.is_alive():
                sender_process.terminate()

            mp.set_start_method(old_start_method, force=True)

    def test_nixl_protocol_communication(self, buffer_config):
        """Test the full protocol communication between sender and receiver."""
        # Setup test parameters
        test_host = "127.0.0.1"
        test_base_port = 50052
        test_rank = 0

        # Set multiprocessing start method
        old_start_method = mp.get_start_method(allow_none=True)
        mp.set_start_method("spawn", force=True)

        # Create events for process synchronization
        receiver_ready = mp.Event()
        stop_receiver = mp.Event()
        protocol_success = mp.Event()

        # Start receiver process with progress loop
        receiver_process = mp.Process(target=run_receiver_with_progress,
                                      args=(buffer_config, test_host,
                                            test_base_port, test_rank,
                                            receiver_ready, stop_receiver))
        receiver_process.start()

        # Start sender process with protocol communication
        sender_process = mp.Process(target=run_sender_with_protocol,
                                    args=(buffer_config, test_host,
                                          test_base_port, test_rank,
                                          receiver_ready, protocol_success))
        sender_process.start()

        try:
            # Wait for protocol communication to complete
            protocol_complete = protocol_success.wait(timeout=20)
            assert protocol_complete, \
                    "Protocol communication failed or timed out"

            # Wait for sender process to complete
            sender_process.join(timeout=5)
            assert sender_process.exitcode == 0, "Sender process failed"

        finally:
            # Cleanup
            stop_receiver.set()
            receiver_process.join(timeout=5)

            # Force terminate if processes haven't exited
            if receiver_process.is_alive():
                receiver_process.terminate()
            if sender_process.is_alive():
                sender_process.terminate()

            mp.set_start_method(old_start_method, force=True)
