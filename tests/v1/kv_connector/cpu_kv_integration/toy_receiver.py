# SPDX-License-Identifier: Apache-2.0

import torch.multiprocessing as mp
import time

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
    NixlCPUReceiver, RingBufferAllocator
)

def main():
    """Main function to run the receiver."""
    # Setup test parameters
    test_host = "127.0.0.1"
    test_base_port = 54321
    test_rank = 0
    
    # Buffer configuration
    buffer_size = 1 << 30  # 1GB
    nixl_page_size = 4096  # Standard page size

    try:
        # Mock tensor_model_parallel_rank for this process
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils as utils
        utils.get_tensor_model_parallel_rank = lambda: test_rank

        # Create ring buffer allocator
        allocator = RingBufferAllocator(
            size=buffer_size,
            align_to=nixl_page_size
        )
        allocator._buffer.fill_(0)

        # Create and start receiver
        receiver = NixlCPUReceiver(
            allocator=allocator,
            nixl_page_size=nixl_page_size
        )
        receiver.start_handshake_listener(test_host, test_base_port)
        
        print(f"Receiver started on {test_host}:{test_base_port}")
        
        # Run progress loop until interrupted
        try:
            while True:
                receiver.progress()
                
                # Check for finished requests
                finished = receiver.get_finished(clear=True)
                if finished:
                    for source_spec, vaddr in finished:
                        print(f"Received data from request {source_spec.request_id}")
                        paddr = allocator.virtual_to_physical(vaddr)
                        
                        # Verify received data
                        num_elements = source_spec.get_size()
                        received_data = allocator._buffer[paddr : paddr + num_elements]
                        print(f"Received {num_elements} elements")
                        print(f"First few values: {received_data[:10]}")
                
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                
        except KeyboardInterrupt:
            print("\nShutting down receiver...")
        
        # Cleanup
        receiver.stop_handshake_listener()
        print("Receiver stopped")
        
    except Exception as e:
        print(f"Receiver error: {e}")
        raise

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main() 
