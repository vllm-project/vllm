# SPDX-License-Identifier: Apache-2.0

import time

import torch.multiprocessing as mp

import vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils as utils
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_cpu_utils import (
    NixlDecodeManager)


def main():
    """Main function to run the receiver."""
    # Setup test parameters
    test_host = "127.0.0.1"
    test_base_port = 54321
    test_rank = 0
    expected_layers = 32

    # Buffer configuration
    buffer_size = 1 << 30  # 1GB

    try:
        # Mock tensor_model_parallel_rank for this process
        utils.get_tensor_model_parallel_rank = lambda: test_rank
        utils.get_tensor_model_parallel_world_size = lambda: 1
        utils.get_tp_group = lambda: None

        decoder_manager = NixlDecodeManager(buffer_size, test_host,
                                            test_base_port)

        print(f"Receiver started on {test_host}:{test_base_port}")

        # Run progress loop until interrupted
        try:
            while True:
                decoder_manager.progress()
                finished = decoder_manager.get_finished(expected_layers)
                print(f"Got {len(finished)} finished requests")

                for req_id in finished:
                    print(f"Processing finished request {req_id}")
                    for i in range(expected_layers):
                        decode_specs = decoder_manager.get_kv_specs(req_id, i)
                        for spec in decode_specs:
                            print(
                                f"Received layer {i} tokens "
                                f"{spec.start} - {spec.stop} request {req_id}. "
                                f"The shape is {spec.buffer.shape}. "
                                f"The digest is {spec.buffer.mean()}.")

                    decoder_manager.free_request(req_id)

                allocator = decoder_manager._allocator
                print("Allocator high/low watermark:",
                      allocator.high_watermark, allocator.low_watermark)
                time.sleep(1)  # Small sleep to prevent busy waiting

        except KeyboardInterrupt:
            decoder_manager.close()
            print("\nShutting down receiver...")

        print("Receiver stopped")

    except Exception as e:
        print(f"Receiver error: {e}")
        raise


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
