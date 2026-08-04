# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import random
import time

# Third Party
from tqdm import tqdm
import numpy as np
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


def generate_test_tokens(num_chunks: int, chunk_size: int) -> torch.Tensor:
    """Generate test tokens for testing.
    The sequence is [0, 1, 2, ..., num_chunks * chunk_size - 1]
    """
    # Create sequential tokens for testing
    return torch.arange(
        0, num_chunks * chunk_size, dtype=torch.long, device=torch_device_type
    )


def generate_kv_cache_paged_list_tensors(
    num_blocks, device, block_size=16, dtype=torch.bfloat16
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [2, num_blocks, block_size, num_heads, head_size]

    for i in range(num_layers):
        # Use a fixed seed for reproducibility between sender and receiver
        torch.manual_seed(42 + i)
        kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)

    return ret


def fill_kv_cache_with_pattern(kv_cache, slot_mapping, pattern_value=0.99):
    """Fill the KV cache at the specified slot mappings with
    a recognizable pattern
    """
    print(slot_mapping.shape)
    for layer_idx, layer_tensor in tqdm(enumerate(kv_cache), total=len(kv_cache)):
        # Fill both K and V with the pattern value
        num_blocks = layer_tensor.shape[1]
        block_size = layer_tensor.shape[2]
        new_shape = (2, num_blocks * block_size, 8, 128)
        layer_tensor.reshape(new_shape)[:, slot_mapping, :, :] = pattern_value

    return kv_cache


def verify_kv_cache_pattern(kv_cache, slot_mapping, pattern_value=0.99, tolerance=0.01):
    """Verify that the KV cache contains the expected pattern at the
    specified slot mappings
    """
    logger.info(f"Verifying KV cache pattern {pattern_value}")
    all_correct = True
    for layer_idx, layer_tensor in tqdm(enumerate(kv_cache), total=len(kv_cache)):
        num_blocks = layer_tensor.shape[1]
        block_size = layer_tensor.shape[2]
        new_shape = (2, num_blocks * block_size, 8, 128)
        actual_values = layer_tensor.reshape(new_shape)[:, slot_mapping, :, :]
        # Check if the mean is close to the pattern value
        mean_value = actual_values.mean().item()
        if abs(mean_value - pattern_value) > tolerance:
            logger.error(
                f"Pattern mismatch at layer {layer_idx}: "
                f"expected mean ~{pattern_value}, got {mean_value}"
            )
            all_correct = False

    return all_correct


def calculate_throughput(total_bytes: int, elapsed_time: float) -> float:
    """Calculate throughput in GB/s"""
    if elapsed_time == 0:
        return float("inf")
    gb = total_bytes / (1024 * 1024 * 1024)
    return gb / elapsed_time


def create_config(role: str, host: str, port: int) -> LMCacheEngineConfig:
    """Create a configuration for the LMCacheEngine with Nixl backend."""
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=False,  # Nixl requires local_cpu=False
        max_local_cpu_size=0,  # Nixl requires max_local_cpu_size=0
        local_disk=None,  # Nixl requires local_disk=None
        max_local_disk_size=0,  # Nixl requires max_local_disk_size=0
        remote_url=None,  # Nixl requires remote_url=None
        remote_serde=None,  # Nixl requires remote_serde=None
        save_decode_cache=False,  # Nixl requires save_decode_cache=False
        enable_p2p=False,  # Nixl requires enable_p2p=False
        enable_nixl=True,  # Enable Nixl
        nixl_role=role,  # 'sender' or 'receiver'
        nixl_receiver_host=host,
        nixl_receiver_port=port,
        nixl_buffer_size=2**30,  # 1GB
        nixl_buffer_device=torch_device_type,
    )
    return config


def create_metadata() -> LMCacheMetadata:
    """Create metadata for the LMCacheEngine."""
    # Define KV shape: (num_layers, 2, chunk_size, num_heads, head_dim)
    chunk_size = 256
    num_layers = 32
    num_heads = 32
    head_dim = 128
    kv_shape = (num_layers, 2, chunk_size, num_heads, head_dim)

    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LMCacheEngine with Nixl backend")
    parser.add_argument(
        "--role",
        type=str,
        required=True,
        choices=["sender", "receiver"],
        help="Role of this instance (sender or receiver)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host name/IP for connection",
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port number for connection"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=10, help="Number of chunks to send"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of times to run the experiment",
    )
    args = parser.parse_args()

    # Set fixed random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Create configuration and metadata
    config = create_config(args.role, args.host, args.port)
    metadata = create_metadata()

    # Parameters for paged KV cache
    num_blocks = 10000
    block_size = 16
    dtype = torch.bfloat16
    device = torch_device_type

    max_chunks = num_blocks * block_size // config.chunk_size
    assert args.num_chunks <= max_chunks, f"Number of chunks must be <= {max_chunks}"

    # Create the VLLMPagedMemGPUConnectorV2
    hidden_dim = 1024
    num_layers = 32
    gpu_connector = VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)

    # Calculate the expected total size of data
    kv_shape = gpu_connector.get_shape(config.chunk_size)
    element_size = torch.tensor([], dtype=metadata.kv_dtype).element_size()
    chunk_size_bytes = torch.prod(torch.tensor(kv_shape)).item() * element_size
    total_size = chunk_size_bytes * args.num_chunks

    # Create the LMCacheEngine (will be reused across rounds)
    engine = LMCacheEngineBuilder.get_or_create(
        "test_engine",
        config,
        metadata,
        gpu_connector,
        mock_up_broadcast_fn,
        mock_up_broadcast_object_fn,
    )

    # Generate or create buffers that will be reused across rounds
    tokens = generate_test_tokens(args.num_chunks, config.chunk_size)
    slot_indices = list(range(0, num_blocks * block_size))
    random.shuffle(slot_indices)
    slot_mapping = torch.tensor(slot_indices[: len(tokens)], device=device)

    if args.role == "sender":
        # Generate KV cache once and reuse
        kv_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype
        )
        pattern_value = 0.99
        kv_cache = fill_kv_cache_with_pattern(kv_cache, slot_mapping, pattern_value)
    else:  # receiver
        # Create retrieval buffer once and reuse
        retrieved_cache = generate_kv_cache_paged_list_tensors(
            num_blocks, device, block_size, dtype
        )

    # Track statistics across rounds
    throughputs = []

    for round_num in range(args.num_rounds):
        logger.info(f"\nStarting round {round_num + 1}/{args.num_rounds}")

        if args.role == "sender":
            # Wait a bit for the receiver to set up
            time.sleep(2)

            logger.info(f"Storing {len(tokens)} tokens ({args.num_chunks} chunks)...")
            start_time = time.time()
            engine.store(tokens, kvcaches=kv_cache, slot_mapping=slot_mapping)

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Stored {len(tokens)} tokens in {elapsed_time:.6f} seconds")
            throughput = calculate_throughput(total_size, elapsed_time)
            logger.info(f"Throughput: {throughput:.2f} GB/s")
            throughputs.append(throughput)

        else:  # receiver
            # Wait for data to be received
            logger.info("Waiting to receive data...")

            # Poll until we receive all chunks or timeout
            received_count = 0
            start_time = time.time()
            timeout = 60  # seconds

            while received_count < args.num_chunks:
                # Check how many chunks we've received by looking up tokens
                received_count = engine.lookup(tokens) // config.chunk_size

                if received_count == args.num_chunks:
                    break

                # Check for timeout
                if time.time() - start_time > timeout:
                    logger.error(
                        "Timed out waiting for data. Received only "
                        f"{received_count}/{args.num_chunks} chunks."
                    )
                    break

                time.sleep(0.1)  # Small sleep to avoid busy waiting

            if received_count == args.num_chunks:
                logger.info(f"Received all {args.num_chunks} chunks")

                # Retrieve and verify the data
                logger.info("Retrieving and verifying data...")
                start_time = time.time()

                # Retrieve tokens from the cache engine
                ret_mask = engine.retrieve(
                    tokens, kvcaches=retrieved_cache, slot_mapping=slot_mapping
                )

                end_time = time.time()
                elapsed_time = end_time - start_time

                # Check if all tokens were retrieved
                retrieved_tokens = torch.sum(ret_mask).item()
                if retrieved_tokens == len(tokens):
                    logger.info(
                        f"Successfully retrieved all {retrieved_tokens} tokens "
                        f"in {elapsed_time:.6f} seconds"
                    )
                    throughput = calculate_throughput(total_size, elapsed_time)
                    logger.info(f"Retrieval throughput: {throughput:.2f} GB/s")

                    # Verify the data by checking if the retrieved KV cache
                    # has the expected pattern
                    pattern_value = 0.99  # Same value used by sender
                    if verify_kv_cache_pattern(
                        retrieved_cache, slot_mapping, pattern_value
                    ):
                        logger.info(
                            "✅ Data verification successful - pattern matches!"
                        )
                    else:
                        logger.error(
                            "❌ Data verification failed - pattern doesn't match!"
                        )
                else:
                    logger.error(
                        "Failed to retrieve all tokens. Retrieved "
                        f"{retrieved_tokens}/{len(tokens)} tokens."
                    )
            else:
                logger.error(f"Only received {received_count}/{args.num_chunks} chunks")

        # Wait between rounds
        time.sleep(2)

    # Print summary statistics
    if throughputs:
        mean_throughput = sum(throughputs) / len(throughputs)
        std_throughput = np.std(throughputs) if len(throughputs) > 1 else 0
        logger.info("\nSummary Statistics:")
        logger.info(
            f"Mean throughput: {mean_throughput:.2f} ± {std_throughput:.2f} GB/s"
        )
        logger.info(f"Min throughput: {min(throughputs):.2f} GB/s")
        logger.info(f"Max throughput: {max(throughputs):.2f} GB/s")

    # Cleanup at the very end
    LMCacheEngineBuilder.destroy("test_engine")
    logger.info("All rounds completed")
