# SPDX-License-Identifier: Apache-2.0
"""
LMCache Standalone Starter

A standalone starter for LMCacheEngine that:
- Loads configuration from YAML file or environment variables
- Supports command-line parameter overrides
- Starts a real LMCacheEngine instance
- Works without vLLM or GPU
- Supports all backend types (CPU, Disk, P2P, Remote, etc.)
- Optionally starts internal API server for remote access
"""

# Standard
from typing import Any, Dict, List, Optional, Tuple
import argparse
import ast
import asyncio
import os
import signal
import sys

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes, lmcache_get_or_create_config
from lmcache.logging import init_logger
from lmcache.utils import EngineType, mock_up_broadcast_fn, mock_up_broadcast_object_fn
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import parse_command_line_extra_params
from lmcache.v1.gpu_connector import CreateGPUConnector
from lmcache.v1.kv_layer_groups import DEFAULT_LAYER_NAME_PREFIX
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.standalone.manager import StandaloneLMCacheManager

# Third Party - Platform detection
try:
    # Third Party
    from vllm.platforms import current_platform
except ImportError:
    # Fallback for when vLLM is not available
    current_platform = None

logger = init_logger(__name__)

dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
}


class LayerGroupSpec:
    """Specification for a layer group with KV shape and dtype

    Attributes:
        layer_count: Number of layers in this group
        shape: Shape as tuple of integers
         may be (num_layers, kv_dim, num_blocks, num_heads, head_size)
        dtype: Data type for this layer group
    """

    def __init__(
        self,
        layer_count: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.layer_count = layer_count
        # May be (num_layers, kv_dim, num_blocks, num_heads, head_size)
        self.shape = shape
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"LayerGroupSpec({self.shape}):{self.dtype}:{self.layer_count}"


def parse_kvcache_shape_spec(spec_str: str) -> List[LayerGroupSpec]:
    """Parse KV shape specification with multiple layer groups.

    Format examples:
    - "(2,2,256,4,16):float16:2" (single group)
    - "(2,2,256,4,16):float16:2;(3,2,256,4,4):bfloat16:2" (two groups)

    Note: The shape string (inside parentheses) is not parsed and kept as string
    to support different Attention implementations with varying shapes.

    Returns a list of LayerGroupSpec objects.
    """
    if not spec_str:
        raise ValueError("KV shape specification cannot be empty")

    groups = []

    # Split by semicolon to get individual group specifications
    group_specs = spec_str.split(";")

    for group_spec in group_specs:
        group_spec = group_spec.strip()
        if not group_spec:
            continue

        # Parse format: (shape_string):dtype:layer_count
        if not (group_spec.startswith("(") and "):" in group_spec):
            raise ValueError(f"Invalid group specification format: {group_spec}")

        # Extract shape string inside parentheses and parse it
        shape_end = group_spec.find(")")
        shape_str = group_spec[1:shape_end]

        # Extract dtype and layer_count after the shape
        remaining = group_spec[shape_end + 2 :]  # Skip "):"
        parts = remaining.split(":")

        if len(parts) != 2:
            raise ValueError(f"Invalid group specification format: {group_spec}")

        dtype_str = parts[0].strip()
        layer_count_str = parts[1].strip()

        try:
            # Parse shape tuple - support arbitrary dimensions
            shape_parts = shape_str.split(",")
            shape = tuple(int(part.strip()) for part in shape_parts)
            layer_count = int(layer_count_str)
            dtype = dtype_map.get(dtype_str.strip().lower(), torch.float16)

            # Create LayerGroupSpec with parsed shape
            groups.append(LayerGroupSpec(layer_count, shape, dtype))
        except ValueError as e:
            raise ValueError(
                f"Invalid number format in group specification: {group_spec}"
            ) from e

    if not groups:
        raise ValueError("No valid layer groups found in specification")

    return groups


def calculate_composite_kv_cache_shape(
    layer_groups: List[LayerGroupSpec],
) -> Tuple[int, int, int, int, int]:
    """Calculate composite KV cache shape from multiple layer groups.

    Returns a shape that represents the KV cache structure:
    - num_layers: sum of all layer counts
    - kv_dim: from first group's shape (assumed consistent)
    - num_blocks: from first group's shape (assumed consistent)
    - num_heads: maximum num_heads across groups
    - head_size: maximum head_size across groups

    Note: This returns the KV cache shape, where the third dimension is num_blocks.
    For metadata KV shape, the third dimension should be chunk_size instead.
    """
    if not layer_groups:
        raise ValueError("No layer groups provided")

    # Get base dimensions from first group's shape
    first_shape = layer_groups[0].shape
    base_num_layers, base_kv_dim, base_num_blocks, base_num_heads, base_head_size = (
        first_shape
    )

    total_layers = sum(group.layer_count for group in layer_groups)

    # Find maximum num_heads and head_size across all groups
    max_num_heads = base_num_heads
    max_head_size = base_head_size

    for group in layer_groups[1:]:
        shape = group.shape
        num_heads = shape[3]
        head_size = shape[4]
        max_num_heads = max(max_num_heads, num_heads)
        max_head_size = max(max_head_size, head_size)

    return (total_layers, base_kv_dim, base_num_blocks, max_num_heads, max_head_size)


def get_composite_kv_dtype(layer_groups: List[LayerGroupSpec]) -> torch.dtype:
    """Get a representative dtype for composite KV cache.

    Returns the dtype of the first layer group for compatibility.
    """
    if not layer_groups:
        raise ValueError("No layer groups provided")
    return layer_groups[0].dtype


class LMCacheStandaloneStarter:
    """Standalone starter for LMCacheEngine"""

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        layer_groups: List[LayerGroupSpec],
        device: str = "cpu",
    ):
        self.config = config
        self.metadata = metadata
        self.layer_groups = layer_groups
        self.device = device

        gpu_connector = CreateGPUConnector(config, metadata, EngineType.MOCK)

        # Create standalone manager directly
        self._manager = StandaloneLMCacheManager(
            config=config,
            metadata=metadata,
            gpu_connector=gpu_connector,
            broadcast_fn=mock_up_broadcast_fn,
            broadcast_object_fn=mock_up_broadcast_object_fn,
            connector=self,
        )

        self.running = False

    @property
    def lmcache_engine(self) -> LMCacheEngine:
        """Get the LMCache engine instance."""
        # Directly access engine from standalone manager
        engine = self._manager.lmcache_engine
        assert engine is not None, "LMCache engine not initialized yet"
        return engine

    @property
    def api_server(self):
        """Get the API server instance."""
        return self._manager.api_server

    def _generate_fixed_kvcaches(self, device: str = "cpu") -> dict:
        """Generate fixed pattern kvcaches for testing and MD5 verification.

        Supports both single group and multiple layer groups.

        Args:
            device: Device to create tensors on (default: "cpu")
        """
        return self._generate_multi_group_kvcaches(device=device)

    def _generate_multi_group_kvcaches(self, device: str = "cpu") -> dict:
        """Generate kvcaches for multiple layer groups configuration

        Args:
            device: Device to create tensors on (default: "cpu")
        """
        if not self.layer_groups:
            raise ValueError("No layer groups specified for multi-group generation")

        kvcaches = {}
        current_layer = 0

        for group_idx, group in enumerate(self.layer_groups):
            # group.shape is already the final tensor shape
            tensor_shape = list(group.shape)

            for layer_in_group in range(group.layer_count):
                layer_idx = current_layer + layer_in_group
                torch.manual_seed(42 + layer_idx)
                tensor = torch.rand(tensor_shape, dtype=group.dtype, device=device)
                layer_name = f"{DEFAULT_LAYER_NAME_PREFIX}{layer_idx}"
                kvcaches[layer_name] = tensor

            current_layer += group.layer_count
            logger.info(
                "Generated layer group %d: %d layers, shape=%s, dtype=%s, device=%s",
                group_idx,
                group.layer_count,
                tensor_shape,
                group.dtype,
                device,
            )

        total_layers = current_layer
        logger.info(
            "Generated multi-group kvcaches: %d total layers across %d groups, "
            "device=%s",
            total_layers,
            len(self.layer_groups),
            device,
        )
        return kvcaches

    def start(self) -> LMCacheEngine:
        """Start the LMCache engine"""
        logger.info("=" * 80)
        logger.info("Starting LMCache Standalone Engine")
        logger.info("=" * 80)

        logger.info("Configuration: %s", self.config)
        logger.info("Metadata: %s", self.metadata)

        if self.layer_groups:
            logger.info("Layer groups: %s", self.layer_groups)

        # Calculate and log chunk storage size
        chunk_size = self.config.chunk_size
        num_layers, kv_dim, _, num_heads, head_size = self.metadata.kv_shape
        chunk_shape = torch.Size([num_layers, kv_dim, chunk_size, num_heads, head_size])
        chunk_storage_bytes = get_size_bytes([chunk_shape], [self.metadata.kv_dtype])
        chunk_storage_mb = chunk_storage_bytes / (1024 * 1024)
        logger.info(
            "Chunk storage size: %d bytes (%.2f MB) for chunk_size=%d",
            chunk_storage_bytes,
            chunk_storage_mb,
            chunk_size,
        )

        instance_id = self.config.lmcache_instance_id
        logger.info("Starting LMCache engine with instance ID: %s", instance_id)

        # Generate fixed pattern kvcaches for testing
        kv_caches = self._generate_fixed_kvcaches(device=self.device)
        self.kv_caches = kv_caches

        # Post-initialize with kvcaches and start API server
        self._manager.post_init()
        logger.info("LMCache engine post-initialized with fixed kvcaches")

        self._manager.start_services()

        self.running = True
        logger.info("LMCache engine started successfully")
        return self.lmcache_engine

    def stop(self):
        """Stop the LMCache engine"""
        if not self.running:
            return

        logger.info("Stopping LMCache engine...")
        self.running = False

        self._manager.stop_services()

        logger.info("LMCache engine stopped")

    async def run_forever(self):
        """Keep the engine running"""
        logger.info("=" * 80)
        logger.info("LMCache engine is running")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)

        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()


def load_config(config_file: Optional[str] = None) -> LMCacheEngineConfig:
    """Load configuration from file or environment"""
    config_file = config_file or os.getenv("LMCACHE_CONFIG_FILE")

    if config_file:
        logger.info(f"Loading configuration from file: {config_file}")
        config = LMCacheEngineConfig.from_file(config_file)
    else:
        logger.info("No config file specified, loading from environment variables")
        config = LMCacheEngineConfig.from_env()

    config.validate()
    config.log_config()

    return config


def override_config_from_dict(config: LMCacheEngineConfig, overrides: Dict[str, Any]):
    """Override configuration with dictionary"""
    for key, value in overrides.items():
        if hasattr(config, key):
            old_value = getattr(config, key)
            if key == "extra_config":
                # convert to dict
                value = ast.literal_eval(value)
                setattr(config, key, value)
            else:
                setattr(config, key, value)
            if old_value != value:
                logger.info(f"Override config: {key} = {value} (was {old_value})")
        else:
            logger.warning(f"Unknown config key: {key}, ignoring")


def parse_kv_shape(shape_str: str) -> Tuple[int, int, int, int, int]:
    """Parse KV shape from string like '32,2,256,32,128'"""
    try:
        parts = tuple(int(x.strip()) for x in shape_str.split(","))
        if len(parts) != 5:
            raise ValueError(
                f"kv_shape must have exactly 5 dimensions, got {len(parts)}"
            )
        return parts  # type: ignore[return-value]
    except ValueError as e:
        raise ValueError(f"Invalid kv_shape format: {shape_str}. Error: {e}") from e


def setup_signal_handlers(starter: LMCacheStandaloneStarter):
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        starter.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="LMCache Standalone Starter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to LMCache configuration file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="standalone_model",
        help="Model name for cache identification",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker ID for distributed setup",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of workers",
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        choices=["float16", "float32", "bfloat16", "uint8"],
        default="float16",
        help="KV cache data type",
    )
    parser.add_argument(
        "--kv-shape",
        type=str,
        default="2,2,256,4,16",
        help=(
            "KV cache shape as comma-separated integers "
            "(num_layer, 2 or 1, chunk_size, num_kv_head, head_size). "
            "num_layers: number of transformer layers, "
            "kv_dim: dimension for K/V (usually 2), "
            "chunk_size: number of memory chunks, "
            "num_heads: number of attention heads, "
            "head_size: size of each attention head. "
            "Example: '2,2,256,4,16' means 2 layers, 2 for K/V, "
            "256 chunks, 4 heads, 16 head size"
        ),
    )
    parser.add_argument(
        "--kvcache-shape-spec",
        type=str,
        default="(2,2,256,4,16):float16:2",
        help=(
            "KV cache shape specification with multiple layer groups. "
            "Format: '(shape_string):dtype:layer_count;[...]'. "
            "shape_string: comma-separated shape (e.g., '2,2,256,4,16'). "
            "Examples: '(2,2,256,4,16):float16:2' (single group), "
            "'(2,2,256,4,16):float16:2;(3,2,256,4,4):float32:3' (two groups)"
        ),
    )
    parser.add_argument(
        "--use-mla",
        action="store_true",
        help="Enable MLA (Multi-Level Attention)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (default: cpu)",
    )

    args, extra = parser.parse_known_args()
    args.extra_params = parse_command_line_extra_params(extra)

    return args


def main():
    """Main entry point"""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("LMCache Standalone Starter")
    logger.info("=" * 80)

    try:
        config_path = args.config
        if config_path:
            logger.info("Loading LMCache config file: %s", config_path)
            config = LMCacheEngineConfig.from_file(config_path)
            # Allow environment variables to override file settings
            config.update_config_from_env()
        else:
            logger.info("No config file specified, loading from environment variables.")
            config = lmcache_get_or_create_config()
        # Override with any extra command-line parameters
        if args.extra_params:
            override_config_from_dict(config, args.extra_params)

        # Handle KV shape specification
        if not args.kvcache_shape_spec:
            logger.error("--kvcache-shape-spec is required")
            sys.exit(1)
        else:
            # Use new multi-group specification
            layer_groups = parse_kvcache_shape_spec(args.kvcache_shape_spec)
            logger.info("Using KV shape specification: %s", args.kvcache_shape_spec)
            for i, group in enumerate(layer_groups):
                logger.info(
                    "  Group %d: %d layers, shape=%s, dtype=%s",
                    i,
                    group.layer_count,
                    group.shape,
                    group.dtype,
                )

        # Use single group specification - kv-shape directly assigned to metadata
        kv_dtype = dtype_map.get(args.kv_dtype, torch.float16)
        kv_shape = parse_kv_shape(args.kv_shape)

        logger.info("Using KV shape: %s", kv_shape)
        logger.info(
            "  num_layers=%d, kv_dim=%d, num_blocks=%d, num_heads=%d, head_size=%d",
            kv_shape[0],
            kv_shape[1],
            kv_shape[2],
            kv_shape[3],
            kv_shape[4],
        )

        metadata = LMCacheMetadata(
            model_name=args.model_name,
            world_size=args.world_size,
            local_world_size=args.world_size,
            worker_id=args.worker_id,
            local_worker_id=args.worker_id,
            kv_dtype=kv_dtype,
            kv_shape=kv_shape,
            use_mla=args.use_mla,
            role="worker",
        )

        starter = LMCacheStandaloneStarter(config, metadata, layer_groups, args.device)
        setup_signal_handlers(starter)

        starter.start()
        asyncio.run(starter.run_forever())

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("LMCache Standalone Starter stopped")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
