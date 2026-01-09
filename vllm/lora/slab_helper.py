# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import threading

# Create the LoRA model instance
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.lora.layers import FusedMoE3DWithLoRA

if TYPE_CHECKING:
    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.lora.lora_model import LoRAModel


from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights

# Import here to avoid circular dependency
from vllm.lora.utils import parse_fine_tuned_lora_name

logger = init_logger(__name__)

# Global slab cache
_GLOBAL_SLAB_CACHE: dict[str, tuple] = {}
_CACHE_LOCK = threading.RLock()

# Global LoRAModel cache for early checking
_GLOBAL_LORA_MODEL_CACHE: dict[str, Any] = {}
_LORA_MODEL_CACHE_LOCK = threading.RLock()


class UltraFastPinnedPool:
    """Lazy-initialized pinned memory pool."""

    def __init__(self):
        self.pool_size = 0
        self.pinned_pool = None  # Lazy - allocated on first use
        self.pool_lock = threading.RLock()
        self.used_ranges = []  # Track used memory ranges

        self.current_slab = None
        self.current_metadata = None

    def _ensure_capacity(self, required_bytes: int) -> None:
        """
        Ensure the pinned memory pool has sufficient capacity.

        Expands the pool if needed by doubling the size or adding required space.
        Preserves existing data when expanding the pool.

        Args:
            required_bytes: The number of bytes required
        """
        if required_bytes > self.pool_size:
            new_size = max(self.pool_size * 2, required_bytes + self.pool_size)
            new_pool = torch.empty(new_size, dtype=torch.uint8).pin_memory()

            # Copy existing data if any
            if self.used_ranges and self.pinned_pool is not None:
                total_used = max(end for start, end in self.used_ranges)
                new_pool[:total_used] = self.pinned_pool[:total_used]

            self.pinned_pool = new_pool
            self.pool_size = new_size

    def allocate_slab_views_directly(
        self, tensor_sizes: list[int], dtype: torch.dtype
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Allocate slab and return views - ZERO copy needed!"""
        total_elements = sum(tensor_sizes)

        if total_elements == 0:
            return torch.empty(0, dtype=dtype, device="cpu").pin_memory(), []

        tensor_bytes = total_elements * dtype.itemsize

        with self.pool_lock:
            # Expand pool if needed
            self._ensure_capacity(tensor_bytes)

            # Find available space
            start_offset = max((end for start, end in self.used_ranges), default=0)
            end_offset = start_offset + tensor_bytes

            if end_offset > self.pool_size:
                # Reset pool - reuse from beginning
                self.used_ranges.clear()
                start_offset = 0
                end_offset = tensor_bytes

            self.used_ranges.append((start_offset, end_offset))

            # Create full slab view
            assert self.pinned_pool is not None
            pool_slice = self.pinned_pool[start_offset:end_offset]
            full_slab = pool_slice.view(torch.uint8).view(dtype)[:total_elements]

            # Create individual tensor views for each component - NO copying!
            tensor_views = []
            current_offset = 0
            for size in tensor_sizes:
                if size > 0:
                    tensor_view = full_slab[current_offset : current_offset + size]
                    tensor_views.append(tensor_view)
                    current_offset += size
                else:
                    tensor_views.append(torch.empty(0, dtype=dtype, device="cpu"))

            return full_slab, tensor_views


# Global ultra-fast pool - initialized ONCE in envs.py
_ULTRA_FAST_POOL = None
_POOL_INIT_LOCK = threading.RLock()


def set_global_pool(pool: UltraFastPinnedPool) -> None:
    """Set the global pool instance."""
    global _ULTRA_FAST_POOL
    with _POOL_INIT_LOCK:
        if _ULTRA_FAST_POOL is None:
            _ULTRA_FAST_POOL = pool


def get_ultra_fast_pool():
    """Get the pre-initialized global pool - NO lazy initialization."""
    global _ULTRA_FAST_POOL
    if _ULTRA_FAST_POOL is None:
        # Fallback - create pool if not set (shouldn't happen)
        with _POOL_INIT_LOCK:
            if _ULTRA_FAST_POOL is None:
                _ULTRA_FAST_POOL = UltraFastPinnedPool()
    return _ULTRA_FAST_POOL


# Main public interface with CPU caching and disk save/load
def build_target_matched_slab(
    lora_model: "LoRAModel",
    target_modules: dict[str, "BaseLayerWithLoRA"] | None,
    max_loras: int,
    lora_config: "LoRAConfig | None",
    slab_path: str | None = None,
):
    """
    Build a slab that exactly matches the per-layer target shapes.
    Ultra-fast cached slab building with minimal overhead.
    Ensures perfect zero-copy during set_lora() and reuses slabs for identical LoRAs.

    Args:
        lora_model: The LoRA model to build slab for
        target_modules: Target modules dictionary
        max_loras: Maximum number of LoRAs
        lora_config: LoRA configuration
        slab_path: Optional path to save/load slab to/from disk
    """

    # Get TP info for cache key when fully_sharded=True
    fully_sharded = lora_config.fully_sharded_loras if lora_config else False
    tp_rank = None
    if fully_sharded and target_modules:
        first_module = next(iter(target_modules.values()), None)
        if first_module:
            tp_rank = getattr(first_module, "tp_rank", 0)

    cache_key = _generate_slab_cache_key(lora_model, "cpu", tp_rank, fully_sharded)

    # Get pre-initialized pool ONCE to avoid repeated calls
    pool = get_ultra_fast_pool()

    # Check CPU cache FIRST - if already on CPU, don't load again
    if cache_key in _GLOBAL_SLAB_CACHE:
        cached_slab, cached_metadata = _GLOBAL_SLAB_CACHE[cache_key]
        return cached_slab, cached_metadata

    # Only take lock if not in memory cache
    with _CACHE_LOCK:
        # Double-check pattern for thread safety
        if cache_key in _GLOBAL_SLAB_CACHE:
            cached_slab, cached_metadata = _GLOBAL_SLAB_CACHE[cache_key]
            return cached_slab, cached_metadata

        all_flattened_tensors = []  # Direct collection of all flattened tensors
        global_metadata = SlabMetadata()
        current_global_offset = 0

        for module_name, module_lora in lora_model.loras.items():
            if module_lora is None:
                continue
            # Process lora_a
            if hasattr(module_lora, "lora_a") and module_lora.lora_a is not None:
                if isinstance(module_lora.lora_a, list):
                    for expert_idx, expert_tensor in enumerate(module_lora.lora_a):
                        if expert_tensor is not None:
                            all_flattened_tensors.append(expert_tensor.flatten())
                            tensor_info = TensorInfo(
                                f"{module_name}.lora_a.expert_{expert_idx}",
                                "a",
                                expert_tensor.shape,
                                expert_tensor.numel(),
                                current_global_offset,
                            )
                            global_metadata.tensor_infos.append(tensor_info)
                            current_global_offset += expert_tensor.numel()
                else:
                    # Single tensor
                    all_flattened_tensors.append(module_lora.lora_a.flatten())
                    tensor_info = TensorInfo(
                        f"{module_name}.lora_a",
                        "a",
                        module_lora.lora_a.shape,
                        module_lora.lora_a.numel(),
                        current_global_offset,
                    )
                    global_metadata.tensor_infos.append(tensor_info)
                    current_global_offset += module_lora.lora_a.numel()

            # Process lora_b (scaling already applied during packing for packed modules)
            if hasattr(module_lora, "lora_b") and module_lora.lora_b is not None:
                if isinstance(module_lora.lora_b, list):
                    module_lora_b_count = 0
                    for expert_idx, expert_tensor in enumerate(module_lora.lora_b):
                        if expert_tensor is not None:
                            all_flattened_tensors.append(expert_tensor.flatten())
                            tensor_info = TensorInfo(
                                f"{module_name}.lora_b.expert_{expert_idx}",
                                "b",
                                expert_tensor.shape,
                                expert_tensor.numel(),
                                current_global_offset,
                            )
                            global_metadata.tensor_infos.append(tensor_info)
                            module_lora_b_count += expert_tensor.numel()
                            current_global_offset += expert_tensor.numel()
                else:
                    # Single tensor
                    all_flattened_tensors.append(module_lora.lora_b.flatten())
                    tensor_info = TensorInfo(
                        f"{module_name}.lora_b",
                        "b",
                        module_lora.lora_b.shape,
                        module_lora.lora_b.numel(),
                        current_global_offset,
                    )
                    global_metadata.tensor_infos.append(tensor_info)
                    current_global_offset += module_lora.lora_b.numel()
        extraction_map = {}
        lookup = {info.module_name: info for info in global_metadata.tensor_infos}

        for module_name, module_lora in lora_model.loras.items():
            if module_lora is None:
                continue
            # Check if module has list structure (packed MoE/QKV) or single tensor
            has_list = (
                isinstance(module_lora.lora_a, list)
                if hasattr(module_lora, "lora_a") and module_lora.lora_a is not None
                else False
            )
            if has_list:
                # Packed module with list - collect all expert tensor infos
                expert_tensors_a = []
                expert_tensors_b = []

                for expert_idx in range(len(module_lora.lora_a)):
                    a_key = f"{module_name}.lora_a.expert_{expert_idx}"
                    b_key = f"{module_name}.lora_b.expert_{expert_idx}"
                    if a_key in lookup:
                        a_info = lookup[a_key]
                        expert_tensors_a.append(
                            (a_info.offset, a_info.size, a_info.shape)
                        )
                    if b_key in lookup:
                        b_info = lookup[b_key]
                        expert_tensors_b.append(
                            (b_info.offset, b_info.size, b_info.shape)
                        )

                # Determine type based on module name
                if module_name.endswith(".mlp.experts"):
                    extraction_map[module_name] = (
                        "moe",
                        expert_tensors_a,
                        expert_tensors_b,
                    )
                elif module_name.endswith(".qkv_proj"):
                    extraction_map[module_name] = (
                        "qkv",
                        expert_tensors_a,
                        expert_tensors_b,
                    )
            else:
                # Single tensor module
                lora_a_key = f"{module_name}.lora_a"
                lora_b_key = f"{module_name}.lora_b"

                if lora_a_key in lookup and lora_b_key in lookup:
                    a_info = lookup[lora_a_key]
                    b_info = lookup[lora_b_key]
                    extraction_map[module_name] = (  # type: ignore[assignment]
                        "linear",
                        a_info.offset,
                        a_info.size,
                        a_info.shape,
                        b_info.offset,
                        b_info.size,
                        b_info.shape,
                    )

        # Store extraction_map in metadata for zero-overhead extraction
        global_metadata.extraction_map = extraction_map
        slab_dtype = torch.float16  # Default fallback
        if lora_config and hasattr(lora_config, "lora_dtype"):
            slab_dtype = lora_config.lora_dtype
        elif all_flattened_tensors:
            # Use dtype from first tensor if available
            slab_dtype = all_flattened_tensors[0].dtype

        if all_flattened_tensors:
            # Calculate tensor sizes for view allocation
            tensor_sizes = [t.numel() for t in all_flattened_tensors]
            total_elements = sum(tensor_sizes)
            global_metadata.total_size = total_elements

            # Allocate slab + individual views DIRECTLY in pinned pool - ZERO copy!
            full_slab, tensor_views = pool.allocate_slab_views_directly(
                tensor_sizes, slab_dtype
            )

            for i, (source_tensor, view_tensor) in enumerate(
                zip(all_flattened_tensors, tensor_views)
            ):
                view_tensor.copy_(source_tensor)
        else:
            # Empty slab case
            full_slab, _ = pool.allocate_slab_views_directly([], slab_dtype)
            global_metadata.total_size = 0

        slab_tensor = full_slab
        metadata = global_metadata

        # Cache the built slab in memory
        with _CACHE_LOCK:
            _GLOBAL_SLAB_CACHE[cache_key] = (slab_tensor, metadata)

        return slab_tensor, metadata


def extract_tensors_from_gpu_slab(gpu_slab, metadata, module_name, scaling=1.0):
    """Extract lora_a and lora_b tensors from GPU slab for a module.
    
    Args:
        gpu_slab: The GPU slab containing all weights
        metadata: Slab metadata with extraction map
        module_name: Name of the module to extract
        scaling: Scaling factor to apply to lora_b (default 1.0 for no scaling)
    """
    extraction_info = metadata.extraction_map.get(module_name)
    if not extraction_info:
        return None, None

    extraction_type = extraction_info[0]

    if extraction_type == "linear":
        # tensor: ('linear', a_offset, a_size, a_shape, b_offset, b_size, b_shape)
        _, a_offset, a_size, a_shape, b_offset, b_size, b_shape = extraction_info
        lora_a = gpu_slab[a_offset : a_offset + a_size].view(a_shape)
        lora_b = gpu_slab[b_offset : b_offset + b_size].view(b_shape)
        
        # Apply scaling to lora_b if needed (slab path skips optimize())
        if scaling != 1.0:
            lora_b = lora_b * scaling
        
        return lora_a, lora_b

    elif extraction_type in ("moe", "qkv"):
        # List of tensors: ('moe'/'qkv', expert_tensors_a, expert_tensors_b)
        _, expert_tensors_a, expert_tensors_b = extraction_info

        lora_a_list = []
        for i, (offset, size, shape) in enumerate(expert_tensors_a):
            tensor = gpu_slab[offset : offset + size].view(shape)
            lora_a_list.append(tensor)

        lora_b_list = []
        for i, (offset, size, shape) in enumerate(expert_tensors_b):
            tensor = gpu_slab[offset : offset + size].view(shape)
            # Apply scaling to each expert's lora_b if needed
            if scaling != 1.0:
                tensor = tensor * scaling
            lora_b_list.append(tensor)
        return lora_a_list, lora_b_list

    return None, None


def process_slab_activation_loop(
    modules_dict,
    lora_model,
    get_lora_layer_weights_fn,
    lora_config,
    gpu_slab,
    metadata,
    index,
):
    """Extract weights from GPU slab and activate."""

    # Loop through model modules
    for module_name, module in modules_dict.items():
        # Get scaling from lora_model (slab path doesn't call optimize())
        module_lora = lora_model.loras.get(module_name)
        scaling = 1.0
        if module_lora and hasattr(module_lora, 'scaling'):
            # Handle both single scaling and list of scalings (for packed modules)
            if isinstance(module_lora.scaling, (list, tuple)):
                scaling = module_lora.scaling[0] if module_lora.scaling[0] else 1.0
            else:
                scaling = module_lora.scaling if module_lora.scaling else 1.0
        
        lora_a_gpu, lora_b_gpu = extract_tensors_from_gpu_slab(
            gpu_slab, metadata, module_name, scaling=scaling
        )

        if lora_a_gpu is None or lora_b_gpu is None:
            # No weights for this module
            module.reset_lora(index)
            continue

        # Special case: MoE3D needs 2-item list format
        if isinstance(module, FusedMoE3DWithLoRA):
            gate_up_scaling = 1.0
            gate_up_lora = lora_model.loras.get(module_name + ".base_layer")
            if gate_up_lora and hasattr(gate_up_lora, 'scaling'):
                gate_up_scaling = gate_up_lora.scaling if gate_up_lora.scaling else 1.0
            
            gate_up_a, gate_up_b = extract_tensors_from_gpu_slab(
                gpu_slab, metadata, module_name + ".base_layer", scaling=gate_up_scaling
            )
            down_a, down_b = lora_a_gpu, lora_b_gpu

            if gate_up_a is not None and down_a is not None:
                lora_a_gpu = [gate_up_a, down_a]
                lora_b_gpu = [gate_up_b, down_b]
        module.set_lora(index, lora_a_gpu, lora_b_gpu)
    return True


def check_slab_cache(lora_dir, peft_helper, target_lora_config, target_modules_dict):
    """Check if LoRAModel is already cached for this LoRA directory."""
    if not lora_dir:
        return False, None

    # Generate simple key based on lora_dir only
    cache_key = hashlib.md5(lora_dir.encode()).hexdigest()

    # Check LoRAModel cache
    with _LORA_MODEL_CACHE_LOCK:
        if cache_key in _GLOBAL_LORA_MODEL_CACHE:
            logger.info("[SLAB_CACHE_HIT] Found cached LoRAModel for %s", lora_dir)
            return True, _GLOBAL_LORA_MODEL_CACHE[cache_key]

    logger.info("[SLAB_CACHE_MISS] No cached LoRAModel for %s", lora_dir)
    return False, None


def cache_lora_model(lora_dir, lora_model):
    """Store LoRAModel in cache for reuse."""
    if not lora_dir:
        return

    cache_key = hashlib.md5(lora_dir.encode()).hexdigest()

    with _LORA_MODEL_CACHE_LOCK:
        _GLOBAL_LORA_MODEL_CACHE[cache_key] = lora_model
        logger.info("[SLAB_CACHE] Stored LoRAModel for %s", lora_dir)


def get_cached_lora_model(cache_key):
    """Get cached LoRA model."""
    with _LORA_MODEL_CACHE_LOCK:
        return _GLOBAL_LORA_MODEL_CACHE.get(cache_key)


def _generate_slab_cache_key(lora_model, device, tp_rank=None, fully_sharded=False):
    """Generate cache key for LoRA slab - includes tp_rank when fully_sharded=True."""
    lora_dir = getattr(lora_model, "_lora_dir", None)

    if not lora_dir:
        lora_dir = f"unknown_path_{lora_model.rank}_{len(lora_model.loras)}"

    # Base key
    key_str = f"{lora_dir}|{lora_model.rank}|{len(lora_model.loras)}|{str(device)}"

    # Include tp_rank when fully_sharded=True (each GPU has different slab)
    if fully_sharded and tp_rank is not None:
        key_str += f"|tp_rank_{tp_rank}"

    cache_key = hashlib.md5(key_str.encode()).hexdigest()

    return cache_key


class TensorInfo:
    """Metadata for a tensor in the slab."""

    def __init__(
        self,
        module_name: str,
        tensor_type: str,
        shape: tuple,
        size: int,
        offset: int = 0,
    ):
        self.module_name = module_name
        self.tensor_type = tensor_type  # 'lora_a', 'lora_b'
        self.shape = shape
        self.size = size
        self.offset = offset


class SlabMetadata:
    """Metadata for the entire slab with pre-computed extraction data."""

    def __init__(self):
        self.tensor_infos: list[TensorInfo] = []
        self.total_size = 0
        # PERFORMANCE: Pre-computed extraction data to eliminate all scatter overhead
        self.extraction_map: dict[
            str, tuple
        ] = {}  # module_name -> (lora_a_slice, lora_b_slice)


def create_slab_optimized_lora_model(
    lora_model_id: int,
    tensors: dict[str, torch.Tensor],
    peft_helper,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    embeddings: dict[str, torch.Tensor] | None = None,
    target_embedding_padding: int | None = None,
    embedding_modules: dict[str, str] | None = None,
    embedding_padding_modules: list[str] | None = None,
    weights_mapper=None,
    lora_dir: str | None = None,
    lora_config=None,
    target_modules_dict=None,
    target_lora_config=None,
    slab_path: str | None = None,
    packed_modules: dict | None = None,
    packed_modules_mapping: dict | None = None,
):
    """Create a LoRAModel with target-aware slab."""
    if get_ultra_fast_pool() is None:
        pool = UltraFastPinnedPool()
        set_global_pool(pool)
    # Create LoRA weights as normal
    loras: dict[str, LoRALayerWeights] = {}

    for tensor_name, tensor in tensors.items():
        module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name, weights_mapper)

        if module_name not in loras:
            loras[module_name] = LoRALayerWeights.from_config(module_name, peft_helper)
        if is_lora_a:
            loras[module_name].lora_a = tensor.to(
                dtype=dtype
            )  # Keep on CPU for slab building
        else:
            loras[module_name].lora_b = tensor.to(
                dtype=dtype
            )  # Keep on CPU for slab building

            assert embedding_padding_modules is not None
            if (
                any(name in module_name for name in embedding_padding_modules)
                and target_embedding_padding is not None
            ):
                lora_b = loras[module_name].lora_b
                assert target_embedding_padding >= lora_b.shape[0]
                addition = target_embedding_padding - lora_b.shape[0]
                loras[module_name].lora_b = torch.nn.functional.pad(
                    lora_b, (0, 0, 0, addition)
                )

    # Create the LoRA model instance
    from vllm.lora.lora_model import LoRAModel

    lora_model_instance = LoRAModel(lora_model_id, peft_helper.r, loras)

    # Store the LoRA directory path for cache key generation
    if lora_dir:
        lora_model_instance._lora_dir = lora_dir  # type: ignore[attr-defined]

    if packed_modules and len(packed_modules) > 0:
        # Helper function to get lora weights (simplified version without model context)
        def get_lora_weights(lora_model, module_name):
            return lora_model.loras.get(module_name, None)

        # Pack modules similar to _create_merged_loras_inplace
        for module_name, new_module_names in packed_modules.items():
            replacement_loras: list[LoRALayerWeights | None] = []
            replaced_module: set[str] = set()
            has_replacement = False

            # Collect individual projections
            for r in new_module_names:
                lora = get_lora_weights(lora_model_instance, r)
                replacement_loras.append(lora)
                if lora:
                    has_replacement = True
                    replaced_module.add(r)

            if not has_replacement:
                continue

            # Ensure None values are explicit
            for i in range(len(replacement_loras)):
                if not replacement_loras[i]:
                    replacement_loras[i] = None

            # Pack based on module type
            if module_name.endswith(".experts"):
                packed_lora = PackedLoRALayerWeights.pack_moe(
                    replacement_loras,
                    module_name,
                )
            else:
                packed_lora = PackedLoRALayerWeights.pack(
                    replacement_loras
                )
            lora_model_instance.loras[module_name] = packed_lora
            # Remove individual projections
            for module in replaced_module:
                lora_model_instance.loras.pop(module, None)
            
            # for lora in lora_model_instance.loras.values():
            #     lora.optimize()


    else:
        logger.warning(
            "[SLAB_PRE_PACK] No packed_modules provided - "
            "slab will build with unpacked structure"
        )

    # TP SHARDING: Shard lora_b weights on CPU if fully_sharded_loras=True
    fully_sharded = (
        target_lora_config.fully_sharded_loras if target_lora_config else False
    )
    if fully_sharded and target_modules_dict:
        logger.info(
            "[SLAB_TP_SHARD] fully_sharded_loras=True, sharding lora_b weights on CPU"
        )

        for module_name, module_lora in lora_model_instance.loras.items():
            target_module = target_modules_dict.get(module_name)
            if not target_module:
                continue

            tp_rank = getattr(target_module, "tp_rank", 0)
            tp_size = getattr(target_module, "tp_size", 1)

            if (
                tp_size > 1
                and hasattr(module_lora, "lora_b")
                and module_lora.lora_b is not None
            ):
                if isinstance(module_lora.lora_b, list):
                    # MoE: shard each expert's lora_b
                    sharded_experts = []
                    for expert_idx, expert_b in enumerate(module_lora.lora_b):
                        if expert_b is not None:
                            shards = expert_b.chunk(tp_size, dim=0)
                            sharded_experts.append(shards[tp_rank])
                        else:
                            sharded_experts.append(None)
                    module_lora.lora_b = sharded_experts
                else:
                    # Single tensor: shard once
                    shards = module_lora.lora_b.chunk(tp_size, dim=0)
                    module_lora.lora_b = shards[tp_rank]

    # MOE WEIGHT STACKING: Process MOE modules to reshape from 2D to 3D
    # This must happen AFTER TP sharding so reshaping uses sharded weights
    if target_modules_dict:
        for module_name, module in target_modules_dict.items():
            if isinstance(module, FusedMoE3DWithLoRA):
                module_lora = lora_model_instance.loras.get(module_name)
                
                # Only process if lora_a is still a tensor (not already processed)
                if module_lora and torch.is_tensor(module_lora.lora_a):
                    gate_up_proj_lora = lora_model_instance.loras.get(module_name + ".base_layer")
                    down_proj_lora = module_lora
                    
                    if gate_up_proj_lora and down_proj_lora:
                        num_experts = module.w13_lora_a_stacked[0].shape[1]

                        # Reshape lora_a to (num_experts, rank, input_size)
                        gate_up_proj_lora.lora_a = gate_up_proj_lora.lora_a.reshape(
                            num_experts, -1, gate_up_proj_lora.lora_a.shape[-1]
                        )
                        down_proj_lora.lora_a = down_proj_lora.lora_a.reshape(
                            num_experts, -1, down_proj_lora.lora_a.shape[-1]
                        )
                        
                        # Reshape lora_b to (output_size, num_experts, rank)
                        gate_up_proj_lora.lora_b = gate_up_proj_lora.lora_b.reshape(
                            gate_up_proj_lora.lora_b.shape[0], -1, num_experts
                        )
                        down_proj_lora.lora_b = down_proj_lora.lora_b.reshape(
                            down_proj_lora.lora_b.shape[0], -1, num_experts
                        )
                        
                        # Permute lora_b to (num_experts, output_size, rank)
                        gate_up_proj_lora.lora_b = gate_up_proj_lora.lora_b.permute(
                            2, 0, 1
                        ).contiguous()
                        down_proj_lora.lora_b = down_proj_lora.lora_b.permute(
                            2, 0, 1
                        ).contiguous()
                        
                        # Convert to list format for MOE
                        module_lora.lora_a = [
                            gate_up_proj_lora.lora_a,
                            down_proj_lora.lora_a,
                        ]
                        module_lora.lora_b = [
                            gate_up_proj_lora.lora_b,
                            down_proj_lora.lora_b,
                        ]

    slab, metadata = build_target_matched_slab(
        lora_model_instance, target_modules_dict, 1, target_lora_config, slab_path
    )

    if not torch.cuda.is_available():
        # Return tuple for consistency even without GPU
        return lora_model_instance, None, None

    lora_model_instance._cached_cpu_slab = slab  # type: ignore[attr-defined]
    lora_model_instance._cached_metadata = metadata  # type: ignore[attr-defined]
    lora_model_instance._loras_dict = loras  # type: ignore[attr-defined]

    # Return CPU slab reference for now - GPU slab created during activation
    return lora_model_instance, None, metadata
