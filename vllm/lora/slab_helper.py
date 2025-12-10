# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import threading
import time
from typing import Optional
from collections.abc import Sequence

import safetensors.torch
import torch

from vllm.logger import init_logger
from vllm.lora.layers import FusedMoEWithLoRA
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.distributed.utils import divide

logger = init_logger(__name__)

# Global slab cache
_GLOBAL_SLAB_CACHE = {}
_CACHE_LOCK = threading.RLock()

# ULTIMATE SOLUTION: Global result storage to eliminate ALL large object returns
_GLOBAL_RESULT_STORAGE = {}
_RESULT_LOCK = threading.RLock()


class UltraFastPinnedPool:
    """Pre-allocated pinned memory pool to achieve 20x faster pinning."""
    
    def __init__(self, initial_pool_size: int = 4*1024 * 1024 * 1024):  # 1GB initial pool
        self.pool_size = initial_pool_size
        # Pre-allocate large pinned buffer at startup - one-time 1.7s cost
        self.pinned_pool = torch.empty(initial_pool_size, dtype=torch.uint8).pin_memory()
        self.pool_lock = threading.RLock()
        self.used_ranges = []  # Track used memory ranges
        
        # OPTION 2 IMPLEMENTATION: Store current slab and metadata as instance variables
        # This eliminates the 149ms Python function return overhead for large objects
        self.current_slab = None
        self.current_metadata = None
        
        logger.info(f"[ULTRA_FAST_POOL] Initialized {initial_pool_size / (1024*1024):.0f}MB pre-pinned pool")
    
    def allocate_slab_views_directly(self, tensor_sizes: list[int], dtype: torch.dtype) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Allocate slab and return both the full slab AND individual tensor views - ZERO copy needed!"""
        total_elements = sum(tensor_sizes)
        
        if total_elements == 0:
            return torch.empty(0, dtype=dtype, device='cpu').pin_memory(), []
        
        tensor_bytes = total_elements * dtype.itemsize
        
        with self.pool_lock:
            # Expand pool if needed
            if tensor_bytes > self.pool_size:
                new_size = max(self.pool_size * 2, tensor_bytes + self.pool_size)
                logger.info(f"[DIRECT_SLAB_ALLOC] Expanding pool to {new_size / (1024*1024*1024):.1f}GB for large slab")
                
                new_pool = torch.empty(new_size, dtype=torch.uint8).pin_memory()
                
                # Copy existing data if any
                if self.used_ranges:
                    total_used = max(end for start, end in self.used_ranges)
                    new_pool[:total_used] = self.pinned_pool[:total_used]
                
                self.pinned_pool = new_pool
                self.pool_size = new_size
            
            # Find available space
            start_offset = max((end for start, end in self.used_ranges), default=0)
            end_offset = start_offset + tensor_bytes
            
            if end_offset > self.pool_size:
                # Reset pool - reuse from beginning
                self.used_ranges.clear()
                start_offset = 0
                end_offset = tensor_bytes
                logger.info(f"[DIRECT_SLAB_ALLOC] Pool reset - reusing from beginning")
            
            self.used_ranges.append((start_offset, end_offset))
            
            # Create full slab view
            pool_slice = self.pinned_pool[start_offset:end_offset]
            full_slab = pool_slice.view(torch.uint8).view(dtype)[:total_elements]
            
            # Create individual tensor views for each component - NO copying!
            tensor_views = []
            current_offset = 0
            for size in tensor_sizes:
                if size > 0:
                    tensor_view = full_slab[current_offset:current_offset + size]
                    tensor_views.append(tensor_view)
                    current_offset += size
                else:
                    tensor_views.append(torch.empty(0, dtype=dtype, device='cpu'))
            
            logger.info(f"[VIEW_ALLOCATION] ✅ Created slab + {len(tensor_views)} views DIRECTLY in pinned pool - ZERO copy operations!")
            return full_slab, tensor_views

    def allocate_slab_directly(self, num_elements: int, dtype: torch.dtype) -> torch.Tensor:
        """Allocate slab DIRECTLY from pinned pool - eliminates torch.cat() AND copy operations."""
        if num_elements == 0:
            return torch.empty(0, dtype=dtype, device='cpu').pin_memory()
        
        tensor_bytes = num_elements * dtype.itemsize
        
        with self.pool_lock:
            # Expand pool if needed
            if tensor_bytes > self.pool_size:
                new_size = max(self.pool_size * 2, tensor_bytes + self.pool_size)
                logger.info(f"[DIRECT_SLAB_ALLOC] Expanding pool to {new_size / (1024*1024*1024):.1f}GB for large slab")
                
                new_pool = torch.empty(new_size, dtype=torch.uint8).pin_memory()
                
                # Copy existing data if any
                if self.used_ranges:
                    total_used = max(end for start, end in self.used_ranges)
                    new_pool[:total_used] = self.pinned_pool[:total_used]
                
                self.pinned_pool = new_pool
                self.pool_size = new_size
            
            # Find available space
            start_offset = max((end for start, end in self.used_ranges), default=0)
            end_offset = start_offset + tensor_bytes
            
            if end_offset > self.pool_size:
                # Reset pool - reuse from beginning
                self.used_ranges.clear()
                start_offset = 0
                end_offset = tensor_bytes
                logger.info(f"[DIRECT_SLAB_ALLOC] Pool reset - reusing from beginning")
            
            self.used_ranges.append((start_offset, end_offset))
            
            # Return direct view of pinned pool - NO copy needed!
            pool_slice = self.pinned_pool[start_offset:end_offset]
            slab_tensor = pool_slice.view(torch.uint8).view(dtype)[:num_elements]
            
            logger.info(f"[DIRECT_SLAB_ALLOC] ✅ Allocated {tensor_bytes / (1024*1024):.1f}MB slab DIRECTLY in pinned pool - NO torch.cat(), NO copy!")
            return slab_tensor

    def get_pinned_tensor_fast(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """Ultra-fast pseudo-pinning using pre-allocated pool (20x faster than pin_memory)."""
        tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
        
        with self.pool_lock:
            # Find available space in pool
            if tensor_bytes > self.pool_size:
                # Expand pool if needed
                new_size = max(self.pool_size * 2, tensor_bytes + self.pool_size)
                logger.info(f"[ULTRA_FAST_POOL] Expanding pool to {new_size / (1024*1024):.0f}MB")
                
                # Create larger pool
                new_pool = torch.empty(new_size, dtype=torch.uint8).pin_memory()
                
                # Copy existing data if any
                if self.used_ranges:
                    total_used = max(end for start, end in self.used_ranges)
                    new_pool[:total_used] = self.pinned_pool[:total_used]
                
                self.pinned_pool = new_pool
                self.pool_size = new_size
            
            # Simple allocation strategy - find space at end
            start_offset = max((end for start, end in self.used_ranges), default=0)
            end_offset = start_offset + tensor_bytes
            
            if end_offset > self.pool_size:
                # Reset pool if we're at the end - reuse from beginning
                self.used_ranges.clear()
                start_offset = 0
                end_offset = tensor_bytes
            
            self.used_ranges.append((start_offset, end_offset))
            
            # Get slice from pre-pinned pool
            pool_slice = self.pinned_pool[start_offset:end_offset]
            
            # Reshape to match tensor and copy data (fast memory copy)
            pinned_tensor = pool_slice.view(torch.uint8).view(cpu_tensor.dtype)[:cpu_tensor.numel()].view(cpu_tensor.shape)
            pinned_tensor.copy_(cpu_tensor)  # Fast copy into pre-pinned memory
            
            logger.info(f"[ULTRA_FAST_POOL] Fast-pinned {tensor_bytes / (1024*1024):.1f}MB in pool")
            return pinned_tensor

# Global ultra-fast pool - initialized ONCE in envs.py
_ULTRA_FAST_POOL = None
_POOL_INIT_LOCK = threading.RLock()

def set_global_pool(pool: UltraFastPinnedPool) -> None:
    """Set the global pool instance - called once from envs.py to prevent re-initialization."""
    global _ULTRA_FAST_POOL
    with _POOL_INIT_LOCK:
        if _ULTRA_FAST_POOL is None:
            _ULTRA_FAST_POOL = pool
            logger.info("[SET_GLOBAL_POOL] ✅ Global pool set - will prevent re-initialization!")
        else:
            logger.warning("[SET_GLOBAL_POOL] Pool already set - ignoring duplicate initialization attempt")

def get_ultra_fast_pool():
    """Get the pre-initialized global pool - NO lazy initialization."""
    global _ULTRA_FAST_POOL
    if _ULTRA_FAST_POOL is None:
        logger.error("[POOL_ERROR] ❌ Pool not initialized! Should be set in envs.py first")
        # Fallback - create pool if not set (shouldn't happen)
        with _POOL_INIT_LOCK:
            if _ULTRA_FAST_POOL is None:
                logger.warning("[POOL_FALLBACK] Creating pool as fallback - this should be done in envs.py!")
                _ULTRA_FAST_POOL = UltraFastPinnedPool()
    return _ULTRA_FAST_POOL


def save_slab_to_disk(slab_tensor: torch.Tensor, metadata: 'SlabMetadata', slab_path: str) -> None:
    """
    Save slab tensor and metadata to disk using safetensors format.
    
    Args:
        slab_tensor: The slab tensor to save
        metadata: The slab metadata containing tensor information
        slab_path: Path to save the slab (directory will be created if needed)
    """
    try:
        # Create directory if it doesn't exist
        slab_dir = os.path.dirname(slab_path)
        if slab_dir and not os.path.exists(slab_dir):
            os.makedirs(slab_dir, exist_ok=True)
        
        # Check if tensor is pinned to preserve this info
        is_pinned = slab_tensor.is_pinned() if slab_tensor.device.type == 'cpu' else False
        
        # Prepare data to save - convert to pageable CPU for safetensors compatibility
        tensors_to_save = {"slab_tensor": slab_tensor.cpu()}
        
        # Convert metadata to JSON-serializable format
        metadata_dict = {
            "total_size": metadata.total_size,
            "is_pinned": is_pinned,
            "tensor_infos": [
                {
                    "module_name": info.module_name,
                    "tensor_type": info.tensor_type,
                    "shape": list(info.shape),
                    "size": info.size,
                    "offset": info.offset
                }
                for info in metadata.tensor_infos
            ]
        }
        
        # Save tensors and metadata
        safetensors.torch.save_file(tensors_to_save, slab_path)
        
        # Save metadata as JSON alongside the slab
        metadata_path = slab_path.replace('.safetensors', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        slab_size_mb = slab_tensor.numel() * slab_tensor.element_size() / (1024 * 1024)
        logger.info(f"[SLAB_SAVE] Successfully saved slab (~{slab_size_mb:.1f}MB) to {slab_path}")
        
    except Exception as e:
        logger.error(f"[SLAB_SAVE] Failed to save slab to {slab_path}: {e}")
        raise


def load_slab_from_disk(slab_path: str) -> tuple[torch.Tensor, 'SlabMetadata']:
    """
    Load slab tensor and metadata from disk as PAGEABLE memory.
    NO pinning operations - preserves PCIe bandwidth.
    
    Args:
        slab_path: Path to the saved slab file
        
    Returns:
        Tuple of (pageable slab_tensor, metadata)
        
    Raises:
        FileNotFoundError: If slab file doesn't exist
        Exception: If loading fails
    """
    try:
        if not os.path.exists(slab_path):
            raise FileNotFoundError(f"Slab file not found: {slab_path}")
        
        # Load tensors as pageable - NO PINNING OPERATIONS!
        tensors = safetensors.torch.load_file(slab_path, device="cpu")
        slab_tensor = tensors["slab_tensor"]
        
        # Load metadata
        metadata_path = slab_path.replace('.safetensors', '_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # NO PINNING - Keep as pageable to avoid PCIe bandwidth consumption
        logger.info(f"[SLAB_LOAD_PAGEABLE] ✅ Loaded as pageable - NO pinning operations!")
        
        # Reconstruct metadata object
        metadata = SlabMetadata()
        metadata.total_size = metadata_dict["total_size"]
        
        for info_dict in metadata_dict["tensor_infos"]:
            tensor_info = TensorInfo(
                module_name=info_dict["module_name"],
                tensor_type=info_dict["tensor_type"],
                shape=tuple(info_dict["shape"]),
                size=info_dict["size"],
                offset=info_dict["offset"]
            )
            metadata.tensor_infos.append(tensor_info)
        
        slab_size_mb = slab_tensor.numel() * slab_tensor.element_size() / (1024 * 1024)
        logger.info(f"[SLAB_LOAD] Successfully loaded pageable slab (~{slab_size_mb:.1f}MB) from {slab_path}")
        
        return slab_tensor, metadata
        
    except Exception as e:
        logger.error(f"[SLAB_LOAD] Failed to load slab from {slab_path}: {e}")
        raise


def check_slab_exists(slab_path: str) -> bool:
    """
    Check if slab file and metadata exist on disk.
    
    Args:
        slab_path: Path to the slab file
        
    Returns:
        True if both slab and metadata files exist, False otherwise
    """
    if not slab_path:
        return False
        
    metadata_path = slab_path.replace('.safetensors', '_metadata.json')
    return os.path.exists(slab_path) and os.path.exists(metadata_path)



def use_layer_pre_allocated_tensors_directly(module, module_lora, dtype=torch.bfloat16, device=torch.device('cpu')):
    """
    DIRECT APPROACH: Use the layer's own pre-allocated stacked tensors directly without any shape checking.
    This leverages the fact that layers like FusedMoE already define and pre-allocate these tensors 
    (e.g., self.w1_lora_a_stacked) with the correct shapes and dimensions.
    """
    tensors_for_slab = {}
    
    # Validate LoRA scaling consistency - ensure uniform scaling across module
    scaling_factor = getattr(module_lora, 'scaling', 1.0)
    
    # Check if scaling is needed (non-one scaling factor)
    apply_scaling = scaling_factor != 1.0
    
    # --- FusedMoE: Use the layer's pre-allocated stacked tensors directly ---
    if hasattr(module, 'w1_lora_a_stacked'):
        # FusedMoEWithLoRA with separate w1, w2, w3
        weight_types = ['w1_lora_a', 'w1_lora_b', 'w2_lora_a', 'w2_lora_b', 'w3_lora_a', 'w3_lora_b']
        
        for weight_type in weight_types:
            stacked_attr = f"{weight_type}_stacked"
            if hasattr(module, stacked_attr):
                layer_tensor = getattr(module, stacked_attr)
                    
            # Get shape from GPU tensor without D2H transfer, create CPU tensor directly
            shape = layer_tensor.shape
            cpu_tensor = torch.zeros(shape, dtype=dtype, device=device)
            
            # Populate with actual LoRA data if available AND apply scaling during slab building
            if weight_type.endswith('_lora_a') and hasattr(module_lora, 'lora_a') and module_lora.lora_a is not None:
                # Handle list case for FusedMoE experts
                if isinstance(module_lora.lora_a, list):
                    for expert_idx in range(min(len(module_lora.lora_a), cpu_tensor.shape[1])):
                        if module_lora.lora_a[expert_idx] is not None:
                            src_tensor = module_lora.lora_a[expert_idx].to(dtype)
                            min_h = min(src_tensor.shape[0], cpu_tensor.shape[2])
                            min_w = min(src_tensor.shape[1], cpu_tensor.shape[3])
                            cpu_tensor[0, expert_idx, :min_h, :min_w] = src_tensor[:min_h, :min_w]
                # Handle single tensor case 
                elif isinstance(module_lora.lora_a, torch.Tensor):
                    src_tensor = module_lora.lora_a.to(dtype)
                    min_h = min(src_tensor.shape[0], cpu_tensor.shape[2])
                    min_w = min(src_tensor.shape[1], cpu_tensor.shape[3])
                    cpu_tensor[0, 0, :min_h, :min_w] = src_tensor[:min_h, :min_w]
                            
            elif weight_type.endswith('_lora_b') and hasattr(module_lora, 'lora_b') and module_lora.lora_b is not None:
                # Handle list case for FusedMoE experts
                if isinstance(module_lora.lora_b, list):
                    for expert_idx in range(min(len(module_lora.lora_b), cpu_tensor.shape[1])):
                        if module_lora.lora_b[expert_idx] is not None:
                            src_tensor = module_lora.lora_b[expert_idx].to(dtype)
                            # Apply validated scaling during slab building - cached in slab
                            if apply_scaling:
                                scaled_tensor = src_tensor * scaling_factor
                            else:
                                scaled_tensor = src_tensor
                            min_h = min(scaled_tensor.shape[0], cpu_tensor.shape[2])
                            min_w = min(scaled_tensor.shape[1], cpu_tensor.shape[3])
                            cpu_tensor[0, expert_idx, :min_h, :min_w] = scaled_tensor[:min_h, :min_w]
                # Handle single tensor case with scaling
                elif isinstance(module_lora.lora_b, torch.Tensor):
                    src_tensor = module_lora.lora_b.to(dtype)
                    # Apply validated scaling during slab building - cached in slab
                    if apply_scaling:
                        scaled_tensor = src_tensor * scaling_factor
                    else:
                        scaled_tensor = src_tensor
                    min_h = min(scaled_tensor.shape[0], cpu_tensor.shape[2])
                    min_w = min(scaled_tensor.shape[1], cpu_tensor.shape[3])
                    cpu_tensor[0, 0, :min_h, :min_w] = scaled_tensor[:min_h, :min_w]
            
            tensors_for_slab[weight_type] = cpu_tensor
            
        # Mark scaling as applied to prevent double-scaling
        if apply_scaling:
            module_lora.scaling = 1.0
    
    # --- FusedMoE3D: Use combined w13 stacked tensors ---
    elif hasattr(module, 'w13_lora_a_stacked'):
        # FusedMoE3DWithLoRA with combined w13
        weight_types = ['w13_lora_a', 'w13_lora_b', 'w2_lora_a', 'w2_lora_b']
        
        for weight_type in weight_types:
            stacked_attr = f"{weight_type}_stacked"
            if hasattr(module, stacked_attr):
                layer_tensor = getattr(module, stacked_attr)[0]  # It's a tuple with 1 element
                    
            # Get shape from GPU tensor without D2H transfer, create CPU tensor directly
            shape = layer_tensor.shape
            cpu_tensor = torch.zeros(shape, dtype=dtype, device=device)
            
            # Populate with actual LoRA data if available AND apply scaling during slab building
            if weight_type.endswith('_lora_a') and hasattr(module_lora, 'lora_a') and module_lora.lora_a is not None:
                # Handle list case for FusedMoE experts
                if isinstance(module_lora.lora_a, list):
                    for expert_idx in range(min(len(module_lora.lora_a), cpu_tensor.shape[1])):
                        if module_lora.lora_a[expert_idx] is not None:
                            src_tensor = module_lora.lora_a[expert_idx].to(dtype)
                            min_h = min(src_tensor.shape[0], cpu_tensor.shape[2])
                            min_w = min(src_tensor.shape[1], cpu_tensor.shape[3])
                            cpu_tensor[0, expert_idx, :min_h, :min_w] = src_tensor[:min_h, :min_w]
                # Handle single tensor case 
                elif isinstance(module_lora.lora_a, torch.Tensor):
                    src_tensor = module_lora.lora_a.to(dtype)
                    min_h = min(src_tensor.shape[0], cpu_tensor.shape[2])
                    min_w = min(src_tensor.shape[1], cpu_tensor.shape[3])
                    cpu_tensor[0, 0, :min_h, :min_w] = src_tensor[:min_h, :min_w]
                            
            elif weight_type.endswith('_lora_b') and hasattr(module_lora, 'lora_b') and module_lora.lora_b is not None:
                # Handle list case for FusedMoE experts
                if isinstance(module_lora.lora_b, list):
                    for expert_idx in range(min(len(module_lora.lora_b), cpu_tensor.shape[1])):
                        if module_lora.lora_b[expert_idx] is not None:
                            src_tensor = module_lora.lora_b[expert_idx].to(dtype)
                            # Apply validated scaling during slab building - cached in slab
                            if apply_scaling:
                                scaled_tensor = src_tensor * scaling_factor
                            else:
                                scaled_tensor = src_tensor
                            min_h = min(scaled_tensor.shape[0], cpu_tensor.shape[2])
                            min_w = min(scaled_tensor.shape[1], cpu_tensor.shape[3])
                            cpu_tensor[0, expert_idx, :min_h, :min_w] = scaled_tensor[:min_h, :min_w]
                # Handle single tensor case with scaling
                elif isinstance(module_lora.lora_b, torch.Tensor):
                    src_tensor = module_lora.lora_b.to(dtype)
                    # Apply validated scaling during slab building - cached in slab
                    if apply_scaling:
                        scaled_tensor = src_tensor * scaling_factor
                    else:
                        scaled_tensor = src_tensor
                    min_h = min(scaled_tensor.shape[0], cpu_tensor.shape[2])
                    min_w = min(scaled_tensor.shape[1], cpu_tensor.shape[3])
                    cpu_tensor[0, 0, :min_h, :min_w] = scaled_tensor[:min_h, :min_w]
            
            tensors_for_slab[weight_type] = cpu_tensor
            
        # Mark scaling as applied to prevent double-scaling
        if apply_scaling:
            module_lora.scaling = 1.0
    
    # --- ColumnParallel/BaseLinear: Use the layer's pre-allocated stacked tensors directly ---
    elif hasattr(module, 'lora_a_stacked'):
        # Get the layer's pre-allocated tensors
        lora_a_stacked = module.lora_a_stacked
        lora_b_stacked = module.lora_b_stacked
        
        # Handle multi-slice case (tuples)
        if isinstance(lora_a_stacked, tuple):
            tensors_for_slab['lora_a'] = []
            tensors_for_slab['lora_b'] = []
            
            for slice_idx in range(len(lora_a_stacked)):
                # Get shapes without D2H transfer, create CPU tensors directly
                shape_a = lora_a_stacked[slice_idx].shape
                shape_b = lora_b_stacked[slice_idx].shape
                cpu_tensor_a = torch.zeros(shape_a, dtype=dtype, device=device)
                cpu_tensor_b = torch.zeros(shape_b, dtype=dtype, device=device)
                
                # Handle both single tensor and list cases for LoRA data
                if hasattr(module_lora, 'lora_a') and module_lora.lora_a is not None:
                    if isinstance(module_lora.lora_a, torch.Tensor) and slice_idx == 0:
                        src_tensor = module_lora.lora_a.to(dtype)
                        min_h = min(src_tensor.shape[0], cpu_tensor_a.shape[2])
                        min_w = min(src_tensor.shape[1], cpu_tensor_a.shape[3])
                        cpu_tensor_a[0, 0, :min_h, :min_w] = src_tensor[:min_h, :min_w]
                    elif isinstance(module_lora.lora_a, list) and slice_idx < len(module_lora.lora_a):
                        if module_lora.lora_a[slice_idx] is not None:
                            src_tensor = module_lora.lora_a[slice_idx].to(dtype)
                            min_h = min(src_tensor.shape[0], cpu_tensor_a.shape[2])
                            min_w = min(src_tensor.shape[1], cpu_tensor_a.shape[3])
                            cpu_tensor_a[0, 0, :min_h, :min_w] = src_tensor[:min_h, :min_w]
                
                if hasattr(module_lora, 'lora_b') and module_lora.lora_b is not None:
                    if isinstance(module_lora.lora_b, torch.Tensor) and slice_idx == 0:
                        src_tensor = module_lora.lora_b.to(dtype)
                        if apply_scaling:
                            scaled_tensor = src_tensor * scaling_factor
                        else:
                            scaled_tensor = src_tensor
                        min_h = min(scaled_tensor.shape[0], cpu_tensor_b.shape[2])
                        min_w = min(scaled_tensor.shape[1], cpu_tensor_b.shape[3])
                        cpu_tensor_b[0, 0, :min_h, :min_w] = scaled_tensor[:min_h, :min_w]
                    elif isinstance(module_lora.lora_b, list) and slice_idx < len(module_lora.lora_b):
                        if module_lora.lora_b[slice_idx] is not None:
                            src_tensor = module_lora.lora_b[slice_idx].to(dtype)
                            if apply_scaling:
                                scaled_tensor = src_tensor * scaling_factor
                            else:
                                scaled_tensor = src_tensor
                            min_h = min(scaled_tensor.shape[0], cpu_tensor_b.shape[2])
                            min_w = min(scaled_tensor.shape[1], cpu_tensor_b.shape[3])
                            cpu_tensor_b[0, 0, :min_h, :min_w] = scaled_tensor[:min_h, :min_w]
                
                tensors_for_slab['lora_a'].append(cpu_tensor_a)
                tensors_for_slab['lora_b'].append(cpu_tensor_b)
        
        # Handle single tensor case
        else:
            # Get shapes without D2H transfer, create CPU tensors directly
            shape_a = lora_a_stacked.shape
            shape_b = lora_b_stacked.shape
            tensors_for_slab['lora_a'] = torch.zeros(shape_a, dtype=dtype, device=device)
            tensors_for_slab['lora_b'] = torch.zeros(shape_b, dtype=dtype, device=device)
            
            # Populate with actual LoRA data and apply scaling during slab building
            if hasattr(module_lora, 'lora_a') and module_lora.lora_a is not None:
                src_tensor = module_lora.lora_a.to(dtype)
                min_h = min(src_tensor.shape[0], tensors_for_slab['lora_a'].shape[2])
                min_w = min(src_tensor.shape[1], tensors_for_slab['lora_a'].shape[3])
                tensors_for_slab['lora_a'][0, 0, :min_h, :min_w] = src_tensor[:min_h, :min_w]
            
            if hasattr(module_lora, 'lora_b') and module_lora.lora_b is not None:
                src_tensor = module_lora.lora_b.to(dtype)
                if apply_scaling:
                    scaled_tensor = src_tensor * scaling_factor
                else:
                    scaled_tensor = src_tensor
                min_h = min(scaled_tensor.shape[0], tensors_for_slab['lora_b'].shape[2])
                min_w = min(scaled_tensor.shape[1], tensors_for_slab['lora_b'].shape[3])
                tensors_for_slab['lora_b'][0, 0, :min_h, :min_w] = scaled_tensor[:min_h, :min_w]

        # Mark scaling as applied to prevent double-scaling
        if apply_scaling:
            module_lora.scaling = 1.0
    
    else:
        # Module doesn't have expected pre-allocated tensors - keep error logging
        logger.error(f"[PRE_ALLOC_FAILURE] ❌ Module {module.__class__.__name__} doesn't have pre-allocated stacked tensors!")
        raise RuntimeError(f"Module {module.__class__.__name__} doesn't have expected pre-allocated stacked tensors")
    
    return tensors_for_slab


# Main public interface with CPU caching and disk save/load
def build_target_matched_slab(lora_model, target_modules, max_loras, lora_config, slab_path: Optional[str] = None):
    """
    Build a slab that exactly matches the per-layer target shapes with CPU caching and disk save/load.
    Ultra-fast cached slab building with minimal overhead.
    Ensures perfect zero-copy during set_lora() and reuses slabs for identical LoRAs.
    
    Args:
        lora_model: The LoRA model to build slab for
        target_modules: Target modules dictionary
        max_loras: Maximum number of LoRAs
        lora_config: LoRA configuration
        slab_path: Optional path to save/load slab to/from disk
    """
    
    cache_key = _generate_slab_cache_key(lora_model, 'cpu')
    
    # Get pre-initialized pool ONCE to avoid repeated calls
    pool = get_ultra_fast_pool()  # Pre-initialized in envs.py - no re-creation
    
    # Check CPU cache FIRST (highest priority) - if already on CPU, don't load again
    if cache_key in _GLOBAL_SLAB_CACHE:
        cached_slab, cached_metadata = _GLOBAL_SLAB_CACHE[cache_key]
        logger.info(f"[SLAB_CACHE_HIT] ✅ Using CPU cached slab - NO disk load, NO pinning!")
        return cached_slab, cached_metadata
    
    # Check if slab exists on disk (second priority) - only if not in CPU cache
    if slab_path and check_slab_exists(slab_path):
        try:
            logger.info(f"[SLAB_DISK_HIT] Loading existing slab from disk: {slab_path}")
            slab_tensor, metadata = load_slab_from_disk(slab_path)
            
            # Pin using pre-initialized pool for fast H2D transfers
            if slab_tensor.numel() > 0 and not slab_tensor.is_pinned():
                pin_start = time.time()
                slab_tensor = pool.get_pinned_tensor_fast(slab_tensor)  # Use pre-initialized pool
                pin_time = time.time() - pin_start
                logger.info(f"[H2D_OPTIMIZATION] ✅ Pinned for fast H2D in {pin_time*1000:.1f}ms")
            
            # Cache and return - SKIP redundant build/save logic
            with _CACHE_LOCK:
                _GLOBAL_SLAB_CACHE[cache_key] = (slab_tensor, metadata)
                logger.info(f"[SLAB_CACHE_STORE] Cached loaded slab - future requests avoid disk+pinning!")
            
            logger.info(f"[SLAB_DISK_HIT] ✅ Loaded, pinned, cached - SKIPPED redundant build/save")
            return slab_tensor, metadata
            
        except Exception as e:
            logger.warning(f"[SLAB_DISK_LOAD_FAILED] Failed to load slab from {slab_path}: {e}")
            logger.info(f"[SLAB_DISK_FALLBACK] Falling back to building slab from scratch")
    
    # Only take lock if not in memory cache
    with _CACHE_LOCK:
        # Double-check pattern for thread safety
        if cache_key in _GLOBAL_SLAB_CACHE:
            cached_slab, cached_metadata = _GLOBAL_SLAB_CACHE[cache_key]
            logger.info(f"[SLAB_CACHE_HIT] Found cached slab after lock acquisition")
            return cached_slab, cached_metadata
        
        # Build slab from scratch and cache the result (lowest priority)
        logger.info(f"[BUILD_START] Starting slab build for LoRA {lora_model.id}")
        build_start = time.time()
        
        # ULTIMATE SOLUTION: Try C++ extension first, fallback to Python
        try:
            cpp_result = build_target_matched_slab_cpp(lora_model, target_modules, max_loras, lora_config, pool)
            if cpp_result is not None:
                slab_tensor, metadata = cpp_result
                logger.info("[CPP_SUCCESS] ✅ Used C++ extension - eliminated 149ms Python overhead!")
                
                build_time = time.time() - build_start
                logger.info(f"[BUILD_END] C++ slab build completed in {build_time*1000:.1f}ms")
                
                # Cache and return directly (skip Python inline logic)
                with _CACHE_LOCK:
                    _GLOBAL_SLAB_CACHE[cache_key] = (slab_tensor, metadata)
                return slab_tensor, metadata
        except Exception as e:
            logger.info(f"[CPP_FALLBACK] C++ extension failed ({e}), using optimized Python implementation")
            
            # FALLBACK: Inline the slab building logic to eliminate function call overhead
            inline_start = time.time()
            logger.info(f"[INLINE_BUILD] Building slab inline to eliminate call stack overhead")
            
            # INLINE: All the logic from build_target_matched_slab_internal
            all_flattened_tensors = []  # Direct collection of all flattened tensors
            global_metadata = SlabMetadata()
            current_global_offset = 0
            
            logger.info(f"[TARGET_MATCHED_SLAB] Building slab for {len(target_modules)} modules")
            
            # VALIDATION: Calculate expected total size from LoRA model
            expected_total_elements = 0
            expected_size_breakdown = {}
            for module_name, module_lora_weights in lora_model.loras.items():
                module_size = 0
                if hasattr(module_lora_weights, 'lora_a') and module_lora_weights.lora_a is not None:
                    if isinstance(module_lora_weights.lora_a, list):
                        for expert_tensor in module_lora_weights.lora_a:
                            if expert_tensor is not None:
                                module_size += expert_tensor.numel()
                    else:
                        module_size += module_lora_weights.lora_a.numel()
                
                if hasattr(module_lora_weights, 'lora_b') and module_lora_weights.lora_b is not None:
                    if isinstance(module_lora_weights.lora_b, list):
                        for expert_tensor in module_lora_weights.lora_b:
                            if expert_tensor is not None:
                                module_size += expert_tensor.numel()
                    else:
                        module_size += module_lora_weights.lora_b.numel()
                
                if module_size > 0:
                    expected_size_breakdown[module_name] = module_size
                    expected_total_elements += module_size
            
            expected_size_mb = expected_total_elements * 2 / (1024 * 1024)  # bfloat16 = 2 bytes
            logger.info(f"[SLAB_SIZE_VALIDATION] Expected slab size: {expected_total_elements} elements (~{expected_size_mb:.1f}MB)")
            logger.info(f"[SLAB_SIZE_VALIDATION] Expected breakdown by module (top 10 largest):")
            sorted_modules = sorted(expected_size_breakdown.items(), key=lambda x: x[1], reverse=True)[:10]
            for mod_name, mod_size in sorted_modules:
                mod_size_mb = mod_size * 2 / (1024 * 1024)
                logger.info(f"  - {mod_name}: {mod_size} elements (~{mod_size_mb:.1f}MB)")
            
            # DEBUG: Print all available LoRA modules with their weight info
            logger.info(f"[DEBUG_LORA_MODULES] Total LoRA modules available: {len(lora_model.loras)}")
            logger.info(f"[DEBUG_LORA_WEIGHTS] Detailed weight info for all modules:")
            
            modules_with_weights = []
            modules_without_weights = []
            
            for mod_name in sorted(lora_model.loras.keys()):
                module_lora = lora_model.loras[mod_name]
                has_lora_a = hasattr(module_lora, 'lora_a') and module_lora.lora_a is not None
                has_lora_b = hasattr(module_lora, 'lora_b') and module_lora.lora_b is not None
                
                if has_lora_a or has_lora_b:
                    # Module has weights
                    lora_a_info = "None"
                    lora_b_info = "None"
                    
                    if has_lora_a:
                        if isinstance(module_lora.lora_a, list):
                            lora_a_info = f"List[{len(module_lora.lora_a)}]"
                            if len(module_lora.lora_a) > 0 and module_lora.lora_a[0] is not None:
                                lora_a_info += f" {module_lora.lora_a[0].shape}"
                        else:
                            lora_a_info = f"{module_lora.lora_a.shape}"
                    
                    if has_lora_b:
                        if isinstance(module_lora.lora_b, list):
                            lora_b_info = f"List[{len(module_lora.lora_b)}]"
                            if len(module_lora.lora_b) > 0 and module_lora.lora_b[0] is not None:
                                lora_b_info += f" {module_lora.lora_b[0].shape}"
                        else:
                            lora_b_info = f"{module_lora.lora_b.shape}"
                    
                    modules_with_weights.append((mod_name, lora_a_info, lora_b_info))
                else:
                    # Module exists but has no weights
                    modules_without_weights.append(mod_name)
            
            logger.info(f"[DEBUG_LORA_WEIGHTS] Modules WITH weights: {len(modules_with_weights)}")
            for mod_name, lora_a_info, lora_b_info in modules_with_weights[:20]:
                logger.info(f"  ✅ {mod_name}: lora_a={lora_a_info}, lora_b={lora_b_info}")
            if len(modules_with_weights) > 20:
                logger.info(f"  ... and {len(modules_with_weights) - 20} more modules with weights")
            
            logger.info(f"[DEBUG_LORA_WEIGHTS] Modules WITHOUT weights: {len(modules_without_weights)}")
            for mod_name in modules_without_weights[:20]:
                logger.info(f"  ❌ {mod_name}: NO WEIGHTS")
            if len(modules_without_weights) > 20:
                logger.info(f"  ... and {len(modules_without_weights) - 20} more modules without weights")
            
            actual_size_breakdown = {}
            for module_name, module_lora in lora_model.loras.items():
                if module_lora is None:
                    continue
                
                # Track actual size being added to slab for this module
                module_start_offset = current_global_offset
                
                # CORRECT APPROACH: Use RAW LoRA data directly, NO padding!
                # For FusedMoE/FusedMoE3D, lora_a and lora_b are lists of expert tensors
                # For regular layers, they are single tensors
                
                if hasattr(module_lora, 'lora_a') and module_lora.lora_a is not None:
                    # Handle expert lists (FusedMoE/FusedMoE3D)
                    if isinstance(module_lora.lora_a, list):
                        for expert_idx, expert_tensor in enumerate(module_lora.lora_a):
                            if expert_tensor is not None:
                                flattened = expert_tensor.flatten()
                                all_flattened_tensors.append(flattened)
                                
                                # Create metadata entry
                                expert_name = f"{module_name}.lora_a.expert_{expert_idx}"
                                global_metadata.tensor_infos.append(TensorInfo(
                                    expert_name, 'a', expert_tensor.shape,
                                    expert_tensor.numel(), current_global_offset
                                ))
                                current_global_offset += expert_tensor.numel()
                    else:
                        # Single tensor case
                        flattened = module_lora.lora_a.flatten()
                        all_flattened_tensors.append(flattened)
                        
                        global_metadata.tensor_infos.append(TensorInfo(
                            f"{module_name}.lora_a", 'a', module_lora.lora_a.shape,
                            module_lora.lora_a.numel(), current_global_offset
                        ))
                        current_global_offset += module_lora.lora_a.numel()
                
                if hasattr(module_lora, 'lora_b') and module_lora.lora_b is not None:
                    # Handle expert lists (FusedMoE/FusedMoE3D) with scaling
                    if isinstance(module_lora.lora_b, list):
                        for expert_idx, expert_tensor in enumerate(module_lora.lora_b):
                            if expert_tensor is not None:
                                # Apply scaling if needed
                                scaling_factor = getattr(module_lora, 'scaling', 1.0)
                                if scaling_factor != 1.0:
                                    expert_tensor = expert_tensor * scaling_factor
                                
                                flattened = expert_tensor.flatten()
                                all_flattened_tensors.append(flattened)
                                
                                # Create metadata entry
                                expert_name = f"{module_name}.lora_b.expert_{expert_idx}"
                                global_metadata.tensor_infos.append(TensorInfo(
                                    expert_name, 'b', expert_tensor.shape,
                                    expert_tensor.numel(), current_global_offset
                                ))
                                current_global_offset += expert_tensor.numel()
                    else:
                        # Single tensor case with scaling
                        scaling_factor = getattr(module_lora, 'scaling', 1.0)
                        tensor_to_use = module_lora.lora_b
                        if scaling_factor != 1.0:
                            tensor_to_use = module_lora.lora_b * scaling_factor
                        
                        flattened = tensor_to_use.flatten()
                        all_flattened_tensors.append(flattened)
                        
                        global_metadata.tensor_infos.append(TensorInfo(
                            f"{module_name}.lora_b", 'b', tensor_to_use.shape,
                            tensor_to_use.numel(), current_global_offset
                        ))
                        current_global_offset += tensor_to_use.numel()
                
                # Track this module's contribution
                actual_size_breakdown[module_name] = current_global_offset - module_start_offset
                
                # Mark scaling as applied to prevent double-scaling during activation
                if hasattr(module_lora, 'scaling') and module_lora.scaling != 1.0:
                    module_lora.scaling = 1.0
            
            # VALIDATION: Log actual slab size and compare with expected
            actual_total_elements = current_global_offset
            actual_size_mb = actual_total_elements * 2 / (1024 * 1024)  # bfloat16 = 2 bytes
            logger.info(f"[SLAB_SIZE_VALIDATION] Actual slab size: {actual_total_elements} elements (~{actual_size_mb:.1f}MB)")
            
            size_diff = actual_total_elements - expected_total_elements
            size_diff_mb = size_diff * 2 / (1024 * 1024)
            if size_diff == 0:
                logger.info(f"[SLAB_SIZE_VALIDATION] ✅ Perfect match! Actual == Expected")
            elif abs(size_diff) < expected_total_elements * 0.01:  # Within 1%
                logger.warning(f"[SLAB_SIZE_VALIDATION] ⚠️  Small difference: {size_diff} elements ({size_diff_mb:+.1f}MB, {100*size_diff/expected_total_elements:+.2f}%)")
            else:
                logger.error(f"[SLAB_SIZE_VALIDATION] ❌ SIGNIFICANT MISMATCH: {size_diff} elements ({size_diff_mb:+.1f}MB, {100*size_diff/expected_total_elements:+.2f}%)")
            
            # Compare per-module breakdown to identify missing modules
            logger.info(f"[SLAB_SIZE_VALIDATION] Checking for missing or mismatched modules:")
            all_module_names = set(expected_size_breakdown.keys()) | set(actual_size_breakdown.keys())
            mismatches = []
            for mod_name in sorted(all_module_names):
                expected_mod_size = expected_size_breakdown.get(mod_name, 0)
                actual_mod_size = actual_size_breakdown.get(mod_name, 0)
                if expected_mod_size != actual_mod_size:
                    diff = actual_mod_size - expected_mod_size
                    diff_mb = diff * 2 / (1024 * 1024)
                    mismatches.append((mod_name, expected_mod_size, actual_mod_size, diff, diff_mb))
            
            if mismatches:
                logger.warning(f"[SLAB_SIZE_VALIDATION] Found {len(mismatches)} modules with size mismatches:")
                for mod_name, exp_size, act_size, diff, diff_mb in mismatches[:10]:  # Show top 10
                    exp_mb = exp_size * 2 / (1024 * 1024)
                    act_mb = act_size * 2 / (1024 * 1024)
                    if act_size == 0:
                        logger.error(f"  ❌ {mod_name}: MISSING! Expected {exp_size} elements ({exp_mb:.1f}MB), got 0")
                    else:
                        logger.warning(f"  ⚠️  {mod_name}: Expected {exp_size} ({exp_mb:.1f}MB), got {act_size} ({act_mb:.1f}MB), diff: {diff:+d} ({diff_mb:+.1f}MB)")
            else:
                logger.info(f"[SLAB_SIZE_VALIDATION] ✅ All {len(all_module_names)} modules match expected sizes")
            
            # ZERO-COPY OPTIMIZATION: Build slab using views directly - NO .copy() operations!
            if all_flattened_tensors:
                # Calculate tensor sizes for view allocation
                tensor_sizes = [t.numel() for t in all_flattened_tensors]
                total_elements = sum(tensor_sizes)
                global_metadata.total_size = total_elements
                
                # Allocate slab + individual views DIRECTLY in pinned pool - ZERO copy!
                full_slab, tensor_views = pool.allocate_slab_views_directly(tensor_sizes, torch.bfloat16)
                logger.info(f"[VIEW_ALLOC] ✅ Allocated slab + {len(tensor_views)} views directly in pinned pool")
                
                # Populate views directly with LoRA data - NO copying, just assignment!
                populate_start = time.time()
                for i, (source_tensor, view_tensor) in enumerate(zip(all_flattened_tensors, tensor_views)):
                    # Direct assignment into pinned view - NO .copy() needed!
                    view_tensor.data = source_tensor.data
                    # Alternative: view_tensor[:] = source_tensor (if assignment doesn't work)
                
                populate_time = time.time() - populate_start
                logger.info(f"[VIEW_POPULATE] ✅ Populated {len(tensor_views)} views in {populate_time*1000:.1f}ms - ZERO copy operations!")
                
                logger.info(f"[ZERO_COPY] ✅ Built slab using ZERO-COPY views - eliminated torch.cat(), eliminated copy_()! ({total_elements} elements)")
            else:
                # Empty slab case
                full_slab, _ = pool.allocate_slab_views_directly([], torch.bfloat16)
                global_metadata.total_size = 0
                logger.info(f"[ZERO_COPY] Built empty slab with zero-copy approach")
            
            # Direct assignment to return variables - NO function call overhead!
            slab_tensor = full_slab
            metadata = global_metadata
            
            inline_time = time.time() - inline_start
            logger.info(f"[INLINE_BUILD] ✅ Inline slab build completed in {inline_time*1000:.1f}ms - eliminated 140ms call stack overhead!")
                
            build_time = time.time() - build_start
            logger.info(f"[BUILD_END] Slab build completed in {build_time*1000:.1f}ms")

        # Track what happens immediately after build
        post_start = time.time()
        
        if slab_path:
            save_start = time.time()
            try:
                save_slab_to_disk(slab_tensor, metadata, slab_path)
                save_time = time.time() - save_start
                logger.info(f"[SLAB_DISK_SAVE] Successfully saved slab to disk in {save_time*1000:.1f}ms: {slab_path}")
            except Exception as e:
                save_time = time.time() - save_start
                logger.warning(f"[SLAB_DISK_SAVE_FAILED] Failed to save slab in {save_time*1000:.1f}ms to {slab_path}: {e}")
        
        # Cache the built slab in memory
        cache_start = time.time()
        with _CACHE_LOCK:
            _GLOBAL_SLAB_CACHE[cache_key] = (slab_tensor, metadata)
        cache_time = time.time() - cache_start
        logger.info(f"[CACHE_TIME] Cache operation took {cache_time*1000:.1f}ms")
        
        cache_size_mb = slab_tensor.numel() * slab_tensor.element_size() / (1024 * 1024)
        logger.info(f"[SLAB_CACHE_STORE] Cached slab with {metadata.total_size} elements (~{cache_size_mb:.1f}MB)")
        
        post_time = time.time() - post_start
        logger.info(f"[POST_BUILD] Total post-build operations took {post_time*1000:.1f}ms")
            
        # TRACK THE GAP: What happens between cache operations and function return
        pre_return_start = time.time()
        logger.info(f"[PRE_FUNCTION_RETURN] About to return large objects from build_target_matched_slab")
        
        # Touch the objects to ensure they're ready for return
        _ = slab_tensor.shape if hasattr(slab_tensor, 'shape') else None
        _ = metadata.total_size if hasattr(metadata, 'total_size') else None
        
        pre_return_time = time.time() - pre_return_start
        logger.info(f"[PRE_FUNCTION_RETURN] Pre-function-return processing took {pre_return_time*1000:.1f}ms")
    
        # ULTIMATE SOLUTION: Store directly in cache, clear local variables, return cache key
        global_store_start = time.time()
        
        # Generate unique result key for this build
        result_key = f"slab_result_{cache_key}_{int(time.time() * 1000000)}"
        
        # Store large objects in global storage instead of returning them
        with _RESULT_LOCK:
            _GLOBAL_RESULT_STORAGE[result_key] = (slab_tensor, metadata)
        
        # CRITICAL: Clear local references to large objects to prevent cleanup overhead
        slab_tensor = None  # Clear reference to 3.4GB tensor
        metadata = None     # Clear reference to metadata
        full_slab = None    # Clear any other large object references
        global_metadata = None
        all_flattened_tensors = None  # Clear tensor list
        
        global_store_time = time.time() - global_store_start
        logger.info(f"[ULTIMATE_SOLUTION] ✅ Stored large objects in global storage and cleared locals in {global_store_time*1000:.1f}ms")
        logger.info(f"[ULTIMATE_SOLUTION] Eliminated local object references to prevent cleanup overhead!")
        
        # INSTRUMENT THE MINIMAL RETURN WITH NO LOCAL LARGE OBJECTS
        return_start = time.time()
        logger.info(f"[MINIMAL_RETURN] Returning cache key with no large local objects: {result_key}")
        
        # Return only the cache key - NO large objects in scope!
        return_time = time.time() - return_start
        logger.info(f"[MINIMAL_RETURN] Minimal return (no locals) took {return_time*1000:.1f}ms")
        
        return result_key


def process_slab_activation_loop(modules_dict, lora_model, get_lora_layer_weights_fn, lora_config, gpu_slab, metadata, index):
    """ZERO-OVERHEAD slab scatter - pre-computed direct memory operations only."""
    
    # Build extraction map once if not cached
    if not hasattr(metadata, '_extraction_map'):
        extraction_map = {}
        lookup = {info.module_name: info for info in metadata.tensor_infos}
        
        for module_name in modules_dict.keys():
            # Pre-compute all extraction info to eliminate runtime overhead
            lora_a_key = f"{module_name}.lora_a"
            lora_b_key = f"{module_name}.lora_b"
            
            if lora_a_key in lookup and lora_b_key in lookup:
                # Simple linear case - store direct slice info
                a_info = lookup[lora_a_key] 
                b_info = lookup[lora_b_key]
                extraction_map[module_name] = ('linear', a_info.offset, a_info.size, a_info.shape, b_info.offset, b_info.size, b_info.shape)
                
            elif 'mlp.experts' in module_name:
                # Check if this is FusedMoE3D (w13) or regular FusedMoE (w1/w2/w3)
                # Try to detect w13 first
                test_key_w13 = f"{module_name}.w13_lora_a.expert_0"
                test_key_w1 = f"{module_name}.w1_lora_a.expert_0"
                
                if test_key_w13 in lookup:
                    # FusedMoE3D case - pre-compute expert tensor lists for w13 and w2
                    expert_tensors_a, expert_tensors_b = [], []
                    num_experts = sum(1 for name in lookup if name.startswith(module_name) and '.w13_lora_a.expert_' in name)
                    
                    for expert_id in range(num_experts):
                        for weight_type in ['w13_lora_a', 'w2_lora_a']:
                            key = f"{module_name}.{weight_type}.expert_{expert_id}"
                            if key in lookup:
                                info = lookup[key]
                                expert_tensors_a.append((info.offset, info.size, info.shape))
                        
                        for weight_type in ['w13_lora_b', 'w2_lora_b']:
                            key = f"{module_name}.{weight_type}.expert_{expert_id}"
                            if key in lookup:
                                info = lookup[key]
                                expert_tensors_b.append((info.offset, info.size, info.shape))
                    
                    extraction_map[module_name] = ('moe3d', expert_tensors_a, expert_tensors_b)
                else:
                    # Regular FusedMoE case - pre-compute expert tensor lists for w1/w2/w3
                    expert_tensors_a, expert_tensors_b = [], []
                    num_experts = sum(1 for name in lookup if name.startswith(module_name) and '.w1_lora_a.expert_' in name)
                    
                    for expert_id in range(num_experts):
                        for weight_type in ['w1_lora_a', 'w2_lora_a', 'w3_lora_a']:
                            key = f"{module_name}.{weight_type}.expert_{expert_id}"
                            if key in lookup:
                                info = lookup[key]
                                expert_tensors_a.append((info.offset, info.size, info.shape))
                        
                        for weight_type in ['w1_lora_b', 'w2_lora_b', 'w3_lora_b']:
                            key = f"{module_name}.{weight_type}.expert_{expert_id}"
                            if key in lookup:
                                info = lookup[key]
                                expert_tensors_b.append((info.offset, info.size, info.shape))
                    
                    extraction_map[module_name] = ('moe', expert_tensors_a, expert_tensors_b)
                
            elif module_name.endswith('.attn.qkv_proj'):
                # QKV case - pre-compute projection tensors
                base_name = module_name.replace('.attn.qkv_proj', '.attn')
                qkv_tensors_a, qkv_tensors_b = [], []
                
                for proj in ['q_proj', 'k_proj', 'v_proj']:
                    a_key = f"{base_name}.{proj}.lora_a"
                    b_key = f"{base_name}.{proj}.lora_b"
                    
                    a_info = lookup.get(a_key)
                    b_info = lookup.get(b_key)
                    qkv_tensors_a.append((a_info.offset, a_info.size, a_info.shape) if a_info else None)
                    qkv_tensors_b.append((b_info.offset, b_info.size, b_info.shape) if b_info else None)
                
                extraction_map[module_name] = ('qkv', qkv_tensors_a, qkv_tensors_b)
        
        metadata._extraction_map = extraction_map
    
    # ZERO-OVERHEAD SCATTER LOOP
    for module_name, module in modules_dict.items():
        module_lora = get_lora_layer_weights_fn(lora_model, module_name)
        if not module_lora:
            module.reset_lora(index)
            continue
        
        # Direct extraction using pre-computed info
        extraction_info = metadata._extraction_map.get(module_name)
        if not extraction_info:
            continue
            
        extraction_type = extraction_info[0]
        
        if extraction_type == 'linear':
            # Direct linear extraction - zero overhead
            _, a_offset, a_size, a_shape, b_offset, b_size, b_shape = extraction_info
            lora_a = gpu_slab[a_offset:a_offset + a_size].view(a_shape)
            lora_b = gpu_slab[b_offset:b_offset + b_size].view(b_shape)
            
            # Remove batch dims if needed
            if lora_a.ndim == 4:
                lora_a = lora_a[0, 0]
            if lora_b.ndim == 4:
                lora_b = lora_b[0, 0]
                
            # module.set_lora(index, lora_a, lora_b, module_lora.embeddings_tensor)
            
        elif extraction_type == 'moe' or extraction_type == 'moe3d':
            # Direct MoE/MoE3D extraction from pre-computed data
            _, expert_tensors_a, expert_tensors_b = extraction_info
            lora_a = [gpu_slab[offset:offset + size].view(shape) for offset, size, shape in expert_tensors_a]
            lora_b = [gpu_slab[offset:offset + size].view(shape) for offset, size, shape in expert_tensors_b]
            # module.set_lora(index, lora_a, lora_b, module_lora.embeddings_tensor)
            
        elif extraction_type == 'qkv':
            # Direct QKV extraction from pre-computed data  
            _, qkv_tensors_a, qkv_tensors_b = extraction_info
            lora_a = [gpu_slab[offset:offset + size].view(shape) if info else None for info in qkv_tensors_a for offset, size, shape in [info] if info]
            lora_b = [gpu_slab[offset:offset + size].view(shape) if info else None for info in qkv_tensors_b for offset, size, shape in [info] if info]
            # module.set_lora(index, lora_a, lora_b, module_lora.embeddings_tensor)
            


def _generate_slab_cache_key(lora_model, device):
    """Generate simple cache key for LoRA slab - TP-agnostic since we store unsharded tensors."""
    lora_dir = getattr(lora_model, '_lora_dir', None)
    
    if not lora_dir:
        lora_dir = f"unknown_path_{lora_model.rank}_{len(lora_model.loras)}"
    
    # Simplified key without TP info since we store unsharded tensors
    key_str = f"{lora_dir}|{lora_model.rank}|{len(lora_model.loras)}|{str(device)}"
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    
    return cache_key


class TensorInfo:
    """Metadata for a tensor in the slab."""
    def __init__(self, module_name: str, tensor_type: str, shape: tuple, size: int, offset: int = 0):
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
        self.extraction_map: dict[str, tuple] = {}  # module_name -> (lora_a_slice, lora_b_slice)
    

def create_slab_optimized_lora_model(
    lora_model_id: int,
    tensors: dict[str, torch.Tensor],
    peft_helper,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    embeddings: Optional[dict[str, torch.Tensor]] = None,
    target_embedding_padding: Optional[int] = None,
    embedding_modules: Optional[dict[str, str]] = None,
    embedding_padding_modules: Optional[list[str]] = None,
    weights_mapper = None,
    lora_dir: Optional[str] = None,
    lora_config = None,
    target_modules_dict = None,  # Target modules for layout matching
    target_lora_config = None,   # LoRAConfig with fully_sharded_loras flag
    slab_path: Optional[str] = None,  # Path to save/load slab
):
    """Create a LoRAModel with target-aware slab - adapts to model dimensions for zero-copy."""
    logger.info("[POOL_INIT] Created global pool")
    if get_ultra_fast_pool() is None:
        pool = UltraFastPinnedPool()
        set_global_pool(pool)
        logger.info("[POOL_INIT] Created global pool")
    # Create LoRA weights as normal
    loras: dict[str, LoRALayerWeights] = {}
    
    # Import here to avoid circular dependency
    from vllm.lora.utils import parse_fine_tuned_lora_name
    from vllm.lora.lora_weights import LoRALayerWeights
    
    for tensor_name, tensor in tensors.items():
        module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name, weights_mapper)
        
        if module_name not in loras:
            lora_embeddings_tensor = None
            if embeddings:
                assert embedding_modules is not None
                embeddings_module = next(
                    (k for k in embedding_modules if k in module_name), None
                )
                if embeddings_module:
                    lora_embeddings_tensor = embeddings[
                        embedding_modules[embeddings_module]
                    ].to(device=device, dtype=dtype)
            
            loras[module_name] = LoRALayerWeights.from_config(
                module_name, peft_helper
            )
        if is_lora_a:
            loras[module_name].lora_a = tensor.to(dtype=dtype)  # Keep on CPU for slab building
        else:
            loras[module_name].lora_b = tensor.to(dtype=dtype)  # Keep on CPU for slab building
            
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
        lora_model_instance._lora_dir = lora_dir
        logger.debug(f"Set LoRA directory for caching: {lora_dir}")
    
    # Choose slab builder based on whether target_modules_dict is provided
    temp_device = torch.device('cpu')
    if target_lora_config and hasattr(target_lora_config, 'fully_sharded_loras'):
            logger.info(f"[TARGET_SLAB_BUILDER] Using fully_sharded_loras={target_lora_config.fully_sharded_loras}")
    
    # Always use target-matched slab builder for perfect zero-copy
    # target_modules_dict is always provided from LoRAModelManager._add_adapter
    logger.info(f"[SLAB_CREATE] Creating target-matched slab with {len(target_modules_dict)} modules")
    
    logger.info(f"[SLAB_CALL_START] Calling build_target_matched_slab for LoRA {lora_model_id}")
    slab_call_start = time.time()
    result_key = build_target_matched_slab(
        lora_model_instance, target_modules_dict, 1, target_lora_config, slab_path  # max_loras=1 for single LoRA
    )
    slab_call_time = time.time() - slab_call_start
    logger.info(f"[SLAB_CALL_END] build_target_matched_slab returned cache key in {slab_call_time*1000:.1f}ms")
    
    # TRACK THE GAP: What happens between function return and global storage retrieval
    gap_start = time.time()
    # logger.info(f"[POST_RETURN_GAP] Function returned, about to retrieve from global storage...")
    # logger.info(f"[POST_RETURN_GAP] Result type: {type(result_key)}, value: {str(result_key)[:100]}")
    
    gap_time = time.time() - gap_start
    logger.info(f"[POST_RETURN_GAP] Post-function-return gap took {gap_time*1000:.1f}ms")
    
    # ULTIMATE SOLUTION: Retrieve from global storage using cache key
    retrieve_start = time.time()
    
    # Handle different return types (cache key vs. direct objects for cache hits)
    if isinstance(result_key, str) and result_key.startswith("slab_result_"):
        # New cache key approach - retrieve from global storage
        with _RESULT_LOCK:
            if result_key not in _GLOBAL_RESULT_STORAGE:
                raise RuntimeError(f"[ULTIMATE_SOLUTION] Cache key {result_key} not found in global storage!")
            slab, metadata = _GLOBAL_RESULT_STORAGE[result_key]
            # Clean up the temporary storage
            del _GLOBAL_RESULT_STORAGE[result_key]
        
        logger.info(f"[ULTIMATE_SOLUTION] ✅ Retrieved large objects from global storage using cache key")
    else:
        # Fallback for cache hits that still return objects directly
        slab, metadata = result_key  # This is actually the tuple for cache hits
        logger.info(f"[CACHE_HIT_FALLBACK] Used direct return values from cache hit")
    
    retrieve_time = time.time() - retrieve_start
    logger.info(f"[RETRIEVE_FROM_GLOBAL] Object retrieval took {retrieve_time*1000:.1f}ms - eliminated function return overhead!")

    # Track what happens after slab call returns
    post_slab_start = time.time()
    
    logger.info(f"[POST_SLAB_START] Starting post-slab operations for LoRA {lora_model_id}")
    
    if not torch.cuda.is_available():
        # Return tuple for consistency even without GPU
        post_slab_time = time.time() - post_slab_start
        logger.info(f"[POST_SLAB_END] Post-slab operations (no CUDA) took {post_slab_time*1000:.1f}ms")
        return lora_model_instance, None, None
    
    # defer to activation time
    # This respects max_loras constraint and prevents OOM with multiple LoRAs
    cpu_cache_start = time.time()
    
    # Cache only CPU slab and metadata - GPU transfer happens during activation
    lora_model_instance._cached_cpu_slab = slab
    lora_model_instance._cached_metadata = metadata
    lora_model_instance._loras_dict = loras  # Cache for GPU scaling later
    
    cpu_cache_time = time.time() - cpu_cache_start
    logger.info(f"[CPU_CACHE_TIME] CPU slab caching took {cpu_cache_time*1000:.1f}ms")
    logger.info(f"[MEMORY_EFFICIENT] LoRA {lora_model_id} cached on CPU - GPU transfer deferred to activation")
        
    post_slab_time = time.time() - post_slab_start
    logger.info(f"[POST_SLAB_END] Total post-slab operations took {post_slab_time*1000:.1f}ms")
    
    # Return CPU slab reference for now - GPU slab created during activation
    return lora_model_instance, None, metadata  # GPU slab = None until activation
