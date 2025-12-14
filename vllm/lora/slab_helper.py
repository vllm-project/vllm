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
    
    def __init__(self, initial_pool_size: int = 4*1024 * 1024 * 1024):  # 4GB initial pool
        self.pool_size = initial_pool_size
        # Pre-allocate large pinned buffer at startup - one-time 1.7s cost
        self.pinned_pool = torch.empty(initial_pool_size, dtype=torch.uint8).pin_memory()
        self.pool_lock = threading.RLock()
        self.used_ranges = []  # Track used memory ranges
        
        # OPTION 2 IMPLEMENTATION: Store current slab and metadata as instance variables
        # This eliminates the 149ms Python function return overhead for large objects
        self.current_slab = None
        self.current_metadata = None
            
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
            
            self.used_ranges.append((start_offset, end_offset))
            
            # Return direct view of pinned pool - NO copy needed!
            pool_slice = self.pinned_pool[start_offset:end_offset]
            slab_tensor = pool_slice.view(torch.uint8).view(dtype)[:num_elements]
            
            return slab_tensor

    def get_pinned_tensor_fast(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        """Ultra-fast pseudo-pinning using pre-allocated pool (20x faster than pin_memory)."""
        tensor_bytes = cpu_tensor.numel() * cpu_tensor.element_size()
        
        with self.pool_lock:
            # Find available space in pool
            if tensor_bytes > self.pool_size:
                # Expand pool if needed
                new_size = max(self.pool_size * 2, tensor_bytes + self.pool_size)
                
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

def get_ultra_fast_pool():
    """Get the pre-initialized global pool - NO lazy initialization."""
    global _ULTRA_FAST_POOL
    if _ULTRA_FAST_POOL is None:
        # Fallback - create pool if not set (shouldn't happen)
        with _POOL_INIT_LOCK:
            if _ULTRA_FAST_POOL is None:
                _ULTRA_FAST_POOL = UltraFastPinnedPool()
    return _ULTRA_FAST_POOL





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
        
        for module_name, module_lora in lora_model.loras.items():
            if module_lora is None:
                continue
            
            # Track actual size being added to slab for this module
            module_start_offset = current_global_offset
                            
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
                            
            # Mark scaling as applied to prevent double-scaling during activation
            if hasattr(module_lora, 'scaling') and module_lora.scaling != 1.0:
                module_lora.scaling = 1.0
        
        # ZERO-COPY OPTIMIZATION: Build slab using views directly - NO .copy() operations!
        if all_flattened_tensors:
            # Calculate tensor sizes for view allocation
            tensor_sizes = [t.numel() for t in all_flattened_tensors]
            total_elements = sum(tensor_sizes)
            global_metadata.total_size = total_elements
            
            # Allocate slab + individual views DIRECTLY in pinned pool - ZERO copy!
            full_slab, tensor_views = pool.allocate_slab_views_directly(tensor_sizes, torch.bfloat16)
            
            for i, (source_tensor, view_tensor) in enumerate(zip(all_flattened_tensors, tensor_views)):
                view_tensor.data = source_tensor.data
        else:
            # Empty slab case
            full_slab, _ = pool.allocate_slab_views_directly([], torch.bfloat16)
            global_metadata.total_size = 0            
        # Direct assignment to return variables - NO function call overhead!
        slab_tensor = full_slab
        metadata = global_metadata
                
        # Cache the built slab in memory
        with _CACHE_LOCK:
            _GLOBAL_SLAB_CACHE[cache_key] = (slab_tensor, metadata)
        
        # Touch the objects to ensure they're ready for return
        _ = slab_tensor.shape if hasattr(slab_tensor, 'shape') else None
        _ = metadata.total_size if hasattr(metadata, 'total_size') else None
                
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
    if get_ultra_fast_pool() is None:
        pool = UltraFastPinnedPool()
        set_global_pool(pool)
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
    
    result_key = build_target_matched_slab(
        lora_model_instance, target_modules_dict, 1, target_lora_config, slab_path  
    )    
    
    # Handle different return types (cache key vs. direct objects for cache hits)
    if isinstance(result_key, str) and result_key.startswith("slab_result_"):
        slab, metadata = _GLOBAL_RESULT_STORAGE[result_key]
        # Clean up the temporary storage
        del _GLOBAL_RESULT_STORAGE[result_key]
        
    else:
        # Fallback for cache hits that still return objects directly
        slab, metadata = result_key
    
    if not torch.cuda.is_available():
        # Return tuple for consistency even without GPU
        return lora_model_instance, None, None
    
    # Cache only CPU slab and metadata - GPU transfer happens during activation
    lora_model_instance._cached_cpu_slab = slab
    lora_model_instance._cached_metadata = metadata
    lora_model_instance._loras_dict = loras  # Cache for GPU scaling later
    
    # Return CPU slab reference for now - GPU slab created during activation
    return lora_model_instance, None, metadata  # GPU slab = None until activation
