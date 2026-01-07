# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Utilities for invalidating compilation and CUDA graph caches.

This module provides utilities to invalidate CUDA graph and torch.compile caches
when model weights change (e.g., when LoRA adapters are loaded). This ensures
that subsequent compilations include the new operations.

Environment Variables:
    VLLM_DISABLE_LORA_CACHE_INVALIDATION: If set to '1', disables automatic
        cache invalidation when LoRA adapters are loaded (not recommended).
"""

import os
import torch
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def is_cache_invalidation_disabled() -> bool:
    """Check if cache invalidation is disabled via environment variable."""
    return os.environ.get('VLLM_DISABLE_LORA_CACHE_INVALIDATION', '0') == '1'


def invalidate_cudagraph_cache(model: Any) -> int:
    """
    Invalidate CUDA graph caches in the model hierarchy.
    
    This function walks through the model hierarchy and clears any
    CUDAGraphWrapper caches it finds. This is necessary when model
    weights change (e.g., LoRA adapters are loaded) to ensure the
    CUDA graphs are recaptured with the new weights.
    
    Args:
        model: The model or module to invalidate caches for.
        
    Returns:
        Number of caches cleared.
    """
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    
    num_caches_cleared = 0
    
    # Check if the model itself is wrapped with CUDAGraphWrapper
    if isinstance(model, CUDAGraphWrapper):
        num_entries = len(model.concrete_cudagraph_entries)
        if num_entries > 0:
            logger.info(
                "Clearing %d CUDA graph cache entries from CUDAGraphWrapper",
                num_entries
            )
            model.concrete_cudagraph_entries.clear()
            num_caches_cleared += 1
    
    # Check if model has a forward method that's wrapped
    if hasattr(model, 'forward') and isinstance(
        getattr(model, 'forward', None), CUDAGraphWrapper
    ):
        wrapper = model.forward
        num_entries = len(wrapper.concrete_cudagraph_entries)
        if num_entries > 0:
            logger.info(
                "Clearing %d CUDA graph cache entries from forward method",
                num_entries
            )
            wrapper.concrete_cudagraph_entries.clear()
            num_caches_cleared += 1
    
    # Recursively check submodules
    if hasattr(model, 'modules'):
        for module in model.modules():
            if module is not model:  # Avoid infinite recursion
                # Check if module's forward is wrapped
                if hasattr(module, 'forward'):
                    forward_fn = getattr(module, 'forward', None)
                    if isinstance(forward_fn, CUDAGraphWrapper):
                        num_entries = len(forward_fn.concrete_cudagraph_entries)
                        if num_entries > 0:
                            logger.debug(
                                "Clearing %d CUDA graph entries from %s",
                                num_entries,
                                type(module).__name__
                            )
                            forward_fn.concrete_cudagraph_entries.clear()
                            num_caches_cleared += 1
    
    # Check for __call__ method wrapper
    if hasattr(model, '__call__') and isinstance(
        getattr(model, '__call__', None), CUDAGraphWrapper
    ):
        wrapper = model.__call__
        num_entries = len(wrapper.concrete_cudagraph_entries)
        if num_entries > 0:
            logger.info(
                "Clearing %d CUDA graph cache entries from __call__ method",
                num_entries
            )
            wrapper.concrete_cudagraph_entries.clear()
            num_caches_cleared += 1
    
    return num_caches_cleared


def invalidate_torch_compile_cache(model: Any) -> None:
    """
    Invalidate torch.compile cache for the model.
    
    This clears the dynamo cache entries for compiled functions in the model,
    forcing recompilation on the next forward pass.
    
    Args:
        model: The model to invalidate compile cache for.
    """
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    
    # Check if model's forward method is compiled
    if hasattr(model, 'forward'):
        forward_fn = getattr(model, 'forward', None)
        
        # Check if it's wrapped with TorchCompileWithNoGuardsWrapper
        if isinstance(forward_fn, TorchCompileWithNoGuardsWrapper):
            # Reset compilation state
            forward_fn.compiled = False
            forward_fn.first_compile = True
            logger.info("Reset torch.compile state for model forward")
            
            # Clear dynamo cache for the original code object
            if hasattr(forward_fn, 'original_code_object'):
                try:
                    code_obj = forward_fn.original_code_object()
                    torch._dynamo.eval_frame.remove_from_cache(code_obj)
                    logger.info("Cleared dynamo cache for forward method")
                except Exception as e:
                    logger.warning("Failed to clear dynamo cache: %s", e)
    
    # Clear dynamo cache for the entire model
    try:
        if hasattr(model, '__class__') and hasattr(model.__class__, 'forward'):
            code_obj = model.__class__.forward.__code__
            torch._dynamo.eval_frame.remove_from_cache(code_obj)
            logger.info("Cleared dynamo cache for model class")
    except Exception as e:
        logger.debug("Could not clear model class cache: %s", e)


def invalidate_all_caches(model: Any) -> None:
    """
    Invalidate all compilation caches (CUDA graphs and torch.compile).
    
    This should be called whenever model weights change in a way that would
    affect the computational graph, such as when LoRA adapters are loaded.
    
    Args:
        model: The model to invalidate caches for.
    """
    logger.info("Invalidating all compilation caches due to model changes")
    
    # Clear CUDA graph caches
    num_cleared = invalidate_cudagraph_cache(model)
    if num_cleared > 0:
        logger.info("Cleared %d CUDA graph cache(s)", num_cleared)
    else:
        logger.debug("No CUDA graph caches found to clear")
    
    # Clear torch.compile caches
    invalidate_torch_compile_cache(model)
    
    # Force CUDA cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Cache invalidation complete. Next forward pass will trigger recompilation.")

