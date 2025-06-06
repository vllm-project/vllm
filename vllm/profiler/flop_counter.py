# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["FlopCounter", "FlopContextManager", "get_flop_counts"]


@dataclass
class FlopCount:
    """Container for FLOP counts for different operation types."""
    # Matrix operations
    mm: int = 0  # Matrix multiplication
    bmm: int = 0  # Batch matrix multiplication
    addmm: int = 0  # Add matrix multiplication

    # Attention operations
    scaled_dot_product_attention: int = 0

    # Activation functions
    softmax: int = 0
    log_softmax: int = 0
    gelu: int = 0
    silu: int = 0
    relu: int = 0

    # Normalization
    layer_norm: int = 0
    rms_norm: int = 0
    group_norm: int = 0

    # Embedding operations
    embedding: int = 0

    # Convolution operations
    conv1d: int = 0
    conv2d: int = 0

    # Other operations
    other: int = 0

    def total(self) -> int:
        """Return total FLOP count."""
        return sum(
            getattr(self, field_name.name)
            for field_name in self.__dataclass_fields__.values())

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            field_name.name: getattr(self, field_name.name)
            for field_name in self.__dataclass_fields__.values()
        }

    def __add__(self, other: 'FlopCount') -> 'FlopCount':
        """Add two FlopCount objects."""
        result = FlopCount()
        for field_name in self.__dataclass_fields__:
            setattr(result, field_name,
                    getattr(self, field_name) + getattr(other, field_name))
        return result

    def __iadd__(self, other: 'FlopCount') -> 'FlopCount':
        """In-place addition of FlopCount objects."""
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name,
                    getattr(self, field_name) + getattr(other, field_name))
        return self


@dataclass
class DetailedFlopCount:
    """Detailed FLOP counts organized by operation type and layer."""
    operation_counts: dict[str, int] = field(default_factory=dict)
    layer_counts: dict[str, FlopCount] = field(default_factory=dict)
    total_flops: int = 0

    def add_operation(self,
                      op_name: str,
                      flops: int,
                      layer_name: Optional[str] = None):
        """Add FLOP count for a specific operation.
        
        Args:
            op_name: Name of the operation (e.g., 'aten::mm').
            flops: Number of floating point operations.
            layer_name: Optional name of the layer performing the operation.
        """
        self.operation_counts[op_name] = (
            self.operation_counts.get(op_name, 0) + flops)
        self.total_flops += flops

        if layer_name:
            if layer_name not in self.layer_counts:
                self.layer_counts[layer_name] = FlopCount()

            flop_field = self._map_op_to_field(op_name)
            if flop_field:
                current_value = getattr(self.layer_counts[layer_name],
                                        flop_field)
                setattr(self.layer_counts[layer_name], flop_field,
                        current_value + flops)

    def _map_op_to_field(self, op_name: str) -> Optional[str]:
        """Map operation name to FlopCount field.
        
        Args:
            op_name: PyTorch operation name (e.g., 'aten::mm').
        """
        mapping = {
            'aten::mm': 'mm',
            'aten::bmm': 'bmm',
            'aten::addmm': 'addmm',
            'aten::scaled_dot_product_attention':
            ('scaled_dot_product_attention'),
            'aten::softmax': 'softmax',
            'aten::log_softmax': 'log_softmax',
            'aten::gelu': 'gelu',
            'aten::silu': 'silu',
            'aten::relu': 'relu',
            'aten::layer_norm': 'layer_norm',
            'aten::rms_norm': 'rms_norm',
            'aten::group_norm': 'group_norm',
            'aten::embedding': 'embedding',
            'aten::conv1d': 'conv1d',
            'aten::conv2d': 'conv2d',
        }
        return mapping.get(op_name, 'other')


class FlopCounter(TorchDispatchMode):
    """A TorchDispatchMode that counts FLOPs for various PyTorch operations.
    
    This class intercepts PyTorch operations and estimates the number of 
    floating-point operations performed, providing both total counts and
    per-operation breakdowns.
    
    Args:
        enabled: Whether FLOP counting is enabled.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.flop_counts = DetailedFlopCount()
        self._op_flop_count = defaultdict(int)
        self._module_stack: list[str] = []
        self._hooks = []
        self._setup_module_hooks()

    def _setup_module_hooks(self):
        """Setup module hooks to track the module execution stack."""
        import torch.nn as nn

        self._original_call = nn.Module.__call__

        def hooked_call(module_self, *args, **kwargs):
            if hasattr(module_self, '_get_name'):
                module_name = module_self._get_name()
            else:
                module_name = module_self.__class__.__name__

            self._module_stack.append(module_name)

            try:
                result = self._original_call(module_self, *args, **kwargs)
                return result
            finally:
                # Always pop the stack, even if an exception occurs
                if self._module_stack:
                    self._module_stack.pop()

        nn.Module.__call__ = hooked_call

    def _cleanup_module_hooks(self):
        """Cleanup module hooks."""
        import torch.nn as nn
        if hasattr(self, '_original_call'):
            nn.Module.__call__ = self._original_call

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if self.enabled:
            flops = self._compute_flops(func, args, kwargs)
            if flops > 0:
                op_name = str(func)
                layer_name = (".".join(self._module_stack)
                              if self._module_stack else None)
                self.flop_counts.add_operation(op_name, flops, layer_name)
                self._op_flop_count[op_name] += flops

        return func(*args, **kwargs)

    def _compute_flops(self, func, args: tuple, kwargs: dict[str, Any]) -> int:
        """Compute FLOPs for a given operation.
        
        Args:
            func: PyTorch function being dispatched.
            args: Function arguments.
            kwargs: Function keyword arguments.
        """
        func_packet = func._overloadpacket

        if func_packet == torch.ops.aten.mm:
            return self._mm_flops(args)
        elif func_packet == torch.ops.aten.bmm:
            return self._bmm_flops(args)
        elif func_packet == torch.ops.aten.addmm:
            return self._addmm_flops(args)
        elif func_packet == torch.ops.aten.scaled_dot_product_attention:
            return self._attention_flops(args)
        elif func_packet in [
                torch.ops.aten.softmax, torch.ops.aten.log_softmax
        ]:
            return self._softmax_flops(args)
        elif func_packet in [
                torch.ops.aten.gelu, torch.ops.aten.silu, torch.ops.aten.relu
        ]:
            return self._activation_flops(args)
        elif func_packet == torch.ops.aten.layer_norm:
            return self._layer_norm_flops(args)
        elif func_packet == torch.ops.aten.rms_norm:
            return self._rms_norm_flops(args)
        elif func_packet == torch.ops.aten.group_norm:
            return self._group_norm_flops(args)
        elif func_packet == torch.ops.aten.embedding:
            return self._embedding_flops(args)
        elif func_packet == torch.ops.aten.conv1d:
            return self._conv1d_flops(args)
        elif func_packet == torch.ops.aten.conv2d:
            return self._conv2d_flops(args)

        return 0

    def _mm_flops(self, args: tuple) -> int:
        """FLOPs for matrix multiplication: 2 * M * N * K.
        
        Args:
            args: Function arguments containing input tensors.
        """
        if len(args) >= 2:
            a, b = args[0], args[1]
            if (hasattr(a, 'shape') and hasattr(b, 'shape')
                    and len(a.shape) >= 2 and len(b.shape) >= 2):
                M, K = a.shape[-2], a.shape[-1]
                N = b.shape[-1]
                return 2 * M * N * K
        return 0

    def _bmm_flops(self, args: tuple) -> int:
        """FLOPs for batch matrix multiplication: batch_size * 2 * M * N * K"""
        if len(args) >= 2:
            a, b = args[0], args[1]
            if (hasattr(a, 'shape') and hasattr(b, 'shape')
                    and len(a.shape) >= 3 and len(b.shape) >= 3):
                batch_size = a.shape[0]
                M, K = a.shape[-2], a.shape[-1]
                N = b.shape[-1]
                return batch_size * 2 * M * N * K
        return 0

    def _addmm_flops(self, args: tuple) -> int:
        """FLOPs for addmm: 2 * M * N * K (matrix mult) + M * N (addition)"""
        mm_flops = self._mm_flops(args[1:])
        if len(args) >= 3:
            result_tensor = args[1] if hasattr(args[1], 'shape') else args[2]
            if (hasattr(result_tensor, 'shape')
                    and len(result_tensor.shape) >= 2):
                M, N = result_tensor.shape[-2], result_tensor.shape[-1]
                add_flops = M * N
                return mm_flops + add_flops
        return mm_flops

    def _attention_flops(self, args: tuple) -> int:
        """FLOPs for scaled dot-product attention."""
        if len(args) >= 3:
            query, key, value = args[0], args[1], args[2]
            if all(hasattr(t, 'shape') for t in [query, key, value]):
                batch_size = query.shape[0]
                if len(query.shape) == 4:
                    seq_len = query.shape[-2]
                    head_dim = query.shape[-1]
                    num_heads = (query.shape[1] if query.shape[1]
                                 < query.shape[2] else query.shape[2])
                else:
                    seq_len = query.shape[1]
                    head_dim = query.shape[2]
                    num_heads = 1

                # QK^T: multiply-add operations count as 2 FLOPs each
                qk_flops = (2 * batch_size * num_heads * seq_len * seq_len *
                            head_dim)
                av_flops = (2 * batch_size * num_heads * seq_len * seq_len *
                            head_dim)
                softmax_flops = batch_size * num_heads * seq_len * seq_len * 5

                return qk_flops + av_flops + softmax_flops
        return 0

    def _softmax_flops(self, args: tuple) -> int:
        """FLOPs for softmax: approximately 5 ops per element."""
        if len(args) >= 1 and hasattr(args[0], 'numel'):
            return 5 * args[0].numel()
        return 0

    def _activation_flops(self, args: tuple) -> int:
        """FLOPs for activation functions: approximately 3 ops per element."""
        if len(args) >= 1 and hasattr(args[0], 'numel'):
            return 3 * args[0].numel()
        return 0

    def _layer_norm_flops(self, args: tuple) -> int:
        """FLOPs for layer normalization: approximately 5 ops per element."""
        if len(args) >= 1 and hasattr(args[0], 'numel'):
            return 5 * args[0].numel()
        return 0

    def _rms_norm_flops(self, args: tuple) -> int:
        """FLOPs for RMS normalization: approximately 4 ops per element."""
        if len(args) >= 1 and hasattr(args[0], 'numel'):
            return 4 * args[0].numel()
        return 0

    def _group_norm_flops(self, args: tuple) -> int:
        """FLOPs for group normalization: approximately 5 ops per element."""
        if len(args) >= 1 and hasattr(args[0], 'numel'):
            return 5 * args[0].numel()
        return 0

    def _embedding_flops(self, args: tuple) -> int:
        """FLOPs for embedding lookup: 0 FLOPs."""
        return 0

    def _conv1d_flops(self, args: tuple) -> int:
        """FLOPs for 1D convolution."""
        if len(args) >= 2:
            input_tensor, weight = args[0], args[1]
            if hasattr(input_tensor, 'shape') and hasattr(weight, 'shape'):
                batch_size = input_tensor.shape[0]
                in_channels = weight.shape[1]
                out_channels = weight.shape[0]
                kernel_size = weight.shape[2] if len(weight.shape) > 2 else 1
                output_length = input_tensor.shape[2]

                return (batch_size * out_channels * output_length *
                        in_channels * kernel_size)
        return 0

    def _conv2d_flops(self, args: tuple) -> int:
        """FLOPs for 2D convolution."""
        if len(args) >= 2:
            input_tensor, weight = args[0], args[1]
            if hasattr(input_tensor, 'shape') and hasattr(weight, 'shape'):
                batch_size = input_tensor.shape[0]
                in_channels = weight.shape[1]
                out_channels = weight.shape[0]
                kernel_h, kernel_w = weight.shape[2], weight.shape[3]
                output_h, output_w = (input_tensor.shape[2],
                                      input_tensor.shape[3])

                return (batch_size * out_channels * output_h * output_w *
                        in_channels * kernel_h * kernel_w)
        return 0

    def get_total_flops(self) -> int:
        """Get total FLOP count."""
        return self.flop_counts.total_flops

    def get_flop_breakdown(self) -> dict[str, int]:
        """Get FLOP breakdown by operation type."""
        return dict(self._op_flop_count)

    def get_detailed_counts(self) -> DetailedFlopCount:
        """Get detailed FLOP counts."""
        return self.flop_counts

    def reset(self):
        """Reset all counters."""
        self.flop_counts = DetailedFlopCount()
        self._op_flop_count.clear()
        self._module_stack.clear()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_module_hooks()
        return super().__exit__(exc_type, exc_val, exc_tb)


# Global thread-local storage for FLOP counters
_thread_local = threading.local()


def _get_flop_counter() -> Optional[FlopCounter]:
    """Get the current thread's FLOP counter."""
    return getattr(_thread_local, 'flop_counter', None)


def _set_flop_counter(counter: Optional[FlopCounter]):
    """Set the current thread's FLOP counter."""
    _thread_local.flop_counter = counter


@contextmanager
def FlopContextManager():
    """Context manager for FLOP counting.
    
    Usage:
        with FlopContextManager() as flop_counter:
            # Your model operations here
            outputs = model(inputs)
        
        print(f"Total FLOPs: {flop_counter.get_total_flops()}")
        print(f"Breakdown: {flop_counter.get_flop_breakdown()}")
    """
    counter = FlopCounter()
    old_counter = _get_flop_counter()
    _set_flop_counter(counter)

    try:
        with counter:
            yield counter
    finally:
        counter._cleanup_module_hooks()
        _set_flop_counter(old_counter)


def get_flop_counts() -> Optional[DetailedFlopCount]:
    """Get FLOP counts from the current context.
    
    Returns:
        DetailedFlopCount if a FLOP counter is active, None otherwise.
    """
    counter = _get_flop_counter()
    return counter.get_detailed_counts() if counter else None


def format_flops(flops: int) -> str:
    """Format FLOP count in human-readable units.
    
    Args:
        flops: Number of floating point operations.
    """
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops} FLOPs"
