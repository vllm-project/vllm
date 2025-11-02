# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.forward_context import ForwardContext, get_forward_context

if TYPE_CHECKING:
    from vllm.attention import Attention
    from vllm.attention.layer import MLAAttention


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        layer_name: str = args[layer_name_index]
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return func(*args, **kwargs)
        assert isinstance(attn_metadata, dict)
        # Wait for KV layer on entry
        connector = get_kv_transfer_group()
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        result = func(*args, **kwargs)

        # Save KV cache layer on exit
        attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[
            layer_name
        ]
        kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata[layer_name])

        return result

    return wrapper
