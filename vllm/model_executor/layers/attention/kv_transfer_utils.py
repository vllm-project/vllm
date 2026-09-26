# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.utils.torch_utils import _resolve_layer_name


def maybe_wait_for_kv_layer(layer_name: str) -> None:
    """Block the caller on the connector's KV load for ``layer_name``.

    ``maybe_transfer_kv_layer`` gives this to every layer that dispatches
    through a unified attention op. Models that own their attention call and
    register themselves directly into ``static_forward_context``, never reach
    that decorator, so they must ask for the wait explicitly. Without it a
    layer-wise connector's asynchronous copies are never ordered against the
    compute that reads them, and the model reads KV that has not landed, causing
    corruption output rather than raising.

    The guards mirror the decorator's.
    """
    # Imported here rather than at module scope: vllm.forward_context imports
    # back into the attention package.
    from vllm.forward_context import get_forward_context

    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    # No metadata means a profile or dummy run, with no request to load for.
    if get_forward_context().attn_metadata is None:
        return
    connector = get_kv_transfer_group()
    if not connector.has_connector_metadata():
        return
    connector.wait_for_layer_load(layer_name)


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    from vllm.model_executor.layers.attention.attention import get_attention_context

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

        layer_name = _resolve_layer_name(args[layer_name_index])

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        result = func(*args, **kwargs)

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        return result

    return wrapper
