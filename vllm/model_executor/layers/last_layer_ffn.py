# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_indices: ContextVar[torch.Tensor | None] = ContextVar(
    "last_layer_ffn_indices", default=None
)


@contextmanager
def using_last_layer_ffn_indices(
    indices: torch.Tensor | None,
) -> Iterator[None]:
    """Publish row indices for this forward and restore the previous value."""
    token = _indices.set(indices)
    try:
        yield
    finally:
        _indices.reset(token)


def current_last_layer_ffn_indices() -> torch.Tensor | None:
    return _indices.get()


@contextmanager
def use_output_lora_mapping(module: torch.nn.Module):
    """Use per-output adapter metadata for an FFN reduced to logits rows."""
    wrappers = {
        child.punica_wrapper
        for child in module.modules()
        if hasattr(child, "punica_wrapper")
    }
    original = [(wrapper, wrapper.token_mapping_meta) for wrapper in wrappers]
    try:
        for wrapper, _ in original:
            wrapper.token_mapping_meta = wrapper.prompt_mapping_meta
        yield
    finally:
        for wrapper, metadata in original:
            wrapper.token_mapping_meta = metadata


def apply_pruned_last_ffn(hidden, residual, norm, mlp):
    indices = _indices.get()
    if indices is not None:
        hidden = hidden.index_select(0, indices)
        residual = residual.index_select(0, indices)
    hidden, residual = norm(hidden, residual)
    if indices is None:
        return mlp(hidden), residual
    with use_output_lora_mapping(mlp):
        return mlp(hidden), residual


def restore_pruned_last_ffn(hidden: torch.Tensor, num_tokens: int) -> torch.Tensor:
    indices = _indices.get()
    if indices is None:
        return hidden
    return hidden.new_zeros((num_tokens, hidden.shape[-1])).index_copy_(
        0, indices, hidden
    )


class _TransformerFFNHooks:
    def __init__(self):
        self.mapping = None
        self.shape = None

    def pre(self, module, args):
        indices = _indices.get()
        if indices is None:
            return None
        (hidden,) = args
        self.shape = hidden.shape
        self.mapping = use_output_lora_mapping(module)
        self.mapping.__enter__()
        return (hidden.index_select(-2, indices),)

    def post(self, module, args, output):
        if self.mapping is not None:
            self.mapping.__exit__(None, None, None)
            self.mapping = None
        indices = _indices.get()
        if output is None or indices is None:
            return output
        return output.new_zeros(self.shape).index_copy_(-2, indices, output)


def install_transformer_ffn_hooks(ffn: torch.nn.Module) -> None:
    hooks = _TransformerFFNHooks()
    ffn.register_forward_pre_hook(hooks.pre)
    ffn.register_forward_hook(hooks.post, always_call=True)
