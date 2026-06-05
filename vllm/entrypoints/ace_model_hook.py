# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE model runner hook — wires attention weight capture into vLLM's forward pass.

## Integration

This module provides two integration paths:

### Path A — PyTorch forward hooks (no kernel changes)

Works for any model that uses standard attention modules. Registers a
``register_forward_hook`` on each attention layer to capture the softmax
weights that the module computed.

Limitation: FlashAttention2 and FlashInfer do NOT return softmax weights in
their optimized kernels. For these backends, ACE automatically falls back to
BM25 scoring (Mode 2) via the ``tracker.has_data`` guard in
``apply_ace_eviction()``.

### Path B — Kernel-level capture (requires backend support)

For backends that expose ``return_softmax=True`` (e.g. TorchSDPABackend in
debug mode), the attention layer can be patched to also return the softmax
matrix. Requires a one-line change per backend's ``forward()`` method.

## Usage

In the model runner, for each request with ``context_compression="ace"`` and
a ``conversation_id``:

    from vllm.entrypoints.ace_model_hook import ACEModelHook

    hook = ACEModelHook(conversation_id, model, max_seq_len=model_config.max_model_len)
    hook.install()          # register PyTorch forward hooks on attention layers
    # ... run model.forward() ...
    hook.flush_and_stop()   # aggregate layer buffers → AttentionImportanceTracker
    # hook.uninstall() called automatically by flush_and_stop
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


# Attention module names that vLLM uses across model families.
# These are the class names of the attention sub-modules in model.named_modules().
_ATTN_MODULE_NAMES = frozenset([
    "Attention",
    "SelfAttention",
    "MultiheadAttention",
    "LlamaAttention",
    "GemmaAttention",
    "MistralAttention",
    "Qwen2Attention",
    "FlashAttention",
    "PagedAttention",
])


class ACEModelHook:
    """
    Installs PyTorch forward hooks on a model's attention layers to capture
    attention weights for ACE Phase 3.

    Thread-safe: one hook per request, each with its own capture instance.

    Args:
        conversation_id:  Stable ID linking this request to its turn history.
        model:            The vLLM model (``nn.Module``).
        max_seq_len:      Max sequence length for the tracker.
    """

    def __init__(
        self,
        conversation_id: str,
        model: "torch.nn.Module",
        max_seq_len: int = 32768,
    ) -> None:
        self.conversation_id = conversation_id
        self.model = model
        self.max_seq_len = max_seq_len
        self._hooks: list[Any] = []
        self._capture = None
        self._lock = threading.Lock()
        self._new_token_start: int = 0

    def set_new_token_start(self, position: int) -> None:
        """
        Set the position of the first newly generated token.

        Must be called once per generation step, before the model forward pass,
        so the capture knows which positions in the attention matrix are new.

        Args:
            position: Absolute sequence position of the first new token.
        """
        self._new_token_start = position

    def install(self) -> None:
        """Register forward hooks on all attention layers."""
        from vllm.entrypoints.attention_capture import get_capture, start_capture

        with self._lock:
            cap = get_capture(self.conversation_id)
            if cap is None:
                cap = start_capture(self.conversation_id, max_seq_len=self.max_seq_len)
            self._capture = cap

        try:
            import torch.nn as nn
        except ImportError:
            return

        for name, module in self.model.named_modules():
            # Match any class whose name contains an attention-related keyword
            cls_name = type(module).__name__
            if any(attn_kw in cls_name for attn_kw in ("Attention", "Attn")):
                hook = module.register_forward_hook(
                    self._make_hook(name), with_kwargs=False
                )
                self._hooks.append(hook)

    def _make_hook(self, layer_name: str):
        """Return a forward hook closure for a specific attention layer."""

        def _hook(module, inputs, output):
            # output may be a tensor (hidden states) or a tuple
            # (hidden_states, attn_weights) depending on the backend.
            attn_weights = None
            if isinstance(output, tuple) and len(output) >= 2:
                candidate = output[1]
                if hasattr(candidate, "shape") and candidate.ndim >= 3:
                    attn_weights = candidate

            if attn_weights is None:
                # Backend did not return softmax weights — skip this layer.
                # apply_ace_eviction() will fall back to BM25 automatically.
                return output

            with self._lock:
                cap = self._capture
                pos = self._new_token_start

            if cap is not None:
                try:
                    import numpy as np
                    w = attn_weights.detach().float().cpu().numpy()
                    # Expected shape: [batch, n_heads, n_new_tokens, seq_len]
                    # or [n_heads, n_new_tokens, seq_len] — normalize to 3D.
                    # In batched inference the hook fires once for the whole
                    # batch; w[0] would attribute another request's attention to
                    # this conversation. Only capture when this request is the
                    # sole batch occupant — otherwise skip and let ACE fall back
                    # to BM25 (Mode 2) via the tracker.has_data guard.
                    if w.ndim == 4:
                        if w.shape[0] != 1:
                            return output  # cannot isolate this request's weights
                        w = w[0]  # safe: batch size 1
                    layer_idx = id(module)  # use object id as stable key
                    cap.on_layer_output(layer_idx, w, new_token_start=pos)
                except Exception:
                    pass  # never block inference

            return output

        return _hook

    def flush_and_stop(self) -> None:
        """Flush accumulated layer buffers, stop capture, remove hooks."""
        self.uninstall()
        from vllm.entrypoints.attention_capture import stop_capture
        stop_capture(self.conversation_id)

    def uninstall(self) -> None:
        """Remove all registered forward hooks."""
        with self._lock:
            for h in self._hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            self._hooks.clear()


# ---------------------------------------------------------------------------
# Convenience: per-request hook registry
# ---------------------------------------------------------------------------

_hook_registry: dict[str, ACEModelHook] = {}
_hook_lock = threading.Lock()


def install_hook(
    conversation_id: str,
    model: "torch.nn.Module",
    max_seq_len: int = 32768,
    new_token_start: int = 0,
) -> ACEModelHook:
    """
    Create and install an ACEModelHook for ``conversation_id``.

    Call once per generation step, before the model forward pass.
    Idempotent: if a hook is already installed for this ID, returns it.
    """
    with _hook_lock:
        if conversation_id in _hook_registry:
            hook = _hook_registry[conversation_id]
            hook.set_new_token_start(new_token_start)
            return hook

    hook = ACEModelHook(conversation_id, model, max_seq_len=max_seq_len)
    hook.set_new_token_start(new_token_start)
    hook.install()
    with _hook_lock:
        _hook_registry[conversation_id] = hook
    return hook


def flush_hook(conversation_id: str) -> None:
    """
    Flush and remove the hook for ``conversation_id`` after generation.

    The tracker remains alive for the next request's eviction step.
    """
    with _hook_lock:
        hook = _hook_registry.pop(conversation_id, None)
    if hook is not None:
        hook.flush_and_stop()


def release_hook(conversation_id: str) -> None:
    """
    Fully release hook + tracker after the conversation ends.

    Call when the client closes the session or sends a new ``conversation_id``.
    """
    flush_hook(conversation_id)
    from vllm.entrypoints.attention_capture import release_capture
    release_capture(conversation_id)
