# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ACE attention capture hook.

This module hooks into vLLM's scheduler output pipeline to capture per-request
attention weights and feed them into the AttentionImportanceTracker registry.

## Integration points

1. **Prefill**: When `context_compression="ace"` is active, register a tracker
   for the request in `_tracker_registry`. The scheduler sets `capture_attn=True`
   on the SequenceGroup metadata.

2. **Attention layer**: When `capture_attn=True`, each Attention layer returns
   its softmax weights alongside the output. The scheduler accumulates these
   via `AttentionImportanceTracker.accumulate()`.

3. **Next turn**: Before applying eviction, `apply_ace_eviction()` calls
   `get_tracker(request_id)` to retrieve the accumulated scores.

## Lightweight integration path (no attention layer changes)

For a zero-change-to-attention-kernels implementation, vLLM's existing
`return_attention_weights` option in flash-attention backends can be used
when available. For models/backends without this option, we fall back to
BM25 (Mode 2) automatically via the `tracker.has_data` guard in
`apply_ace_eviction()`.

The integration is purely additive: no existing paths are modified.
`ACEAttentionCapture` is a no-op when not enabled.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


class ACEAttentionCapture:
    """
    Per-request accumulator that sits between the attention layers and the
    ACE context_compression module.

    Usage in the scheduler / model runner::

        capture = ACEAttentionCapture(request_id, max_seq_len=model_config.max_model_len)
        capture.start()           # registers tracker, marks request as capturing

        # ... inference runs; each attention layer calls:
        capture.on_layer_output(layer_idx, attn_weights, new_token_start)

        capture.stop()            # signals that this generation step is done

        # Next turn: context_compression reads the tracker via get_tracker()

    Thread-safe: multiple generation threads can call on_layer_output concurrently
    for different requests because each capture instance owns its own tracker.
    """

    def __init__(self, request_id: str, max_seq_len: int = 32768) -> None:
        self.request_id = request_id
        self.max_seq_len = max_seq_len
        self._active = False
        self._layer_buffers: dict[int, list] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        """Register this request's tracker and activate capture."""
        from vllm.entrypoints.context_compression import register_tracker
        register_tracker(self.request_id, max_seq_len=self.max_seq_len)
        self._active = True

    def on_layer_output(
        self,
        layer_idx: int,
        attn_weights: "np.ndarray",
        new_token_start: int,
    ) -> None:
        """
        Called by each attention layer after computing softmax weights.

        Args:
            layer_idx:       Index of the layer (0-based).
            attn_weights:    Softmax attention weights, shape
                             [n_heads, n_new_tokens, seq_len].
            new_token_start: Position of the first new token.
        """
        if not self._active:
            return
        with self._lock:
            self._layer_buffers.setdefault(layer_idx, []).append(
                (attn_weights, new_token_start)
            )

    def flush(self) -> None:
        """
        Aggregate layer buffers and push to the tracker.

        Call this once per generation step, after all layers have completed.
        Resets internal buffers for the next step.
        """
        if not self._active:
            return

        from vllm.entrypoints.context_compression import get_tracker
        import numpy as np

        tracker = get_tracker(self.request_id)
        if tracker is None:
            return

        with self._lock:
            buffers = self._layer_buffers
            self._layer_buffers = {}

        if not buffers:
            return

        # Stack layers: [n_layers, n_heads, n_new_tokens, seq_len]
        # Each buffer entry is (attn_weights[n_heads, n_new_tok, seq_len], tok_start)
        stacked: list[np.ndarray] = []
        tok_start = None
        for layer_idx in sorted(buffers.keys()):
            for weights, start in buffers[layer_idx]:
                arr = np.asarray(weights, dtype=np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis]  # [1, n_new_tok, seq_len] — single head
                stacked.append(arr)
                tok_start = start  # all layers share the same new_token_start

        if not stacked or tok_start is None:
            return

        # Pad to uniform seq_len dimension before stacking
        max_seq = max(a.shape[-1] for a in stacked)
        padded = [
            np.pad(a, ((0, 0), (0, 0), (0, max_seq - a.shape[-1])))
            if a.shape[-1] < max_seq else a
            for a in stacked
        ]
        batch = np.stack(padded)  # [n_layers, n_heads, n_new_tokens, seq_len]
        tracker.accumulate(batch, new_token_start=tok_start)

    def stop(self) -> None:
        """Flush remaining buffers and deactivate. Tracker stays registered."""
        self.flush()
        self._active = False

    def release(self) -> None:
        """Remove the tracker after the request is fully complete."""
        from vllm.entrypoints.context_compression import release_tracker
        release_tracker(self.request_id)
        self._active = False


# ---------------------------------------------------------------------------
# Global request-capture registry (request_id → ACEAttentionCapture)
# ---------------------------------------------------------------------------
# Populated by serving.py when context_compression="ace" is requested.
# Queried by the model runner / attention layer hook.

_capture_registry: dict[str, ACEAttentionCapture] = {}
_capture_lock = threading.Lock()


def start_capture(
    request_id: str,
    max_seq_len: int = 32768,
) -> ACEAttentionCapture:
    """Create and start an attention capture for a request."""
    cap = ACEAttentionCapture(request_id, max_seq_len=max_seq_len)
    cap.start()
    with _capture_lock:
        _capture_registry[request_id] = cap
    return cap


def get_capture(request_id: str) -> Optional[ACEAttentionCapture]:
    """Retrieve the active capture for a request."""
    with _capture_lock:
        return _capture_registry.get(request_id)


def stop_capture(request_id: str) -> None:
    """Flush and stop capture, but keep tracker alive for next-turn eviction."""
    with _capture_lock:
        cap = _capture_registry.pop(request_id, None)
    if cap is not None:
        cap.stop()


def release_capture(request_id: str) -> None:
    """
    Fully release a request's capture and tracker.

    Call this when the conversation is over and the tracker is no longer
    needed — not between turns.
    """
    with _capture_lock:
        cap = _capture_registry.pop(request_id, None)
    if cap is not None:
        cap.release()
    else:
        # Tracker may exist without a capture if the generation step
        # finished already — release it directly.
        from vllm.entrypoints.context_compression import release_tracker
        release_tracker(request_id)
