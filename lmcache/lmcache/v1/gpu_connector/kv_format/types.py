# SPDX-License-Identifier: Apache-2.0
"""Foundational types for the KV-cache format layer.

``DiscoverableKVCache`` is the canonical KV-cache structure every ``kv_format``
module operates on; ``LayoutHints`` carries the engine-supplied registration
hints. The ``utils.py`` facade re-exports both for backward-compatible call
sites.
"""

# Standard
from typing import Literal, TypedDict, Union

# Third Party
import torch

# Canonical recursive type consumed by ``detect_format`` and the
# downstream format-aware helpers. A value is either a single
# :class:`torch.Tensor` (e.g. vLLM cross-layer, TRT-LLM) or a list of
# nested ``DiscoverableKVCache`` values (per-layer lists, SGLang's
# two-list MHA, deeper nesting). Engine adapters that hand us other
# containers (e.g. vLLM's ``dict[str, torch.Tensor]``) are responsible
# for unwrapping to this form before calling the helpers.
DiscoverableKVCache = Union[torch.Tensor, list["DiscoverableKVCache"]]


class LayoutHints(TypedDict, total=False):
    """Hints passed from a serving engine to LMCache during KV cache
    registration (``REGISTER_KV_CACHE``).

    Serving engines may pass a plain ``dict`` that satisfies this
    schema -- importing this type is optional.

    Keys:
        kv_layout: Physical ordering of the KV cache dimensions.
            ``"NHD"`` -- heads after block-size (default for most
            vLLM builds).
            ``"HND"`` -- heads before block-size (``VLLM_KV_CACHE_LAYOUT=HND``).
        num_kv_heads: Number of KV heads per layer. Used by TRT-LLM to
            reshape its 4-D pool tensor into the canonical 6-D form.
        tokens_per_block: Tokens per paged block. Used by TRT-LLM (to
            reshape its pool tensor) and by SGLang MHA (to split the
            folded ``page_buffer_size`` dimension into separate
            ``num_blocks`` and ``block_size``). Presence of this field
            on a SGLang registration is what triggers the daemon-side
            depth-1 -> depth-2 un-flatten + 3-D -> 4-D reshape.
        head_dim: Per-head dimension. Used by TRT-LLM (same).
    """

    kv_layout: Literal["NHD", "HND"]
    num_kv_heads: int
    tokens_per_block: int
    head_dim: int
