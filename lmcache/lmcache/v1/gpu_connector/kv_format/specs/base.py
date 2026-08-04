# SPDX-License-Identifier: Apache-2.0
"""Per-format geometry interface for GPU KV caches.

Each :class:`KVFormatSpec` holds the geometry accessors for one
``EngineKVFormat`` -- methods that read shape off a normalized ``kv_caches``.
The format's static facts (MLA and the structural shape) live on the
``EngineKVFormat`` enum itself (``csrc/engine_kv_format.h``, read via
``lmc_ops``), shared with the device kernels. The enum is the single source of
truth for which formats exist; engine identity lives only in detection. The
format -> spec table is in ``registry.py``.
"""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
import lmcache.c_ops as lmc_ops

# A format's enum name *is* its shape: ``_``-joined tokens, with ``X`` marking a
# list level. ``TWO_X_NL_X_NBBS_NH_HS`` reads as ``2 x NL x [PBS, NH, HS]``.
# describe_shape and concrete_shape both render from it, so they can never drift.
_LABELS = {
    "ONE": "1",
    "TWO": "2",
    "NBBS": "PBS",
    "NB": "NB",
    "NL": "NL",
    "BS": "BS",
    "NH": "NH",
    "HS": "HS",
    "CS": "CS",
    # Blocked-scale indexer cache: per-block value / scale regions. Rendered
    # symbolically; the logical per-layer tensor is [NB, BS, HS(=132)].
    "BSV": "BSxVALS",
    "BSS": "BSxSCALES",
}
_ACCESSORS = {
    "NB": "num_blocks",
    "NL": "num_layers",
    "BS": "block_size",
    "NH": "num_heads",
    "HS": "head_size",
    "CS": "head_size",
    "PBS": "page_buffer_size",
    # Blocked-scale regions: sized by tokens per block (the value/scale byte
    # widths are format constants, not per-model geometry).
    "BSxVALS": "block_size",
    "BSxSCALES": "block_size",
}


def _render_shape(fmt: "lmc_ops.EngineKVFormat", token: Callable[[str], str]) -> str:
    *lists, inner = fmt.name.split("_X_")
    body = ", ".join(token(t) for t in inner.split("_"))
    return " x ".join([token(t) for t in lists] + [f"[{body}]"])


def describe_shape(fmt: "lmc_ops.EngineKVFormat") -> str:
    """Symbolic shape of a format, e.g. ``NL_X_NB_BS_HS`` -> ``NL x [NB, BS, HS]``.

    Named ``describe_shape`` (not ``shape_desc``) to avoid confusion with the
    unrelated :class:`lmc_ops.PageBufferShapeDesc` and its ``shape_desc``
    instances used on the transfer path.
    """
    return _render_shape(fmt, lambda t: _LABELS[t])


def concrete_shape(fmt: "lmc_ops.EngineKVFormat", size: Callable[[str], int]) -> str:
    """Numeric shape of a format; ``size(label)`` gives each axis's dimension.

    E.g. ``NL_X_TWO_NB_BS_NH_HS`` with ``NL=32, NB=2048, BS=16, NH=8, HS=128``
    -> ``32 x [2, 2048, 16, 8, 128]``.
    """
    return _render_shape(
        fmt, lambda t: _LABELS[t] if t in ("ONE", "TWO") else str(size(_LABELS[t]))
    )


class KVFormatSpec(ABC):
    """Pure geometry accessors for one ``EngineKVFormat``.

    Wraps an already-normalized ``kv_caches`` (from ``detect_format``) and
    answers geometry questions about it. Layout only -- no engine identity,
    since one format may come from many (engine, backend) pairs.

    Class attributes: ``engine_kv_format`` (the format this describes, its
    identity) and ``attention_backends`` (diagnostic labels; first is the
    representative). The format's static facts -- ``is_mla`` and the structural
    shape ``is_cross_layer`` / ``is_kv_list`` / ``is_layer_list`` -- live on the
    ``EngineKVFormat`` itself (``csrc/engine_kv_format.h``, read via ``lmc_ops``),
    shared with the device kernels.

    Method usage by mode -- every spec is consumed through the ``get_*``
    facade in ``gpu_connector.utils``:

    * Used by **both** MP (multiprocess) and non-MP (in-process) connectors:
      :meth:`num_layers`, :meth:`num_blocks`, :meth:`block_size`,
      :meth:`num_heads`, :meth:`hidden_dim`, :meth:`head_size`, :meth:`dtype`,
      :meth:`data_ptrs`.
    * Used by **non-MP only** (the legacy in-process GPU/XPU/HPU/MUSA
      connectors, none of the MP transfer path): :meth:`page_buffer_size`,
      :meth:`tokens_per_layer`, :meth:`elements_per_layer`. The MP path derives
      these from a per-group :class:`PageBufferShapeDesc` instead.

    Lifetime: a spec **borrows** ``kv_caches`` -- it does not own the GPU KV
    tensors. ``get_spec`` builds a fresh instance per call and callers use it
    transiently (``get_spec(...).num_layers()``), so the borrowed reference is
    released as soon as the spec is dropped. Never cache a spec on a long-lived
    object: that would keep the engine's GPU KV tensors alive past disconnect.
    """

    engine_kv_format: ClassVar["lmc_ops.EngineKVFormat"]
    attention_backends: ClassVar[tuple[str, ...]] = ()

    def __init__(self, kv_caches: DiscoverableKVCache) -> None:
        # Borrowed, not owned: see the class docstring's "Lifetime" note. The
        # spec must stay transient so it never outlives the engine's KV tensors.
        self.kv_caches = kv_caches

    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers."""

    @abstractmethod
    def num_blocks(self) -> int:
        """Return the number of paged blocks (raises for NBBS-fused formats)."""

    @abstractmethod
    def block_size(self, layer_idx: int = 0) -> int:
        """Return the block size (tokens per block; raises for NBBS-fused)."""

    @abstractmethod
    def page_buffer_size(self) -> int:
        """Return ``num_blocks * block_size`` (or the fused PBS axis).

        Non-MP only (see the class docstring).
        """

    @abstractmethod
    def kv_size(self) -> int:
        """Return the number of dimensions for "KV". For normal specs, it's 2.
        Some specs has a fused KV (e.g., MLA, or new vLLM format), so it's 1.
        """

    @abstractmethod
    def num_heads(self, layer_idx: int = 0) -> int:
        """Return the number of KV heads for ``layer_idx`` (1 for MLA)."""

    @abstractmethod
    def hidden_dim(self, layer_idx: int = 0) -> int:
        """Return the hidden dimension (``num_heads * head_size``) for a layer."""

    @abstractmethod
    def head_size(self, layer_idx: int = 0) -> int:
        """Return the per-head dimension for ``layer_idx``."""

    @abstractmethod
    def tokens_per_layer(self) -> int:
        """Return the token capacity per layer (``num_blocks * block_size``).

        Non-MP only (see the class docstring).
        """

    @abstractmethod
    def elements_per_layer(self) -> int:
        """Return the element count per layer (both K and V for non-MLA).

        Non-MP only (see the class docstring).
        """

    @abstractmethod
    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        """Return the tensor dtype for ``layer_idx``."""

    @abstractmethod
    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        """Return device pointers for ``layer_indices`` in kernel-expected order.

        Per-layer formats: one pointer per layer. SGLang two-list MHA: all K
        pointers then all V. Cross-layer: a single base pointer (the kernel
        walks layers itself, so ``layer_indices`` is ignored).
        """

    def concrete_shape_str(self) -> str:
        """``describe_shape`` with real dims, e.g. ``80 x [2, 2048, 128, 8, 128]``."""
        return concrete_shape(
            self.engine_kv_format, lambda label: getattr(self, _ACCESSORS[label])()
        )
