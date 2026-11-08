# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernels speculative decoding adds to verify and commit, by key, and the
registry that binds them (see ``docs/features/speculative_decoding/round_contract.md``).

``SpeculatorConfig.spec_kernels`` names the speculator's kernels; a layer missing a
kernel verifies chains.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm.v1.spec_decode.resources import PlanKind
from vllm.v1.worker.gpu.spec_decode.capabilities import SpecLayerKind

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.spec_decode.round import AcceptResult

Op = Literal["commit"]
KernelKey = tuple[str, Op]
"""``(SpecLayerKind.key(), op)``, e.g. ``("layout:paged_token", "commit")``."""


def key_name(key: KernelKey) -> str:
    """``"layout:paged_token/commit"``: a key as logs and docs spell it."""
    return f"{key[0]}/{key[1]}"


@dataclass(frozen=True)
class SpecLayerSpec:
    """A target layer as a spec kernel sees it."""

    name: str
    """The layer's prefix, its key in the forward context."""
    kind: SpecLayerKind
    module: Any
    """The layer itself: its per-rank shapes, dtypes and ``kv_cache``."""
    layout: str = ""
    """The ``layout`` variant its commit moves; ``""`` keeps the model's own."""
    block_table: torch.Tensor | None = None
    """At bind: the block table of the layer's KV cache group."""
    block_size: int = 0
    """At bind: tokens per block of that group."""
    num_blocks: int = 0
    """At bind: kernel-size blocks in the layer's cache."""


PAGED_KV_COMMIT: KernelKey = ("layout:paged_token", "commit")


@dataclass(frozen=True)
class PagedKVLayout:
    """Where the KV caches of one KV cache group keep each token, for a
    ``layout:paged_token`` commit kernel (fields and indexing:
    ``round_contract.md``, Kernel registry)."""

    caches: tuple[torch.Tensor, ...]
    shape: tuple[int, int, int, int, int]
    strides: tuple[int, int, int, int, int]
    dtype: torch.dtype
    block_table: torch.Tensor
    table_block_size: int

    @property
    def block_size(self) -> int:
        return self.shape[1]


def paged_kv_layout_of(
    cache: torch.Tensor, num_blocks: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(shape, strides)`` in :class:`PagedKVLayout` order of an attention
    ``[blocks, heads, block_size, content]`` or MLA ``[blocks, block_size,
    content]`` cache view."""
    shape, stride = tuple(cache.shape), cache.stride()
    if cache.dim() == 4 and shape[0] == num_blocks:
        blocks, heads, slots, dim = shape
        strides = (stride[0], stride[2], 0, stride[1], stride[3])
        return (blocks, slots, 1, heads, dim), strides
    if cache.dim() == 3 and shape[0] == num_blocks:
        blocks, slots, dim = shape
        return (blocks, slots, 1, 1, dim), (stride[0], stride[1], 0, 0, stride[2])
    raise ValueError(
        f"no paged KV layout for a KV cache view of shape {shape} and strides "
        f"{stride} ({num_blocks} kernel blocks expected first)"
    )


def paged_kv_layouts(layers: Sequence[SpecLayerSpec]) -> tuple[PagedKVLayout, ...]:
    """The :class:`PagedKVLayout` of each KV cache group of ``layers``."""
    # (block table, table block size, shape, strides, dtype) -> caches by data_ptr
    groups: dict[tuple, dict[int, torch.Tensor]] = {}
    tables: dict[int, torch.Tensor] = {}
    for layer in layers:
        cache, table = layer.module.kv_cache, layer.block_table
        if table is None or table.dtype != torch.int32:
            raise ValueError(f"{layer.name}: a paged KV commit needs an int32 table")
        shape, strides = paged_kv_layout_of(cache, layer.num_blocks)
        if strides[4] != 1:
            raise ValueError(f"{layer.name}: the KV cache's head dim is strided")
        tables[table.data_ptr()] = table
        key = (table.data_ptr(), layer.block_size, shape, strides, cache.dtype)
        groups.setdefault(key, {}).setdefault(cache.data_ptr(), cache)
    layouts = []
    for (table_ptr, table_block_size, shape, strides, dtype), caches in groups.items():
        layouts.append(
            PagedKVLayout(
                caches=tuple(caches.values()),
                shape=shape,
                strides=strides,
                dtype=dtype,
                block_table=tables[table_ptr],
                table_block_size=table_block_size,
            )
        )
    return tuple(layouts)


@dataclass(frozen=True)
class KernelContext:
    """What every kernel is bound with, once per worker before capture."""

    device: torch.device
    plan_kind: PlanKind
    """The plan kind the target runs (the engine core's ``SpecResources``)."""
    max_num_reqs: int
    max_rows: int
    """Verify rows per request, the root included."""
    capture_rows: tuple[int, ...]
    """Rows per request of each decode CUDA graph."""
    kv_layouts: tuple[PagedKVLayout, ...] = ()
    """``layout:paged_token`` commit kernels only."""
    tree_parents: torch.Tensor | None = None
    """``[max_num_tokens]`` int32, static: during each forward, every batch
    token's parent row within its request's query (-1 for the first; chains
    and prefill tokens ``i - 1``), as ``CommonAttentionMetadata.tree_parents``.
    None when the plan kind is chain. For layers that verify trees themselves."""


@dataclass(frozen=True)
class CommitStep:
    """The accepted paths of one step, handed to every bound commit kernel."""

    result: "AcceptResult"
    num_reqs: int
    rows: torch.Tensor
    """``[R]`` int32: the verify rows of each request."""
    seq_lens: torch.Tensor
    """``[R]`` int32: each request's sequence length, its verify rows included."""


class SpecKernel:
    """A kernel for one key: implement :meth:`supports` and
    ``commit(step: CommitStep)``. It runs inside captured graphs: no host syncs
    and no allocations sized by data.
    """

    def supports(self, layer: SpecLayerSpec) -> bool:
        """Whether this kernel runs ``layer``, on chains and trees alike."""
        raise NotImplementedError

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        """Called once per worker before capture; allocate and raise here."""


class KernelRegistry:
    """The speculator's kernels, by key. Built once per worker and bound once,
    before capture."""

    def __init__(self, kernels: Mapping[KernelKey, SpecKernel] | None = None):
        # Nothing in tree: a layer verifies trees only with the speculator's kernels.
        self._kernels = dict(kernels or {})
        self._commits: list[SpecKernel] = []

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "KernelRegistry":
        """The registry of the speculator's ``spec_kernels``."""
        from vllm.utils.import_utils import resolve_obj_by_qualname

        speculative_config = vllm_config.speculative_config
        if speculative_config is None:
            return cls()
        name = speculative_config.speculator_config.spec_kernels
        return cls(resolve_obj_by_qualname(name)(vllm_config) if name else None)

    @staticmethod
    def required_keys(layer: SpecLayerSpec) -> list[KernelKey]:
        """The kernels ``layer`` needs to verify trees."""
        kind = layer.kind
        if kind.family == "recurrent":
            return [(SpecLayerKind("layout", "recurrent_state").key(), "commit")]
        if layer.layout:
            return [(SpecLayerKind("layout", layer.layout).key(), "commit")]
        return []

    def kernel_for(self, key: KernelKey, layer: SpecLayerSpec) -> SpecKernel | None:
        """The kernel that runs ``key`` on ``layer``, or None."""
        kernel = self._kernels.get(key)
        if kernel is None or not kernel.supports(layer):
            return None
        return kernel

    def plan_kind(self, layer: SpecLayerSpec) -> tuple[PlanKind, str]:
        """``("tree", "")`` when every kernel ``layer`` needs runs it, else
        ``"chain"`` and why."""
        for key in self.required_keys(layer):
            kernel = self._kernels.get(key)
            if kernel is None:
                return "chain", f"no {key_name(key)} kernel"
            if not kernel.supports(layer):
                return "chain", f"{type(kernel).__name__} declines the layer"
        return "tree", ""

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        """Bind each kernel once, to every layer it runs on. A chain binds
        none."""
        # One kernel may serve several keys: bind it once to its unique layers.
        served: dict[int, tuple[SpecKernel, dict[KernelKey, list[SpecLayerSpec]]]] = {}
        for layer in layers if ctx.plan_kind != "chain" else ():
            for key in self.required_keys(layer):
                kernel = self.kernel_for(key, layer)
                if kernel is None:
                    continue
                _, by_key = served.setdefault(id(kernel), (kernel, {}))
                by_key.setdefault(key, []).append(layer)
        self._commits = []
        for kernel, by_key in served.values():
            unique = {layer.name: layer for keyed in by_key.values() for layer in keyed}
            kernel_ctx = ctx
            if PAGED_KV_COMMIT in by_key:
                layouts = paged_kv_layouts(by_key[PAGED_KV_COMMIT])
                kernel_ctx = replace(ctx, kv_layouts=layouts)
            kernel.bind(kernel_ctx, list(unique.values()))
            self._commits.append(kernel)

    def commit(self, step: CommitStep) -> None:
        """Run every bound commit kernel on ``step``."""
        for kernel in self._commits:
            kernel.commit(step)  # type: ignore[attr-defined]
