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

Op = Literal["verify", "commit"]
KernelKey = tuple[str, Op]
"""``(SpecLayerKind.key(), op)``, e.g. ``("mamba:gdn@slots", "verify")`` or
``("layout:paged_token", "commit")``."""


def key_name(key: KernelKey) -> str:
    """``"mamba:gdn@slots/verify"``: a key as logs and docs spell it."""
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
    """What committing the layer's accepted path moves (``SpecLayerKind`` of
    family ``layout``): ``"paged_token"``, ``"paged_split_scale"`` or
    ``"mamba_state"``; ``""`` for a cache not paged by slot, whose model keeps
    its own commit."""
    block_table: torch.Tensor | None = None
    """At bind: the block table of the layer's KV cache group, as the forward
    reads it."""
    block_size: int = 0
    """At bind: tokens per block of that group (a hybrid group's blocks can be
    larger than the layer's own spec says)."""
    num_blocks: int = 0
    """At bind: blocks of the backend's kernel block size in the layer's cache."""


PAGED_KV_COMMIT: KernelKey = ("layout:paged_token", "commit")


@dataclass(frozen=True)
class PagedKVLayout:
    """Where the KV caches of one KV cache group keep each token, for a
    ``layout:paged_token`` commit kernel. Element ``(block, slot, plane, head,
    dim)`` of a cache sits ``sum(index * stride)`` elements past the cache's
    first element; ``dim`` is contiguous. The caches vLLM binds keep K and V
    packed in one row (``[K | V]`` content, or MLA's latent), so ``num_planes``
    is 1 and ``head_dim`` spans the packed row.

    Token ``t`` of a request (by its position) lives in kernel block
    ``block_table[b, t // table_block_size] * table_block_size + t %
    table_block_size``, split by ``block_size`` into block and slot.
    """

    caches: tuple[torch.Tensor, ...]
    """One per distinct cache: layers sharing a cache (KV sharing) appear once."""
    shape: tuple[int, int, int, int, int]
    """``(num_blocks, block_size, num_planes, num_heads, head_dim)``, per cache."""
    strides: tuple[int, int, int, int, int]
    """In elements, in the order of ``shape``; ``strides[4] == 1``."""
    dtype: torch.dtype
    block_table: torch.Tensor
    """``[max_num_reqs, max_blocks]`` int32: the group's block table, as the
    forward reads it."""
    table_block_size: int
    """Tokens per block of ``block_table`` (the group's block size)."""

    @property
    def block_size(self) -> int:
        """Tokens per block of the caches: the backend's kernel block size,
        which may divide ``table_block_size``."""
        return self.shape[1]


def paged_kv_layout_of(
    cache: torch.Tensor, num_blocks: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(shape, strides)`` of a layer's KV cache view in
    :class:`PagedKVLayout` order. Reads ``[blocks, heads, block_size, content]``
    (attention, K and V packed in ``content``) and ``[blocks, block_size,
    content]`` (MLA) views; raises ``ValueError`` for anything else."""
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
    """The :class:`PagedKVLayout` of each KV cache group of ``layers``. Raises
    ``ValueError`` at bind for a cache it cannot describe."""
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
    """For the ``layout:paged_token`` commit kernel: the layouts of the layers
    it commits. Empty for every other kernel."""


@dataclass(frozen=True)
class CommitStep:
    """The accepted paths of one step, handed to every bound commit kernel."""

    result: "AcceptResult"
    num_reqs: int
    rows: torch.Tensor
    """``[R]`` int32: the verify rows of each request."""
    seq_lens: torch.Tensor
    """``[R]`` int32: each request's sequence length, its verify rows included."""


@dataclass(frozen=True)
class GDNSpecRows:
    """The spec-decode rows of one GDN layer's forward.

    ``T`` is the number of spec-decode tokens and ``B`` the number of their
    requests (a CUDA graph batch may pad ``B``; the in-tree kernels see the
    same padding). Heads and dims are per tensor-parallel rank.
    """

    layer_name: str
    """The layer's prefix, its key in the forward context."""
    mixed_qkv: torch.Tensor
    """``[T, conv_dim]`` convolution input (the q, k and v columns). May be a
    view whose row stride is larger than ``conv_dim``."""
    a: torch.Tensor
    """``[T, num_v_heads]`` decay gate input. May be a strided view."""
    b: torch.Tensor
    """``[T, num_v_heads]`` beta gate input. May be a strided view."""
    conv_state: torch.Tensor
    """``[num_slots, conv_dim, kernel_size - 1]`` convolution states."""
    conv_weight: torch.Tensor
    """``[conv_dim, kernel_size]``."""
    conv_bias: torch.Tensor | None
    activation: str | None
    ssm_state: torch.Tensor
    """``[num_slots, num_v_heads, head_v_dim, head_k_dim]`` recurrent states."""
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    state_indices: torch.Tensor
    """``[B, max_query_len]``: the state slot of each row of each request."""
    num_accepted_tokens: torch.Tensor
    """``[B]``: request ``b`` starts from ``state_indices[b, n - 1]``."""
    query_start_loc: torch.Tensor
    """``[B + 1]``: the rows of each request in ``mixed_qkv``."""
    parents: torch.Tensor | None
    """``[T]`` int32: each row's parent row within its request (-1 for the
    root), or None for chains."""
    out: torch.Tensor
    """``[T, num_v_heads, head_v_dim]``, written by the kernel."""


class SpecKernel:
    """A kernel for one key: implement :meth:`supports` and the key's op,
    ``verify(rows: GDNSpecRows)`` for ``mamba:gdn@*`` keys or
    ``commit(step: CommitStep)`` for ``layout:*`` keys. A verify kernel keeps the
    in-tree state layout and runs inside captured graphs: no host syncs and no
    allocations sized by data.
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

    def get(self, key: KernelKey) -> SpecKernel | None:
        """The kernel that won ``key``, None when nothing provides it."""
        return self._kernels.get(key)

    @staticmethod
    def required_keys(layer: SpecLayerSpec) -> list[KernelKey]:
        """The kernels ``layer`` needs to verify more than a chain: a recurrent
        layer its ``verify`` and its state's ``commit``, an attention layer its
        KV ``commit`` (none for a cache not paged by slot)."""
        kind = layer.kind
        if kind.family == "mamba":
            state = SpecLayerKind("layout", "mamba_state", kind.state_mode)
            return [(kind.key(), "verify"), (state.key(), "commit")]
        if layer.layout:
            return [(SpecLayerKind("layout", layer.layout).key(), "commit")]
        return []

    def kernel_for(self, key: KernelKey, layer: SpecLayerSpec) -> SpecKernel | None:
        """The kernel that runs ``key`` on ``layer``; None when nothing
        provides the key or its kernel declines the layer."""
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
        """Bind each kernel once, to every layer it runs on. A recurrent layer
        gets its verify kernel as ``spec_verify_kernel`` (None for its own
        kernels), which it runs on chains too. Commit kernels are bound only
        for plans above a chain: a chain commits nothing."""
        commits = ctx.plan_kind != "chain"
        # One kernel may serve several keys: bind it once to its unique layers
        # and commit with it if any of its ops is a commit.
        served: dict[int, tuple[SpecKernel, dict[KernelKey, list[SpecLayerSpec]]]] = {}
        for layer in layers:
            if layer.kind.family == "mamba":
                layer.module.spec_verify_kernel = None
            for key in self.required_keys(layer):
                if key[1] == "commit" and not commits:
                    continue
                kernel = self.kernel_for(key, layer)
                if kernel is None:
                    continue
                _, by_key = served.setdefault(id(kernel), (kernel, {}))
                by_key.setdefault(key, []).append(layer)
                if key[1] == "verify":
                    layer.module.spec_verify_kernel = kernel
        self._commits = []
        for kernel, by_key in served.values():
            unique = {layer.name: layer for keyed in by_key.values() for layer in keyed}
            kernel_ctx = ctx
            if PAGED_KV_COMMIT in by_key:
                layouts = paged_kv_layouts(by_key[PAGED_KV_COMMIT])
                kernel_ctx = replace(ctx, kv_layouts=layouts)
            kernel.bind(kernel_ctx, list(unique.values()))
            if any(op == "commit" for _, op in by_key):
                self._commits.append(kernel)

    def commit(self, step: CommitStep) -> None:
        """Run every bound commit kernel on ``step``."""
        for kernel in self._commits:
            kernel.commit(step)  # type: ignore[attr-defined]
