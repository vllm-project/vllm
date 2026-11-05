# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernels speculative decoding adds to verify and commit, by key, and the
registry that binds them (see ``docs/features/speculative_decoding/round_contract.md``).

A provider named in ``SpeculatorConfig.spec_kernel_providers`` may supply any subset
of keys; a later provider wins a key, and a layer missing a kernel verifies chains.
"""

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.resources import PLAN_KINDS, PlanKind, plan_kind_rank
from vllm.v1.worker.gpu.spec_decode.capabilities import SpecLayerKind

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.spec_decode.round import AcceptResult

logger = init_logger(__name__)

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

    def supports(self, layer: SpecLayerSpec) -> frozenset[PlanKind]:
        """The plan kinds this kernel runs for ``layer``; empty declines it.
        A verify kernel without ``"chain"`` is not used."""
        raise NotImplementedError

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        """Called once per worker before capture; allocate and raise here."""


class SpecKernelProvider(Protocol):
    """Kernels a speculator brings. ``SpeculatorConfig.spec_kernel_providers``
    names it by the fully qualified name of a ``(vllm_config) ->
    SpecKernelProvider`` factory (a class works)."""

    name: str

    def kernels(self) -> Mapping[KernelKey, SpecKernel]:
        """Any subset of keys; called once per worker."""
        ...


class KernelRegistry:
    """The in-tree kernels overlaid by ``providers`` in order: a later
    provider wins a key, and each override is logged once. Built once per
    worker and bound once, before capture."""

    def __init__(self, providers: Iterable[SpecKernelProvider] = ()):
        from vllm.v1.worker.gpu.spec_decode.commit import KvCompactor

        # In tree: the attention KV commit only. Recurrent layers verify trees
        # only with a provider's verify and state commit kernels.
        self._kernels: dict[KernelKey, SpecKernel] = {
            ("layout:paged_token", "commit"): KvCompactor()
        }
        for provider in providers:
            for key, kernel in provider.kernels().items():
                if key[1] not in get_args(Op):
                    raise ValueError(
                        f"{provider.name} provides {key_name(key)}: the op must "
                        f"be one of {get_args(Op)}"
                    )
                logger.info_once(
                    "Speculative decoding: %s %s %s",
                    provider.name,
                    "overrides" if key in self._kernels else "provides",
                    key_name(key),
                )
                self._kernels[key] = kernel
        self._commits: list[SpecKernel] = []

    @classmethod
    def from_config(cls, vllm_config: "VllmConfig") -> "KernelRegistry":
        """The registry of the speculator's ``spec_kernel_providers``."""
        from vllm.utils.import_utils import resolve_obj_by_qualname

        speculative_config = vllm_config.speculative_config
        names = (
            speculative_config.speculator_config.spec_kernel_providers
            if speculative_config is not None
            else ()
        )
        return cls(resolve_obj_by_qualname(name)(vllm_config) for name in names)

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
        """The highest plan kind that every kernel ``layer`` needs supports for
        it, and why it is not higher. A layer verifies chains without any."""
        best: tuple[PlanKind, str] = (PLAN_KINDS[-1], "")
        for key in self.required_keys(layer):
            kernel = self._kernels.get(key)
            if kernel is None:
                found: tuple[PlanKind, str] = ("chain", f"no {key_name(key)} kernel")
            elif not kernel.supports(layer):
                found = ("chain", f"{type(kernel).__name__} declines the layer")
            elif key[1] == "verify" and "chain" not in kernel.supports(layer):
                # The layer runs its verify kernel on chain batches too.
                found = ("chain", f"{type(kernel).__name__} does not verify chains")
            else:
                kind = max(kernel.supports(layer), key=plan_kind_rank)
                found = (kind, f"{type(kernel).__name__} verifies {kind}")
            if plan_kind_rank(found[0]) < plan_kind_rank(best[0]):
                best = found
        return best

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        """Bind each kernel once, to every layer it runs on. A recurrent layer
        gets its verify kernel as ``spec_verify_kernel`` (None for its own
        kernels), which it runs on chains too. Commit kernels are bound only
        for plans above a chain: a chain commits nothing."""
        commits = plan_kind_rank(ctx.plan_kind) > plan_kind_rank("chain")
        # One kernel may serve several keys: bind it once to its unique layers
        # and commit with it if any of its ops is a commit.
        served: dict[int, tuple[SpecKernel, set[Op], dict[str, SpecLayerSpec]]] = {}
        for layer in layers:
            if layer.kind.family == "mamba":
                layer.module.spec_verify_kernel = None
            for key in self.required_keys(layer):
                if key[1] == "commit" and not commits:
                    continue
                kernel = self.kernel_for(key, layer)
                if kernel is None or ctx.plan_kind not in kernel.supports(layer):
                    continue
                _, ops, served_layers = served.setdefault(
                    id(kernel), (kernel, set(), {})
                )
                ops.add(key[1])
                served_layers[layer.name] = layer
                if key[1] == "verify":
                    layer.module.spec_verify_kernel = kernel
        self._commits = []
        for kernel, ops, served_layers in served.values():
            kernel.bind(ctx, list(served_layers.values()))
            if "commit" in ops:
                self._commits.append(kernel)

    def commit(self, step: CommitStep) -> None:
        """Run every bound commit kernel on ``step``."""
        for kernel in self._commits:
            kernel.commit(step)  # type: ignore[attr-defined]

    def capture(
        self,
        ctx: KernelContext,
        run: Callable[[Callable[[], object]], object] = lambda fn: fn(),
    ) -> None:
        """Run every bound commit kernel once per capture shape, each through
        ``run``, on identity paths (each row stays where it is), so their
        compilation and launch failures surface at startup, not in serving.
        Verify kernels run in the layers' forward when the decode graphs are
        captured."""
        from vllm.v1.worker.gpu.spec_decode.round import AcceptResult

        if not self._commits:
            return
        num_reqs, rows = ctx.max_num_reqs, ctx.max_rows
        path = torch.arange(rows, dtype=torch.int32, device=ctx.device)
        path = path.expand(num_reqs, rows).contiguous()
        for n in ctx.capture_rows:
            rows = torch.full((num_reqs,), n, dtype=torch.int32, device=ctx.device)
            step = CommitStep(
                result=AcceptResult(
                    path=path, path_len=rows, sampled=torch.zeros_like(path)
                ),
                num_reqs=num_reqs,
                rows=rows,
                seq_lens=rows,
            )
            run(functools.partial(self.commit, step))
