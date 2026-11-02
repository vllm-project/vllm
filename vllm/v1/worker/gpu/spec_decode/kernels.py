# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernels speculative decoding adds to verify and commit, by key.

A spec kernel serves one :data:`KernelKey`: a layer kind (``SpecLayerKind.key()``)
and an op.

- ``verify`` runs the convolution and recurrence of a recurrent layer's
  spec-decode rows (keys ``"mamba:<variant>@<state mode>"``). The layer calls it
  from its forward, so it runs inside the captured decode graphs.
- ``commit`` moves what the accepted path wrote to where the next step reads it
  (``"layout:paged_token"`` for attention KV, ``"layout:mamba_state@<state
  mode>"`` for recurrent state). The runner calls it after acceptance on a step
  whose plan is not a chain; a chain commits nothing.

:class:`KernelRegistry` is :class:`ReferenceKernels` overlaid by the providers
the speculator names in ``SpeculatorConfig.spec_kernel_providers``, in order: a
later provider wins a key. The registry also decides what the target verifies:
a layer verifies the highest plan kind that every kernel it needs
(:meth:`KernelRegistry.required_keys`) supports for it, and a chain when one is
missing. So a provider may supply any subset of keys; a key it leaves out keeps
the reference kernel, or lowers the plan kind when there is none, and a missing
kernel is never called.

A provider, for example of a faster GDN tree kernel::

    class MyGDNVerify(SpecKernel):
        def supports(self, layer):
            return frozenset({"chain", "tree"})

        def verify(self, rows: GDNSpecRows) -> None: ...


    class MyProvider:
        name = "my_provider"

        def __init__(self, vllm_config): ...

        def kernels(self):
            return {("mamba:gdn@slots", "verify"): MyGDNVerify()}


    class MySpeculatorConfig(SpeculatorConfig):
        spec_kernel_providers = ("my_package.MyProvider",)
"""

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.capabilities import (
    PLAN_KINDS,
    PlanKind,
    SpecLayerKind,
    plan_kind_rank,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.spec_decode.round import AcceptResult, PlanShape

logger = init_logger(__name__)

Op = Literal["verify", "commit"]
KernelKey = tuple[str, Op]
"""``(SpecLayerKind.key(), op)``, e.g. ``("mamba:gdn@slots", "verify")`` or
``("layout:paged_token", "commit")``."""


def key_name(key: KernelKey) -> str:
    """``"mamba:gdn@slots/verify"``: a key as logs and docs spell it."""
    return f"{key[0]}/{key[1]}"


# --- what a kernel is handed -------------------------------------------------


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
    max_candidates: int
    """Verify rows per request, the root included."""
    capture_shapes: "tuple[PlanShape, ...]"
    """The shapes the decode CUDA graphs are captured at."""


@dataclass(frozen=True)
class CommitStep:
    """The accepted paths of one step, handed to every bound commit kernel."""

    result: "AcceptResult"
    num_reqs: int
    count: torch.Tensor
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


@dataclass(frozen=True)
class Mamba2SpecRows:
    """The decode rows of one Mamba2 layer's forward: its spec-decode rows and
    its single-token decodes, which are one-row chains.

    ``T`` is the number of decode tokens and ``B`` the number of their requests
    (a CUDA graph batch may pad ``B``; the in-tree kernels see the same
    padding). Heads and groups are per tensor-parallel rank.
    """

    layer_name: str
    """The layer's prefix, its key in the forward context."""
    hidden_states_B_C: torch.Tensor
    """``[T, conv_dim]`` convolution input (the x, B and C columns). May be a
    view whose row stride is larger than ``conv_dim``."""
    dt: torch.Tensor
    """``[T, num_heads]`` time step before ``dt_bias`` and softplus. May be a
    strided view."""
    conv_state: torch.Tensor
    """``[num_slots, conv_dim, kernel_size - 1 + num_spec]`` convolution
    states."""
    conv_weight: torch.Tensor
    """``[conv_dim, kernel_size]``."""
    conv_bias: torch.Tensor | None
    activation: str | None
    ssm_state: torch.Tensor
    """``[num_slots, num_heads, head_dim, ssm_state_size]`` SSM states."""
    A: torch.Tensor
    """``[num_heads]`` float32 (``-exp(A_log)``)."""
    D: torch.Tensor
    """``[num_heads]``."""
    dt_bias: torch.Tensor
    """``[num_heads]``; the in-tree kernels use ``softplus(dt + dt_bias)``."""
    num_heads: int
    head_dim: int
    n_groups: int
    ssm_state_size: int
    state_indices: torch.Tensor
    """``[B, max_query_len]``: the state slot of each row of each request."""
    num_accepted_tokens: torch.Tensor
    """``[B]``: request ``b`` starts from ``state_indices[b, n - 1]``."""
    query_start_loc: torch.Tensor
    """``[B + 1]``: the rows of each request in ``hidden_states_B_C``."""
    parents: torch.Tensor | None
    """``[T]`` int32: each row's parent row within its request (-1 for the
    root), or None for chains."""
    out: torch.Tensor
    """``[T, num_heads * head_dim]`` SSM output (before the gated norm),
    written by the kernel."""


# --- kernels and providers ---------------------------------------------------


class SpecKernel:
    """A kernel for one key. Subclass it, implement :meth:`supports` and the
    key's op:

    - ``verify(rows) -> None`` (``mamba:*`` keys): ``rows`` is the layer's
      :class:`GDNSpecRows` or :class:`Mamba2SpecRows`. Write every row's output
      to ``rows.out`` and keep the in-tree state layout: row ``i`` of request
      ``b`` stores its convolution and recurrent state at ``state_indices[b, i]``
      and continues from row ``parents[i]`` (row ``i - 1`` when ``parents`` is
      None); the root continues from ``state_indices[b, num_accepted_tokens[b] -
      1]``. Called by every forward with spec-decode rows, CUDA graph captures
      included: no host syncs and no allocations sized by data.
    - ``commit(step: CommitStep) -> None`` (``layout:*`` keys): move what the
      rows on ``step.result.path`` wrote to where the next step reads it.
    """

    def supports(self, layer: SpecLayerSpec) -> frozenset[PlanKind]:
        """The plan kinds this kernel runs for ``layer``; empty declines the
        layer (e.g. on its state dtype, head size or conv width), which then
        verifies chains. Called before :meth:`bind`."""
        raise NotImplementedError

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        """Called once per worker, before capture, with every layer the kernel
        runs on. Allocate here, and raise here on what it cannot run, not at
        the first step."""


class SpecKernelProvider(Protocol):
    """Kernels a speculator brings. ``SpeculatorConfig.spec_kernel_providers``
    names it by the fully qualified name of a ``(vllm_config) ->
    SpecKernelProvider`` factory (a class works)."""

    name: str

    def kernels(self) -> Mapping[KernelKey, SpecKernel]:
        """Any subset of keys; called once per worker."""
        ...


ABOVE_NONE: frozenset[PlanKind] = frozenset(PLAN_KINDS[1:])
"""Every plan kind with verify rows: a commit reads only the accepted path."""


class InTreeVerify(SpecKernel):
    """The layer's own spec-row kernels (the convolution and recurrence of
    ``variant`` with tree parents, a state slot per draft row). The layer runs
    them when no other verify kernel is bound to it, so ``verify`` is never
    called."""

    def __init__(self, variant: str):
        self.variant = variant

    def supports(self, layer: SpecLayerSpec) -> frozenset[PlanKind]:
        return frozenset({"chain", "tree"})

    def verify(self, rows: Any) -> None:
        raise AssertionError(f"the {self.variant} layer runs its own kernels")


class ReferenceKernels:
    """The in-tree kernels, which run every key no provider overrides.

    There is no ``mamba:short_conv@slots/verify``: the short-conv layer's
    convolution takes no tree parents, so it verifies chains. Nor is there a
    ``layout:paged_split_scale/commit`` yet.
    """

    name = "reference"

    def kernels(self) -> dict[KernelKey, SpecKernel]:
        from vllm.v1.worker.gpu.spec_decode.commit import (
            ConvWindowAligner,
            KvCompactor,
        )

        return {
            ("mamba:gdn@slots", "verify"): InTreeVerify("gdn"),
            ("mamba:mamba2@slots", "verify"): InTreeVerify("mamba2"),
            ("layout:paged_token", "commit"): KvCompactor(),
            ("layout:mamba_state@slots", "commit"): ConvWindowAligner(),
        }


# --- the registry ------------------------------------------------------------


class KernelRegistry:
    """:class:`ReferenceKernels` overlaid by ``providers`` in order: a later
    provider wins a key, and each override is logged once. Built once per
    worker and bound once, before capture."""

    def __init__(self, providers: Iterable[SpecKernelProvider] = ()):
        self._kernels: dict[KernelKey, SpecKernel] = ReferenceKernels().kernels()
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
            else:
                kind = max(kernel.supports(layer), key=plan_kind_rank)
                found = (kind, f"{type(kernel).__name__} verifies {kind}")
            if plan_kind_rank(found[0]) < plan_kind_rank(best[0]):
                best = found
        if plan_kind_rank(best[0]) < plan_kind_rank("chain"):
            best = ("chain", best[1])
        return best

    def bind(self, ctx: KernelContext, layers: Sequence[SpecLayerSpec]) -> None:
        """Bind each kernel once, to every layer it runs on. A recurrent layer
        gets its verify kernel as ``spec_verify_kernel`` (None for its own
        kernels), which it runs on chains too. Commit kernels are bound only
        for plans above a chain: a chain commits nothing."""
        commits = plan_kind_rank(ctx.plan_kind) > plan_kind_rank("chain")
        served: dict[int, tuple[KernelKey, SpecKernel, list[SpecLayerSpec]]] = {}
        for layer in layers:
            if layer.kind.family == "mamba":
                layer.module.spec_verify_kernel = None
            if ctx.plan_kind == "none":
                continue
            for key in self.required_keys(layer):
                if key[1] == "commit" and not commits:
                    continue
                kernel = self.kernel_for(key, layer)
                if kernel is None:
                    continue
                served.setdefault(id(kernel), (key, kernel, []))[2].append(layer)
                if key[1] == "verify" and not isinstance(kernel, InTreeVerify):
                    layer.module.spec_verify_kernel = kernel
        self._commits = []
        for key, kernel, served_layers in served.values():
            kernel.bind(ctx, served_layers)
            if key[1] == "commit":
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
        num_reqs, rows = ctx.max_num_reqs, ctx.max_candidates
        path = torch.arange(rows, dtype=torch.int32, device=ctx.device)
        path = path.expand(num_reqs, rows).contiguous()
        for shape in ctx.capture_shapes:
            count = torch.full(
                (num_reqs,),
                shape.uniform_count or rows,
                dtype=torch.int32,
                device=ctx.device,
            )
            step = CommitStep(
                result=AcceptResult(
                    path=path, path_len=count, sampled=torch.zeros_like(path)
                ),
                num_reqs=num_reqs,
                count=count,
                seq_lens=count,
            )
            run(functools.partial(self.commit, step))
