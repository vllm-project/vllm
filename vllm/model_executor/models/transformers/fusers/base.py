# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base classes for the Transformers backend fusers."""

import ast
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.models.transformers.fx_utils import (
    aliasing_names,
    bypass_existence_guard,
    replace_expr,
    self_call_and_refs,
    written_names,
)
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass
class BaseFuser(ABC):
    """A detected fusion and how to apply it.

    `match` analyses the module *class* once (cached, see `get_fusers`); `fuse`
    then applies the fusion to an instance in `recursive_replace`, returning the
    module to install in its place.
    """

    redefines_forward: ClassVar[bool] = True
    """Whether `fuse` gives the module a different forward,
    by rewriting its source or by returning a different module in its place."""

    @abstractmethod
    def info(self, name: str) -> str:
        """A human-readable description of the fusion at `name`, for logging."""

    @classmethod
    @abstractmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "BaseFuser | None":
        """Match the pattern in `graph`, returning a fuser if found."""

    @abstractmethod
    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Whether this fuser can be applied to this `module` instance."""

    @abstractmethod
    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        """Apply the fusion to an already-validated `module`, returning the
        module to install in its place (mutated in place, or freshly built)."""

    def orig_to_new_stacked(self, prefix: str) -> dict[str, tuple[str, ShardId]]:
        """`WeightsMapper.orig_to_new_stacked` entries this fuser contributes
        (none unless it stacks weights)."""
        return {}

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        """`packed_modules_mapping` entries this fuser contributes (none unless
        it stacks weights)."""
        return {}


def fused_head_size(module: nn.Module, vllm_config: "VllmConfig") -> int:
    """The head size of `module`, which head counts are derived from.

    Prefer the module's own `head_dim`, which Transformers sets per instance:
    the model-wide value is the largest head size across layers, so on a
    heterogeneous checkpoint it would divide out the wrong number of heads, and
    it is the text head size, so it does not describe a vision tower at all.
    """
    return getattr(module, "head_dim", None) or vllm_config.model_config.get_head_size()


def local_output_sizes(merged_name: str) -> str:
    """Source for the per-rank widths of the merged linear `self.<merged_name>`."""
    merged = f"self.{merged_name}"
    return f"[s // {merged}.tp_size for s in {merged}.output_sizes]"


@dataclass
class RewriteFuser(BaseFuser):
    """A fuser that rewrites the module's forward and rebinds it.

    `match` and `update_forward` analyse the class once; `fuse` swaps the
    submodules and binds the compiled forward on an instance in place, so it
    keeps its class and any attribute the fusion does not consume.
    """

    source_cls: str
    """Class of the HF module the fused projections belonged to (for logging)."""

    fused_forward: Callable = field(init=False, repr=False)
    """The compiled rewritten forward, set by `update_forward`."""

    @abstractmethod
    def update_forward(self, module: nn.Module) -> None:
        """Rewrite and compile `type(module)`'s forward source.

        Raises if the source does not admit the rewrite (fusion is then skipped).
        """

    @abstractmethod
    def update_attrs(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> None:
        """Replace `module`'s submodules with their vLLM equivalents."""

    def fuse(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> nn.Module:
        """Fuse an already-validated `module` in place (see `Fusers.__getitem__`).

        Builds the merged submodule and binds the compiled forward."""
        self.update_attrs(module, prefix, vllm_config)
        module.forward = types.MethodType(self.fused_forward, module)
        return module


@dataclass
class StackedFuser(RewriteFuser):
    """A fuser that merges sibling projections into one stacked linear and
    rewrites the forward to call it."""

    merged_name: ClassVar[str]
    """Attribute name of the merged module created by `update_attrs`."""
    merged_cls_name: ClassVar[str]
    """Name of the vLLM class the merged projection becomes (for logging)."""

    def info(self, name: str) -> str:
        sources = " + ".join(shard for shard, _ in self.shards)
        return (
            f"Fused: {sources} ({name}: {self.source_cls}) -> "
            f"{self.merged_name} ({self.merged_cls_name})"
        )

    @property
    @abstractmethod
    def shards(self) -> list[tuple[str, ShardId]]:
        """Each projection's original name and its shard id in the merged module.

        Source for both `orig_to_new_stacked` and `packed_modules_mapping`."""

    def orig_to_new_stacked(self, prefix: str) -> dict[str, tuple[str, ShardId]]:
        """`WeightsMapper.orig_to_new_stacked` entries for one fused instance.

        Maps each checkpoint name to `(merged_name, shard_id)`, keyed by qualname
        so only this exact layer is remapped, never a same-named projection
        elsewhere (e.g. an unfused MoE expert's `gate_proj`)."""
        merged = maybe_prefix(prefix, self.merged_name)
        return {
            maybe_prefix(prefix, name): (merged, shard) for name, shard in self.shards
        }

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        """`{merged_name: [projection names]}` so quantization can unpack the
        fused layer into its per-shard configs."""
        return {self.merged_name: [name for name, _ in self.shards]}

    def _unguarded_calls(
        self, funcdef: ast.FunctionDef, names: Iterable[str]
    ) -> list[ast.Call]:
        """One `self.<name>(arg)` call per projection, existence guards folded.

        `update_attrs` deletes the projections it merges, so any reference to one beyond
        its call site must be a guard on its existence; folding those to their constant
        value keeps the rewritten forward off a name that no longer exists. A reference
        that is not such a guard raises, so fusion is skipped."""
        calls = []
        for name in names:
            call, refs = self_call_and_refs(funcdef, name)
            for ref in refs:
                bypass_existence_guard(funcdef, ref, name)
            calls.append(call)
        return calls

    def _check_input_stable(
        self,
        funcdef: ast.FunctionDef,
        module: nn.Module,
        calls: list[ast.Call],
        block: list[ast.stmt],
        indices: list[int],
    ) -> None:
        """Raise unless hoisting the merged GEMM preserves what it reads.

        Fusing moves every projection to one call at `min(indices)`, so the
        merged GEMM reads the input once, up front, where the last of `calls`
        would have read it later. That holds only if nothing in between changes
        the input, and a change need not name it: writing any view that shares
        its storage changes it too. Both halves of the check over-approximate,
        since a false hit costs a fusion while a miss returns wrong numbers.
        """
        arg_names = {
            node.id
            for node in ast.walk(calls[0].args[0])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        # Protect the names the input reads and anything that may alias them.
        tracked = aliasing_names(funcdef, arg_names, module)
        # Any of them rebound, written through, or passed to a call that writes
        # its argument would leave the hoisted GEMM reading a different value.
        region = block[min(indices) : max(indices) + 1]
        if tracked & written_names(region):
            raise ValueError("projection input is rebound or mutated before all calls")

    def _splice_merged_split(
        self,
        funcdef: ast.FunctionDef,
        calls: list[ast.Call],
        block: list[ast.stmt],
        index: int,
    ) -> None:
        """Insert `temps = self.<merged_name>(arg).split(sizes, -1)` at
        `block[index]` and replace each of `calls` with its temp name.

        `calls` must share one input argument; `block[index]` must be where
        they are (or would be) evaluated. Raises if a generated temporary
        would shadow an existing name in `funcdef`.
        """
        temps = [f"_vllm_merged_{i}" for i in range(len(calls))]
        names = {node.id for node in ast.walk(funcdef) if isinstance(node, ast.Name)}
        if names & set(temps):
            raise ValueError("fused temporaries would shadow existing names")
        targets = ", ".join(temps)
        sections = local_output_sizes(self.merged_name)
        source = f"{targets} = self.{self.merged_name}(__arg__).split({sections}, -1)"
        assign = ast.parse(source).body[0]
        arg = next(
            node
            for node in ast.walk(assign)
            if isinstance(node, ast.Name) and node.id == "__arg__"
        )
        replace_expr(assign, arg, calls[0].args[0])
        ast.copy_location(assign, block[index])
        block.insert(index, assign)
        for call, temp in zip(calls, temps):
            replace_expr(funcdef, call, ast.Name(id=temp, ctx=ast.Load()))
