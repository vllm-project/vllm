# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused-QKV projection fuser: a single `c_attn(x).split(q, kv, kv)` linear.

Some architectures (e.g. GPTBigCode/StarCoder) pack Q, K and V into one linear
whose output is split inside the attention forward, rather than three sibling
`q`, `k`, `v` linears (handled by `QKVFuser`). A plain replicate/colwise shard
of that packed projection is wrong: colwise cuts through the K/V region, while
replicate leaves the query at full width even though the attention head count is
divided across ranks, which fails at `Attention.forward`'s head reshape.

This fuser replaces the packed linear with a `QKVParallelLinear` (Q sharded by
heads, K/V replicated for multi-query/GQA), rewrites the source split sizes to
the per-rank widths, and makes the output projection row-parallel.
"""

import ast
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.transformers.fusers.base import BaseFuser
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    recover_forward,
)
from vllm.model_executor.models.transformers.utils import (
    log_replacement,
    replace_linear_class,
)
from vllm.model_executor.models.utils import maybe_prefix

if TYPE_CHECKING:
    from vllm.config.model import ModelConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig

logger = init_logger(__name__)


@dataclass
class FusedQKVFuser(BaseFuser):
    """Fuser for a single packed QKV projection split as `q(x) -> q, k, v`."""

    source_cls: str
    qkv_name: str
    o_name: str | None
    q_size: int
    kv_size: int
    fused_forward: Callable = field(init=False, repr=False)

    def info(self, name: str) -> str:
        return (
            f"Sharded: {self.qkv_name} ({name}: {self.source_cls}) -> "
            f"{self.qkv_name} (QKVParallelLinear)"
        )

    @staticmethod
    def _linear_source(node: object, module: nn.Module) -> fx.Node | None:
        """Walk back through reshapes/casts to the `nn.Linear` producing `node`."""
        for _ in range(8):
            if not isinstance(node, fx.Node):
                return None
            if node.op == "call_module":
                child = module.get_submodule(node.target)
                return node if isinstance(child, nn.Linear) else None
            if node.args and isinstance(node.args[0], fx.Node):
                node = node.args[0]
            else:
                return None
        return None

    @classmethod
    def _find_fused_qkv(
        cls, graph: fx.Graph, module: nn.Module
    ) -> tuple[fx.Node, int, int] | None:
        """Find `linear(x)...split((q, kv, kv))` with `q >= kv` and `k == v`."""
        for node in graph.nodes:
            if node.op != "call_method" or node.target != "split":
                continue
            if len(node.args) < 2:
                continue
            sizes = node.args[1]
            if not (isinstance(sizes, (tuple, list)) and len(sizes) == 3):
                continue
            if not all(isinstance(s, int) for s in sizes):
                continue
            q_size, k_size, v_size = sizes
            if k_size != v_size or q_size < k_size:
                continue
            linear = cls._linear_source(node.args[0], module)
            if linear is None:
                continue
            return linear, q_size, k_size
        return None

    @staticmethod
    def _find_split_call(funcdef: ast.FunctionDef, qkv_name: str) -> ast.Call | None:
        """The unique `self.<qkv_name>(...)....split((a, b, c), ...)` call."""
        calls = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and node.args
            and isinstance(node.args[0], ast.Tuple)
            and len(node.args[0].elts) == 3
            and any(
                isinstance(inner, ast.Attribute) and inner.attr == qkv_name
                for inner in ast.walk(node.func.value)
            )
        ]
        return calls[0] if len(calls) == 1 else None

    def _rewrite_forward(self, module: nn.Module) -> None:
        """Rewrite the packed split sizes to the merged linear's per-rank widths."""
        funcdef, fn = recover_forward(type(module))
        split = self._find_split_call(funcdef, self.qkv_name)
        if split is None:
            raise ValueError("could not locate a unique fused qkv split")
        # (q, kv, kv) -> [s // qkv.tp_size for s in qkv.output_sizes]
        sizes = ast.parse(
            f"[__s // self.{self.qkv_name}.tp_size "
            f"for __s in self.{self.qkv_name}.output_sizes]",
            mode="eval",
        ).body
        split.args[0] = sizes
        ast.fix_missing_locations(funcdef)
        self.fused_forward = compile_forward(funcdef, fn)

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "FusedQKVFuser | None":
        found = cls._find_fused_qkv(graph, module)
        if found is None:
            return None
        qkv_node, q_size, kv_size = found
        qkv_name = qkv_node.target
        candidates = [
            name
            for name, child in module.named_children()
            if isinstance(child, nn.Linear)
            and name != qkv_name
            and child.in_features == q_size
        ]
        if len(candidates) != 1:
            logger.debug(
                "Skipping fused QKV fusion for %s: expected exactly one output "
                "projection with in_features=%d, found %d",
                type(module),
                q_size,
                len(candidates),
            )
            return None
        o_name = candidates[0]
        fuser = cls(
            source_cls=type(module).__name__,
            qkv_name=qkv_name,
            o_name=o_name,
            q_size=q_size,
            kv_size=kv_size,
        )
        try:
            fuser._rewrite_forward(module)
        except Exception as exc:
            logger.debug("Could not rewrite %s for fusion: %s", type(module), exc)
            return None
        return fuser

    def validate(self, module: nn.Module, model_config: "ModelConfig") -> bool:
        head_size = model_config.get_head_size()
        qkv = module.get_submodule(self.qkv_name)
        compatible = (
            self.q_size % head_size == 0
            and self.kv_size % head_size == 0
            and qkv.out_features == self.q_size + 2 * self.kv_size
        )
        if not compatible:
            logger.debug("%s is not compatible with fused QKV fusion", type(module))
        return compatible

    def fuse(
        self,
        module: nn.Module,
        prefix: str,
        model_config: "ModelConfig",
        quant_config: "QuantizationConfig",
    ) -> nn.Module:
        head_size = model_config.get_head_size()
        qkv = module.get_submodule(self.qkv_name)
        merged = QKVParallelLinear(
            hidden_size=qkv.in_features,
            head_size=head_size,
            total_num_heads=self.q_size // head_size,
            total_num_kv_heads=self.kv_size // head_size,
            bias=qkv.bias is not None,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, self.qkv_name),
            return_bias=False,
        )
        setattr(module, self.qkv_name, merged)
        log_replacement(maybe_prefix(prefix, self.qkv_name), qkv, merged)
        if self.o_name is not None:
            o_prefix = maybe_prefix(prefix, self.o_name)
            o_proj = module.get_submodule(self.o_name)
            new_o = replace_linear_class(
                o_proj, "rowwise", quant_config, prefix=o_prefix
            )
            setattr(module, self.o_name, new_o)
            log_replacement(o_prefix, o_proj, new_o)
        module.forward = types.MethodType(self.fused_forward, module)
        return module
