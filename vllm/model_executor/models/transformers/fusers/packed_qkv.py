# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-QKV fuser: `c_attn(x).split((q, kv, kv))` -> a `QKVParallelLinear`."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.transformers.fusers.base import (
    RewriteFuser,
    local_output_sizes,
)
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    is_method,
    recover_forward,
    returned_linear,
    upstream_linear,
)
from vllm.model_executor.models.transformers.utils import (
    log_replacement,
    replace_linear_class,
)
from vllm.model_executor.models.utils import maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class PackedQKVFuser(RewriteFuser):
    """Fuser for attention with q, k and v packed into one projection."""

    qkv_name: str
    o_name: str | None
    q_size: int
    kv_size: int

    def info(self, name: str) -> str:
        return (
            f"Fused: {self.qkv_name} ({name}: {self.source_cls}) -> QKVParallelLinear"
        )

    @staticmethod
    def _packed_sizes(node: fx.Node) -> tuple[int, int] | None:
        """`(q, kv)` from a `split((q, kv, kv), ...)` call, if it is one."""
        if not is_method(node, "split") or len(node.args) < 2:
            return None
        sizes = node.args[1]
        if not isinstance(sizes, (tuple, list)) or len(sizes) != 3:
            return None
        if not all(isinstance(size, int) for size in sizes):
            return None
        q_size, k_size, v_size = sizes
        if k_size != v_size or q_size < k_size:
            return None
        return q_size, k_size

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "PackedQKVFuser | None":
        for node in graph.nodes:
            if (sizes := cls._packed_sizes(node)) is None:
                continue
            q_size, kv_size = sizes
            qkv_node = upstream_linear(node.args[0], module)
            if qkv_node is None:
                continue
            qkv_name = str(qkv_node.target)
            # The split must consume the whole projection.
            if module.get_submodule(qkv_name).out_features != q_size + 2 * kv_size:
                continue
            # o_proj produces the module's output and consumes the query width.
            o_name = returned_linear(graph, module)
            if o_name == qkv_name or (
                o_name is not None
                and module.get_submodule(o_name).in_features != q_size
            ):
                o_name = None
            return cls(
                source_cls=type(module).__name__,
                qkv_name=qkv_name,
                o_name=o_name,
                q_size=q_size,
                kv_size=kv_size,
            )
        return None

    def _split_call(self, funcdef: ast.FunctionDef) -> ast.Call:
        """The unique `self.<qkv_name>(...)....split((a, b, c), ...)` call."""
        calls = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "split"
            and node.args
            and isinstance(node.args[0], (ast.Tuple, ast.List))
            and len(node.args[0].elts) == 3
            and any(
                isinstance(inner, ast.Attribute) and inner.attr == self.qkv_name
                for inner in ast.walk(node.func.value)
            )
        ]
        if len(calls) != 1:
            raise ValueError(f"{self.qkv_name} has {len(calls)} three-way splits")
        return calls[0]

    def update_forward(self, module: nn.Module) -> None:
        """Rewrite the split sizes to the sharded projection's per-rank widths."""
        funcdef, fn = recover_forward(type(module))
        split = self._split_call(funcdef)
        # (q, kv, kv) -> [s // qkv.tp_size for s in qkv.output_sizes]
        sections = local_output_sizes(self.qkv_name)
        split.args[0] = ast.parse(sections, mode="eval").body
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Shapes must be compatible with a head-sharded packed GEMM."""
        head_size = vllm_config.model_config.get_head_size()
        qkv = module.get_submodule(self.qkv_name)
        compatible = (
            self.q_size % head_size == 0
            and self.kv_size % head_size == 0
            and qkv.out_features == self.q_size + 2 * self.kv_size
        )
        if not compatible:
            logger.debug("%s is not compatible with packed QKV fusion", type(module))
        return compatible

    def update_attrs(
        self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"
    ) -> None:
        quant_config = vllm_config.quant_config
        head_size = vllm_config.model_config.get_head_size()
        qkv_prefix = maybe_prefix(prefix, self.qkv_name)
        qkv = module.get_submodule(self.qkv_name)
        merged = QKVParallelLinear(
            hidden_size=qkv.in_features,
            head_size=head_size,
            total_num_heads=self.q_size // head_size,
            total_num_kv_heads=self.kv_size // head_size,
            bias=qkv.bias is not None,
            quant_config=quant_config,
            prefix=qkv_prefix,
            return_bias=False,
        )
        setattr(module, self.qkv_name, merged)
        log_replacement(qkv_prefix, qkv, merged)
        # If there is an output projection, we know it must be rowwise.
        if self.o_name is not None:
            o_prefix = maybe_prefix(prefix, self.o_name)
            o_proj = module.get_submodule(self.o_name)
            new_o = replace_linear_class(
                o_proj, "rowwise", quant_config, prefix=o_prefix
            )
            setattr(module, self.o_name, new_o)
            log_replacement(o_prefix, o_proj, new_o)
