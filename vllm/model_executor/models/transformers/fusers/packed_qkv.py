# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-QKV fuser: `c_attn(x).split((q, kv, kv))` -> a `QKVParallelLinear`.

Also matches a packed projection sliced into q, k and v along the last dim.
"""

import ast
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import fx, nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.models.transformers.fusers.base import (
    RewriteFuser,
    fused_head_size,
    local_output_sizes,
)
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    is_fn,
    is_linear,
    is_method,
    recover_forward,
    returned_linear,
    single_self_call,
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
    def _split_sizes(node: fx.Node) -> tuple[int, int] | None:
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

    @staticmethod
    def _sliced_sizes(node: fx.Node) -> tuple[int, int] | None:
        """`(q, kv)` if the output of `node` is only sliced into q, k and v.

        The three users must be `x[..., :q]`, `x[..., q:q + kv]` and
        `x[..., q + kv:]` on `node` itself, so slices of anything derived from
        it (e.g. rotary) never match.
        """
        stops: dict[object, object] = {}
        for user in node.users:
            index = user.args[1] if is_fn(user, operator.getitem) else None
            if not (
                isinstance(index, tuple)
                and len(index) == 2
                and index[0] is Ellipsis
                and isinstance(part := index[1], slice)
                and part.step is None
            ):
                return None
            stops[part.start] = part.stop
        q_size = stops.get(None)
        k_end = stops.get(q_size)
        if not (
            len(stops) == len(node.users) == 3
            and isinstance(q_size, int)
            and isinstance(k_end, int)
            and k_end in stops
            and stops[k_end] is None
        ):
            return None
        kv_size = k_end - q_size
        return (q_size, kv_size) if 0 < kv_size <= q_size else None

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "PackedQKVFuser | None":
        for node in graph.nodes:
            if (sizes := cls._split_sizes(node)) is not None:
                qkv_node = upstream_linear(node.args[0], module)
            elif (sizes := cls._sliced_sizes(node)) is not None:
                qkv_node = node if is_linear(node, module) else None
            else:
                continue
            if qkv_node is None:
                continue
            q_size, kv_size = sizes
            qkv_name = str(qkv_node.target)
            # q, k and v must consume the whole projection.
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

    def _rewrite_split(self, funcdef: ast.FunctionDef) -> bool:
        """Point the unique `self.<qkv_name>(...)....split((a, b, c), ...)` at the
        per-rank sections."""
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
            return False
        # (q, kv, kv) -> [s // qkv.tp_size for s in qkv.output_sizes]
        sections = local_output_sizes(self.qkv_name)
        calls[0].args[0] = ast.parse(sections, mode="eval").body
        return True

    def _rewrite_slices(self, funcdef: ast.FunctionDef) -> bool:
        """Point the three `qkv[..., a:b]` slices at the per-rank sections."""
        call = single_self_call(funcdef, self.qkv_name)
        targets = [
            node.targets[0] if isinstance(node, ast.Assign) else node.target
            for node in ast.walk(funcdef)
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is call
        ]
        target = targets[0] if len(targets) == 1 else None
        if not isinstance(target, ast.Name):
            return False
        slices = [
            part
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == target.id
            and isinstance(node.slice, ast.Tuple)
            and isinstance(part := node.slice.elts[-1], ast.Slice)
        ]
        # The fx match checked the bounds; q has no lower and v no upper bound
        positions = [
            0 if part.lower is None else 2 if part.upper is None else 1
            for part in slices
        ]
        if sorted(positions) != [0, 1, 2]:
            return False

        qkv = f"self.{self.qkv_name}"
        q = f"{qkv}.output_sizes[0] // {qkv}.tp_size"
        k_end = f"{q} + {qkv}.output_sizes[1] // {qkv}.tp_size"
        bounds = {0: (None, q), 1: (q, k_end), 2: (k_end, None)}
        for part, pos in zip(slices, positions):
            lower, upper = bounds[pos]
            part.lower = None if lower is None else ast.parse(lower, mode="eval").body
            part.upper = None if upper is None else ast.parse(upper, mode="eval").body
        return True

    def update_forward(self, module: nn.Module) -> None:
        """Rewrite the split sizes or slice bounds to the per-rank widths."""
        funcdef, fn = recover_forward(type(module))
        if not (self._rewrite_split(funcdef) or self._rewrite_slices(funcdef)):
            raise ValueError(f"{self.qkv_name} is not split or sliced into q, k and v")
        self.fused_forward = compile_forward(funcdef, fn)

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        """Shapes must be compatible with a head-sharded packed GEMM."""
        head_size = fused_head_size(module, vllm_config)
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
        head_size = fused_head_size(module, vllm_config)
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
