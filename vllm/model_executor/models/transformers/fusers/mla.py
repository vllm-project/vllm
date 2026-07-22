# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA fuser: adapt a Transformers MLA attention module for vLLM's MLA layer."""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models.transformers.fusers.base import StackedFuser
from vllm.model_executor.models.transformers.fusers.rms_norm import RMSNormFuser
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    is_leaf_call,
    is_linear,
    output_value,
    recover_forward,
    replace_expr,
    single_self_call,
    trace,
)
from vllm.model_executor.models.transformers.utils import replace_linear_class
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Temporaries the fused down-projection binds in the rewritten forward.
_Q_A_TEMP = "__q_a_fused"
_KV_A_TEMP = "__kv_a_fused"


def _consumes_placeholder(node: fx.Node) -> bool:
    """Whether `node` is a linear applied directly to a `forward` input."""
    return (
        len(node.args) == 1
        and isinstance(node.args[0], fx.Node)
        and node.args[0].op == "placeholder"
    )


def _upstream_linear(node: object, module: nn.Module) -> fx.Node | None:
    """Nearest linear producing `node`, walking back through splits/reshapes."""
    stack = [node]
    seen: set[fx.Node] = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, fx.Node) or current in seen:
            continue
        seen.add(current)
        if is_linear(current, module):
            return current
        if current.op in ("call_function", "call_method"):
            stack.extend(current.args)
    return None


def _downstream_linear(node: fx.Node, module: nn.Module) -> fx.Node | None:
    """Nearest linear consuming `node`'s output, walking through casts/scalings."""
    queue = list(node.users)
    seen: set[fx.Node] = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if is_linear(current, module):
            return current
        # Skip leaf calls created by _as_leaf_call (e.g. attention interfaces).
        if current.op in ("call_function", "call_method") and not is_leaf_call(current):
            queue.extend(current.users)
    return None


def _norm_size(norm: nn.Module) -> int:
    weight = getattr(norm, "weight", None)
    return weight.numel() if weight is not None else -1


def _is_rms_norm(module: nn.Module) -> bool:
    """Whether `module` computes an RMSNorm, verified by `RMSNormFuser`'s matcher."""
    graph = trace(module)
    return graph is not None and RMSNormFuser.match(graph, module) is not None


def _returned_linear(graph: fx.Graph, module: nn.Module) -> str | None:
    """Name of the Linear producing the graph's (first) output value."""
    value = output_value(graph)
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    linear = _upstream_linear(value, module)
    return None if linear is None else str(linear.target)


def _top_level_index(funcdef: ast.FunctionDef, node: ast.AST) -> int:
    """Index in `funcdef.body` of the top-level statement containing `node`."""
    for index, stmt in enumerate(funcdef.body):
        if any(child is node for child in ast.walk(stmt)):
            return index
    raise ValueError("node is not in the function body")


def _single_expand_call(
    funcdef: ast.FunctionDef, module: nn.Module, kv_b_proj_name: str
) -> ast.Call:
    """The KV expansion call `self.<method>(kv_c_normed, k_pe)`: the unique
    two-argument call to a method of `module` with the expansion's signature.

    The expansion is `kv_c_normed, k_pe -> key, value`: two inputs, every
    `return` a pair, and it applies `kv_b_proj` -- the projection `MLAAttention`
    owns after the bypass, absorbed into the attention computation.
    """

    def is_expansion_method(name: str) -> bool:
        method = getattr(type(module), name, None)
        if method is None:
            return False
        code = getattr(inspect.unwrap(method), "__code__", None)
        if (
            code is None
            or code.co_argcount != 3  # self, kv_c_normed, k_pe
            or kv_b_proj_name not in code.co_names
        ):
            return False
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        except (OSError, TypeError):
            return False
        returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
        return bool(returns) and all(
            isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 2
            for ret in returns
        )

    calls = [
        node
        for node in ast.walk(funcdef)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and len(node.args) == 2
        and not node.keywords
        and is_expansion_method(node.func.attr)
    ]
    if len(calls) != 1:
        raise ValueError("expected exactly one KV expansion method call")
    return calls[0]


@dataclass
class MLAFuser(StackedFuser):
    """Fuser for the MLA attention pattern."""

    q_proj_name: str | None
    q_a_proj_name: str | None
    q_a_layernorm_name: str | None
    q_b_proj_name: str | None
    kv_a_proj_name: str
    kv_a_layernorm_name: str
    kv_b_proj_name: str
    o_proj_name: str | None
    merged_name: ClassVar[str] = "fused_qkv_a_proj"
    merged_cls: ClassVar[str] = "MergedColumnParallelLinear"

    @property
    def has_q_lora(self) -> bool:
        return self.q_a_proj_name is not None

    def info(self, name: str) -> str:
        info_str = (
            f"Fused: {name} ({self.source_cls}) -> MLAAttention (attention interface)"
        )
        if self.has_q_lora:
            info_str += "; " + super().info(name).removeprefix("Fused: ")
        return info_str

    @property
    def shards(self) -> list[tuple[str, ShardId]]:
        """`q_a_proj` and `kv_a_proj_with_mqa` stack into one down-projection."""
        if self.has_q_lora:
            return [(self.q_a_proj_name, 0), (self.kv_a_proj_name, 1)]
        return []

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        if self.has_q_lora:
            return super().packed_modules_mapping
        return {}

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "MLAFuser | None":
        # Find all `rms_norm(linear(placeholder))` chains.
        chains = []
        for node in graph.nodes:
            if node.op != "call_module" or is_linear(node, module) or not node.args:
                continue
            source = _upstream_linear(node.args[0], module)
            if source is None or not _consumes_placeholder(source):
                continue
            if not _is_rms_norm(module.get_submodule(node.target)):
                continue
            chains.append((source, node))

        # Tell the chains apart by width.
        def is_kv_a(chain) -> bool:
            source, rms_norm = chain
            source_mod = module.get_submodule(source.target)
            rms_norm_mod = module.get_submodule(rms_norm.target)
            return source_mod.out_features != _norm_size(rms_norm_mod)

        kv_a_chains = [chain for chain in chains if is_kv_a(chain)]
        q_a_chains = [chain for chain in chains if not is_kv_a(chain)]
        # Exactly one KV chain is MLA's signature. The query chain is optional.
        if len(kv_a_chains) != 1 or len(q_a_chains) > 1:
            return None
        kv_a_proj, kv_a_layernorm = kv_a_chains[0]

        # Linear children claimed for a role so far; the rest resolve by elimination.
        claimed_linears = {kv_a_proj.target}
        q_proj_name = q_a_proj = q_a_layernorm = q_b_proj = None
        if q_a_chains:
            # Find `q_b_proj(q_a_layernorm(q_a_proj(placeholder)))`.
            q_a_proj, q_a_layernorm = q_a_chains[0]
            q_b_proj = _downstream_linear(q_a_layernorm, module)
            if q_b_proj is None:
                return None
            claimed_linears |= {q_a_proj.target, q_b_proj.target}
        else:
            # Find `q_proj(placeholder)`.
            placeholder_linears = {
                node.target
                for node in graph.nodes
                if is_linear(node, module) and _consumes_placeholder(node)
            }
            if len(q_proj_candidates := placeholder_linears - claimed_linears) != 1:
                return None
            q_proj_name = next(iter(q_proj_candidates))
            claimed_linears.add(q_proj_name)

        # Find `kv_b_proj(kv_a_layernorm(...))`.
        kv_b_proj = _downstream_linear(kv_a_layernorm, module)
        if kv_b_proj is None or kv_b_proj.target in claimed_linears:
            return None
        claimed_linears.add(kv_b_proj.target)

        # Find `o_proj` if it is returned by the forward graph.
        o_proj_name = _returned_linear(graph, module)
        if o_proj_name in claimed_linears:
            o_proj_name = None

        return cls(
            source_cls=type(module).__name__,
            q_proj_name=q_proj_name,
            q_a_proj_name=q_a_proj.target if q_a_proj else None,
            q_a_layernorm_name=q_a_layernorm.target if q_a_layernorm else None,
            q_b_proj_name=q_b_proj.target if q_b_proj else None,
            kv_a_proj_name=kv_a_proj.target,
            kv_a_layernorm_name=kv_a_layernorm.target,
            kv_b_proj_name=kv_b_proj.target,
            o_proj_name=o_proj_name,
        )

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        return vllm_config.model_config.use_mla

    def update_forward(self, module: nn.Module) -> None:
        """Merge `q_a_proj` and `kv_a_proj` into one fused down proj then split.
        Bypass the KV expansion method so the compressed latent reaches the `vllm_mla`
        attention interface unexpanded."""
        funcdef, fn = recover_forward(type(module))
        if self.has_q_lora:
            # q_a_proj is usually inside the `else` of `if self.q_lora_rank is None`.
            # The fused call is inserted at the top-level statement preceding both.
            q_call = single_self_call(funcdef, self.q_a_proj_name)
            kv_call = single_self_call(funcdef, self.kv_a_proj_name)
            if ast.dump(q_call.args[0]) != ast.dump(kv_call.args[0]):
                raise ValueError("down-projections read different inputs")
            names = {n.id for n in ast.walk(funcdef) if isinstance(n, ast.Name)}
            if names & {_Q_A_TEMP, _KV_A_TEMP}:
                raise ValueError("fused temporaries would shadow existing names")

            merged = f"self.{self.merged_name}"
            targets = f"{_Q_A_TEMP}, {_KV_A_TEMP}"
            sections = f"[s // {merged}.tp_size for s in {merged}.output_sizes]"
            source = f"{targets} = {merged}(__arg__).split({sections}, -1)"
            assign = ast.parse(source=source).body[0]
            placeholder = next(
                node
                for node in ast.walk(assign)
                if isinstance(node, ast.Name) and node.id == "__arg__"
            )
            replace_expr(assign, placeholder, q_call.args[0])

            index = min(_top_level_index(funcdef, call) for call in (q_call, kv_call))
            ast.copy_location(assign, funcdef.body[index])
            funcdef.body.insert(index, assign)
            replace_expr(funcdef, q_call, ast.Name(id=_Q_A_TEMP, ctx=ast.Load()))
            replace_expr(funcdef, kv_call, ast.Name(id=_KV_A_TEMP, ctx=ast.Load()))

        # Transformers expands the latent into full key/value in a dedicated method.
        # `MLAAttention` consumes the latent directly (absorbing `kv_b_proj`),
        # so replace the expansion call with its own arguments so `kv_c_normed, k_pe`
        # flow to the interface in place of `key, value`.
        expand_call = _single_expand_call(funcdef, module, self.kv_b_proj_name)
        replace_expr(
            funcdef, expand_call, ast.Tuple(elts=list(expand_call.args), ctx=ast.Load())
        )
        self.fused_forward = compile_forward(funcdef, fn)

    def update_attrs(self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"):
        quant_config = vllm_config.quant_config

        def replace_linear_by_name(name: str, style: str):
            linear = module.get_submodule(name)
            replacement = replace_linear_class(
                linear, style, quant_config, prefix=maybe_prefix(prefix, name)
            )
            setattr(module, name, replacement)

        if self.has_q_lora:
            q_a = module.get_submodule(self.q_a_proj_name)
            kv_a = module.get_submodule(self.kv_a_proj_name)
            merged = MergedColumnParallelLinear(
                input_size=q_a.in_features,
                output_sizes=[q_a.out_features, kv_a.out_features],
                bias=q_a.bias is not None,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, self.merged_name),
                return_bias=False,
                disable_tp=True,
            )
            setattr(module, self.merged_name, merged)
            # The rewritten forward calls the merged projection instead.
            delattr(module, self.q_a_proj_name)
            delattr(module, self.kv_a_proj_name)
            replace_linear_by_name(self.q_b_proj_name, "colwise")
        else:
            replace_linear_by_name(self.kv_a_proj_name, "replicate")
            replace_linear_by_name(self.q_proj_name, "colwise")

        replace_linear_by_name(self.kv_b_proj_name, "colwise")
        # MLAAttention calls kv_b_proj and expects vLLM's default return_bias=True
        module.get_submodule(self.kv_b_proj_name).return_bias = True
        if self.o_proj_name is not None:
            replace_linear_by_name(self.o_proj_name, "rowwise")
