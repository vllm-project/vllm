# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA fuser: replace a Transformers MLA attention module with vLLM's MLA layer."""

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch import fx, nn

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.models.transformers.fusers.base import StackedFuser
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    is_linear,
    recover_forward,
    replace_expr,
    single_self_call,
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
    """A linear directly consuming `node`'s output, if any."""
    return next((user for user in node.users if is_linear(user, module)), None)


def _norm_size(norm: nn.Module) -> int:
    weight = getattr(norm, "weight", None)
    return weight.numel() if weight is not None else -1


def _callee_name(call: ast.Call) -> str | None:
    """The bare name of what `call` calls (`torch.split(...)` -> `split`)."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _enclosing_assign(funcdef: ast.FunctionDef, node: ast.AST) -> ast.Assign:
    """The unique `Assign` statement whose value contains `node`."""
    assigns = [
        stmt
        for stmt in ast.walk(funcdef)
        if isinstance(stmt, ast.Assign)
        and any(child is node for child in ast.walk(stmt.value))
    ]
    if len(assigns) != 1:
        raise ValueError("expression is not inside exactly one assignment")
    return assigns[0]


def _single_target_name(assign: ast.Assign) -> str:
    """The single plain `Name` this statement assigns to."""
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        raise ValueError("statement does not assign to a single name")
    return assign.targets[0].id


def _tuple_target_names(assign: ast.Assign) -> list[str]:
    """Names bound by a tuple-unpacking assignment."""
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Tuple):
        raise ValueError("statement does not unpack into a tuple")
    elements = assign.targets[0].elts
    if not all(isinstance(element, ast.Name) for element in elements):
        raise ValueError("tuple unpacking has a non-name target")
    return [element.id for element in elements]  # type: ignore[attr-defined]


def _top_level_index(funcdef: ast.FunctionDef, node: ast.AST) -> int:
    """Index in `funcdef.body` of the top-level statement containing `node`."""
    for index, stmt in enumerate(funcdef.body):
        if any(child is node for child in ast.walk(stmt)):
            return index
    raise ValueError("node is not in the function body")


def _replace_stmt(funcdef: ast.FunctionDef, old: ast.stmt, new: ast.stmt) -> None:
    """Replace statement `old` (by identity) with `new`, in place."""
    for parent in ast.walk(funcdef):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if not isinstance(block, list):
                continue
            for index, stmt in enumerate(block):
                if stmt is old:
                    ast.copy_location(new, old)
                    block[index] = new
                    return
    raise ValueError("statement not found in the function body")


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
    o_proj_name: str
    merged_name: ClassVar[str] = "fused_qkv_a_proj"
    merged_cls: ClassVar[str] = "MergedColumnParallelLinear"

    @property
    def has_q_lora(self) -> bool:
        return self.q_a_proj_name is not None

    def info(self, name: str) -> str:
        info_str = f"Fused: {name} ({self.source_cls}) -> MLAAttention"
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
        """Detect MLA by its compressed-KV signature: a linear whose output is
        normalized and fed to a second linear (a down/up projection pair).

        There are two such chains when the query is also low-rank; KV is the one
        whose down-projection is wider than its latent norm (it also carries the
        rope key), and the query chain, if present, matches its norm exactly."""
        chains = []
        for node in graph.nodes:
            if node.op != "call_module" or is_linear(node, module) or not node.args:
                continue
            source = _upstream_linear(node.args[0], module)
            sink = _downstream_linear(node, module)
            if source is None or sink is None or not _consumes_placeholder(source):
                continue
            chains.append((source, node, sink))

        def is_kv(chain) -> bool:
            source, norm, _ = chain
            source_mod = module.get_submodule(source.target)
            norm_mod = module.get_submodule(norm.target)
            return source_mod.out_features != _norm_size(norm_mod)

        kv_chains = [chain for chain in chains if is_kv(chain)]
        q_chains = [chain for chain in chains if not is_kv(chain)]
        if len(kv_chains) != 1 or len(q_chains) > 1:
            return None
        kv_source, kv_norm, kv_sink = kv_chains[0]

        consumed = {kv_source.target, kv_sink.target}
        q_proj_name = q_a_proj_name = q_a_layernorm_name = q_b_proj_name = None
        if q_chains:
            q_source, q_norm, q_sink = q_chains[0]
            q_a_proj_name, q_a_layernorm_name, q_b_proj_name = (
                q_source.target,
                q_norm.target,
                q_sink.target,
            )
            consumed |= {q_source.target, q_sink.target}
        else:
            # Without a query chain, the other linear reading the hidden states
            # (besides KV's down-projection) is `q_proj`.
            placeholder_linears = {
                node.target
                for node in graph.nodes
                if is_linear(node, module) and _consumes_placeholder(node)
            }
            others = placeholder_linears - {kv_source.target}
            if len(others) != 1:
                return None
            q_proj_name = next(iter(others))
            consumed.add(q_proj_name)

        # `o_proj` is the sole remaining linear child (it sits past the attention
        # op, so it never reaches the traced graph).
        remaining = [
            name
            for name, child in module.named_children()
            if isinstance(child, nn.Linear) and name not in consumed
        ]
        if len(remaining) != 1:
            return None

        return cls(
            source_cls=type(module).__name__,
            q_proj_name=q_proj_name,
            q_a_proj_name=q_a_proj_name,
            q_a_layernorm_name=q_a_layernorm_name,
            q_b_proj_name=q_b_proj_name,
            kv_a_proj_name=kv_source.target,
            kv_a_layernorm_name=kv_norm.target,
            kv_b_proj_name=kv_sink.target,
            o_proj_name=remaining[0],
        )

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        return vllm_config.model_config.use_mla

    def _merge_down_projections(self, funcdef: ast.FunctionDef) -> None:
        """`q_a_proj(x)`, `kv_a_proj_with_mqa(x)` -> one fused call plus a split.

        Unlike `QKVFuser`, the two calls sit in *different* blocks (`q_a_proj` is
        inside the `else` of `if self.q_lora_rank is None`), so the fused call is
        inserted at the top-level statement preceding both.
        """
        q_call = single_self_call(funcdef, self.q_a_proj_name)
        kv_call = single_self_call(funcdef, self.kv_a_proj_name)
        if ast.dump(q_call.args[0]) != ast.dump(kv_call.args[0]):
            raise ValueError("down-projections read different inputs")
        names = {node.id for node in ast.walk(funcdef) if isinstance(node, ast.Name)}
        if names & {_Q_A_TEMP, _KV_A_TEMP}:
            raise ValueError("fused temporaries would shadow existing names")

        merged = f"self.{self.merged_name}"
        sections = f"[s // {merged}.tp_size for s in {merged}.output_sizes]"
        source = f"{_Q_A_TEMP}, {_KV_A_TEMP} = {merged}(__arg__).split({sections}, -1)"
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

    def update_forward(self, module: nn.Module) -> None:
        funcdef, fn = recover_forward(type(module))

        if self.has_q_lora:
            self._merge_down_projections(funcdef)

        # `kv_b_proj(kv_a_layernorm(k_pass)).view(...).transpose(...)` -> the
        # normalized latent, with a head axis so it still concatenates with the
        # (already 4D) rope key. `.unsqueeze(1)` avoids naming batch/seq locals.
        kv_b_call = single_self_call(funcdef, self.kv_b_proj_name)
        kv_b_assign = _enclosing_assign(funcdef, kv_b_call)
        latent_name = _single_target_name(kv_b_assign)
        kv_b_assign.value = ast.Call(
            func=ast.Attribute(
                value=kv_b_call.args[0], attr="unsqueeze", ctx=ast.Load()
            ),
            args=[ast.Constant(value=1)],
            keywords=[],
        )

        # Splitting the expanded key into (nope key, value) has no meaning now;
        # `value_states` only has to stay bound for the interface call, which
        # rejects a non-None value as proof the rewrite did not run.
        splits = [
            stmt
            for stmt in ast.walk(funcdef)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Call)
            and _callee_name(stmt.value) == "split"
            and stmt.value.args
            and isinstance(stmt.value.args[0], ast.Name)
            and stmt.value.args[0].id == latent_name
        ]
        if len(splits) != 1:
            raise ValueError(f"{latent_name} is not split exactly once")
        value_name = _tuple_target_names(splits[0])[1]
        _replace_stmt(
            funcdef,
            splits[0],
            ast.Assign(
                targets=[ast.Name(id=value_name, ctx=ast.Store())],
                value=ast.Constant(value=None),
            ),
        )
        # The rope key's `.expand(*k_pass.shape[:-1], -1)` is left alone: the
        # latent is now `[batch, 1, seq, kv_lora_rank]`, so it is already a no-op.

        self.fused_forward = compile_forward(funcdef, fn)

    def update_attrs(self, module: nn.Module, prefix: str, vllm_config: "VllmConfig"):
        quant_config = vllm_config.quant_config

        def replace_linear_by_name(name: str, style: str) -> nn.Module:
            linear = module.get_submodule(name)
            _prefix = maybe_prefix(prefix, name)
            replacement = replace_linear_class(linear, style, quant_config, _prefix)
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
        replace_linear_by_name(self.o_proj_name, "rowwise")
