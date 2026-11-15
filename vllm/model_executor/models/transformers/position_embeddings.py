# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recognize the learned position lookup added to a decoder's input."""

import ast
import inspect
import textwrap
from collections import Counter
from enum import Enum, auto

import torch
from torch import nn

from .fx_utils import aliasing_names, forward_parameters, recover_forward, written_names


class _Input(Enum):
    TOKENS = auto()
    POSITIONS = auto()
    EMBEDS = auto()


def _self_attr(expr: ast.expr) -> str | None:
    match expr:
        case ast.Attribute(value=ast.Name(id="self"), attr=name):
            return name
    return None


def _position_argument(table: nn.Embedding) -> str | None:
    """For a custom table, verify which argument actually indexes its weight."""
    method = type(table).forward
    if (
        hasattr(method, "__wrapped__")
        or getattr(super(type(table), table).forward, "__func__", None)
        is not nn.Embedding.forward
    ):
        return None
    try:
        (fn,) = ast.parse(textwrap.dedent(inspect.getsource(method))).body
    except (OSError, TypeError, SyntaxError, ValueError):
        return None
    if not isinstance(fn, ast.FunctionDef):
        return None
    match fn.body[-1]:
        case ast.Return(value=ast.Call(func=func, args=[index], keywords=[])) if (
            ast.unparse(func) == "super().forward"
        ):
            pass
        case _:
            return None
    if isinstance(index, ast.BinOp) and isinstance(index.op, ast.Add):
        offset = index.right
        constant = (
            offset.value
            if isinstance(offset, ast.Constant)
            else vars(table).get(_self_attr(offset))
        )
        if type(constant) is int:
            index = index.left
    if not isinstance(index, ast.Name):
        return None
    # OPT computes default positions from a mask only when no index is passed.
    for stmt in fn.body[:-1]:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if not (
            isinstance(stmt, ast.If)
            and not stmt.orelse
            and ast.unparse(stmt.test) == f"{index.id} is None"
        ):
            return None
    return index.id


def _loop_target(loop: ast.For, stack: str | None) -> str | None:
    seq, item = loop.iter, loop.target
    if isinstance(seq, ast.Call):
        match seq, item:
            case ast.Call(func=ast.Name(id="enumerate"), args=[seq]), ast.Tuple(
                elts=[_, item]
            ):
                pass
            case _:
                return None
    if isinstance(seq, ast.Subscript):
        seq = seq.value
    if stack is not None and _self_attr(seq) == stack and isinstance(item, ast.Name):
        return item.id
    return None


def find_position_embedding(
    decoder: nn.Module, input_embedding: nn.Module, layers: nn.ModuleList
) -> str | None:
    """Find a single-use position lookup added to the input of the decoder stack."""
    children = dict(decoder.named_children())
    tables = {
        m: name
        for name, m in children.items()
        if isinstance(m, nn.Embedding) and m is not input_embedding
    }
    if not tables:
        return None
    try:
        forward, fn = recover_forward(type(decoder))
    except (AttributeError, OSError, TypeError, ValueError, SyntaxError):
        return None
    # A write through an expression such as x.view(...).zero_() has no
    # named receiver for the alias analysis to invalidate.
    if any(
        isinstance(node, (ast.Attribute, ast.Subscript))
        and not isinstance(node.value, ast.Name)
        and (
            isinstance(node.ctx, (ast.Store, ast.Del))
            or isinstance(node, ast.Attribute)
            and node.attr.endswith("_")
        )
        for node in ast.walk(forward)
    ):
        return None
    stack = next((name for name, child in children.items() if child is layers), None)
    names = Counter(
        node.id
        for node in ast.walk(forward)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    refs = Counter(
        name
        for node in ast.walk(forward)
        if isinstance(node, ast.Attribute) and (name := _self_attr(node)) is not None
    )
    # A helper, property or indirect access to the decoder could read the same
    # table again. Only inspect direct accesses to stored members.
    members = {
        **vars(decoder),
        **decoder._modules,
        **decoder._buffers,
        **decoder._parameters,
    }
    if names["self"] != refs.total() or any(
        name not in members or name in ("_modules", "__dict__") for name in refs
    ):
        return None
    # Embedding instances denote lookup results; names denote completed sums.
    values: dict[str, _Input | nn.Embedding | str | None] = dict(
        input_ids=_Input.TOKENS,
        position_ids=_Input.POSITIONS,
        inputs_embeds=_Input.EMBEDS,
    )
    values = {k: v for k, v in values.items() if k in forward_parameters(type(decoder))}

    def global_binding(expr: ast.expr):
        if isinstance(expr, ast.Name) and expr.id not in fn.__code__.co_varnames:
            return fn.__globals__.get(expr.id)
        return None

    def add_operands(expr: ast.expr):
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return expr.left, expr.right
        if not isinstance(expr, ast.Call):
            return None
        args = list(expr.args)
        if isinstance(expr.func, ast.Attribute) and expr.func.attr == "add":
            receiver = expr.func.value
            if global_binding(receiver) is not torch:
                args.insert(0, receiver)
        elif global_binding(expr.func) is not torch.add:
            return None
        if len(args) > 2:
            return None
        operands = dict(zip(("input", "other"), args))
        for kw in expr.keywords:
            if (
                kw.arg == "alpha"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == 1
            ):
                continue
            if kw.arg not in ("input", "other") or kw.arg in operands:
                return None
            operands[kw.arg] = kw.value
        if "input" in operands and "other" in operands:
            return operands["input"], operands["other"]
        return None

    def value(expr: ast.expr):
        if (operands := add_operands(expr)) is not None:
            left, right = map(value, operands)
            for token, pos in ((left, right), (right, left)):
                if token is _Input.EMBEDS and isinstance(pos, nn.Embedding):
                    return tables[pos]
                if (
                    isinstance(token, str)
                    and pos is None
                    and isinstance(expr, ast.BinOp)
                ):
                    return token  # An additional token-type embedding.
            return None
        match expr:
            case ast.Name(id=name):
                source = values.get(name)
                if isinstance(source, nn.Embedding) and names[name] != 1:
                    return None  # Another consumer would also lose the lookup.
                return source
            case ast.Call(func=ast.Attribute() as fn, args=args, keywords=kws):
                attr = _self_attr(fn)
                if attr is None:
                    source = value(fn.value)
                    if fn.attr == "to" or (
                        source is _Input.TOKENS and fn.attr in ("view", "reshape")
                    ):
                        return source
                    return None
                child = children.get(attr)
                first = value(args[0]) if args else None
                if child is input_embedding and (
                    first is _Input.TOKENS
                    or any(
                        kw.arg == "input" and value(kw.value) is _Input.TOKENS
                        for kw in kws
                    )
                ):
                    return _Input.EMBEDS
                if isinstance(child, nn.Dropout) and isinstance(first, str):
                    return first
                if (
                    isinstance(child, nn.Embedding)
                    and child in tables
                    and refs[attr] == 1
                ):
                    params = list(forward_parameters(type(child)))[1:]
                    args_by_name = dict(zip(params, args))
                    args_by_name.update((kw.arg, kw.value) for kw in kws if kw.arg)
                    param = (
                        "input"
                        if type(child).forward is nn.Embedding.forward
                        else _position_argument(child)
                    )
                    if (
                        param in args_by_name
                        and value(args_by_name[param]) is _Input.POSITIONS
                    ):
                        return child
        return None

    def known_condition(test: ast.expr) -> bool | None:
        match test:
            case ast.Compare(
                left=lhs,
                ops=[ast.Is() | ast.IsNot() as op],
                comparators=[ast.Constant(value=None)],
            ):
                name = _self_attr(lhs)
                if name is not None and name in members:
                    missing = members[name] is None
                elif value(lhs) is _Input.POSITIONS:
                    missing = False  # The backend always passes position_ids.
                else:
                    return None
                return missing == isinstance(op, ast.Is)
        return None

    def forget_writes(stmt: ast.stmt):
        writes = written_names([stmt])
        if writes:
            for name in tuple(values):
                if writes & aliasing_names(forward, {name}, decoder):
                    values.pop(name)

    def visit(stmts: list[ast.stmt]) -> str | None:
        nonlocal values
        for stmt in stmts:
            match stmt:
                case ast.Assign(targets=[ast.Name(id=name)], value=rhs):
                    # An assignment can also write through an argument on its RHS.
                    forget_writes(ast.Assign(targets=[], value=rhs))
                    values[name] = value(rhs)
                case ast.Return():
                    return None
                case ast.If(test=test, body=yes, orelse=no):
                    condition = known_condition(test)
                    if condition is not None:
                        visit(yes if condition else no)
                        continue
                    before = values.copy()
                    visit(yes)
                    yes_values, values = values, before
                    visit(no)
                    # Retain only facts true on both paths.
                    values = {k: v for k, v in values.items() if v == yes_values.get(k)}
                case ast.For() if layer := _loop_target(stmt, stack):
                    # Only recognize a sum already present at the loop boundary.
                    # A lookup inside the loop belongs to every decoder layer.
                    calls = [
                        n
                        for n in ast.walk(stmt)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id == layer
                    ]
                    if len(calls) != 1 or not calls[0].args:
                        return None
                    call = calls[0]
                    for line in stmt.body:
                        if getattr(line, "value", None) is call:
                            arg = call.args[0]
                            result = (
                                values.get(arg.id)
                                if isinstance(arg, ast.Name)
                                else None
                            )
                            return result if isinstance(result, str) else None
                        forget_writes(line)
                    return None
                case _:
                    forget_writes(stmt)
        return None

    return visit(forward.body)
