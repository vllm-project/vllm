# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import textwrap
from dataclasses import MISSING, Field, field, fields, is_dataclass
from typing import (TYPE_CHECKING, Any, Literal, TypeVar, Union, get_args,
                    get_origin)

from vllm.logger import init_logger

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
    ConfigType = type[DataclassInstance]
else:
    QuantizationConfig = Any
    ConfigType = type

logger = init_logger(__name__)

ConfigT = TypeVar("ConfigT", bound=ConfigType)


def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    """

    def pairwise(iterable):
        """
        Manually implement https://docs.python.org/3/library/itertools.html#itertools.pairwise

        Can be removed when Python 3.9 support is dropped.
        """
        iterator = iter(iterable)
        a = next(iterator, None)

        for b in iterator:
            yield a, b
            a = b

    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (not isinstance(a, (ast.Assign, ast.AnnAssign))
                or not isinstance(b, ast.Expr)
                or not isinstance(b.value, ast.Constant)
                or not isinstance(b.value.value, str)):
            continue

        doc = inspect.cleandoc(b.value.value)

        # An assignment can have multiple targets (a = b = v), but an
        # annotated assignment only has one target.
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out


def config(cls: ConfigT) -> ConfigT:
    """
    A decorator that ensures all fields in a dataclass have default values
    and that each field has a docstring.

    If a `ConfigT` is used as a CLI argument itself, the default value provided
    by `get_kwargs` will be the result parsing a JSON string as the kwargs
    (i.e. `ConfigT(**json.loads(cli_arg))`). However, if a particular `ConfigT`
    requires custom construction from CLI (i.e. `CompilationConfig`), it can
    have a `from_cli` method, which will be called instead.
    """
    if not is_dataclass(cls):
        raise TypeError("The decorated class must be a dataclass.")
    attr_docs = get_attr_docs(cls)
    for f in fields(cls):
        if f.init and f.default is MISSING and f.default_factory is MISSING:
            raise ValueError(
                f"Field '{f.name}' in {cls.__name__} must have a default value."
            )

        if f.name not in attr_docs:
            raise ValueError(
                f"Field '{f.name}' in {cls.__name__} must have a docstring.")

        if get_origin(f.type) is Union:
            args = get_args(f.type)
            literal_args = [arg for arg in args if get_origin(arg) is Literal]
            if len(literal_args) > 1:
                raise ValueError(
                    f"Field '{f.name}' in {cls.__name__} must use a single "
                    "Literal type. Please use 'Literal[Literal1, Literal2]' "
                    "instead of 'Union[Literal1, Literal2]'.")
    return cls


def get_field(cls: ConfigType, name: str) -> Field:
    """Get the default factory field of a dataclass by name. Used for getting
    default factory fields in `EngineArgs`."""
    if not is_dataclass(cls):
        raise TypeError("The given class is not a dataclass.")
    cls_fields = {f.name: f for f in fields(cls)}
    if name not in cls_fields:
        raise ValueError(f"Field '{name}' not found in {cls.__name__}.")
    named_field: Field = cls_fields[name]
    if (default_factory := named_field.default_factory) is not MISSING:
        return field(default_factory=default_factory)
    if (default := named_field.default) is not MISSING:
        return field(default=default)
    raise ValueError(
        f"{cls.__name__}.{name} must have a default value or default factory.")


def is_init_field(cls: ConfigType, name: str) -> bool:
    return next(f for f in fields(cls) if f.name == name).init
