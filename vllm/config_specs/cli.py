# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Argparse adapter for the portable scheduler field declarations."""

import argparse
import inspect
from collections.abc import Callable
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from . import cli_default, cli_fields


def arguments(
    cls: type,
    *,
    human_readable_int: Callable[[str], int],
    optional_type: Callable[[Callable[[str], Any]], Callable[[str], Any | None]],
    include_help: bool = True,
):
    for field, declaration in cli_fields(cls):
        dest = declaration.dest or field.name
        flags = declaration.flags or (f"--{dest.replace('_', '-')}",)
        help_text = (
            inspect.cleandoc(field.metadata.get("doc", "")) if include_help else ""
        )
        kwargs: dict[str, Any] = {
            "dest": dest,
            "default": cli_default(cls, field.name),
            "help": help_text.replace("%", "%%"),
        }
        annotation = field.type
        types = (
            set(get_args(annotation))
            if get_origin(annotation) in (Union, UnionType)
            else {annotation}
        )
        if bool in types:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif get_origin(annotation) is Literal:
            choices = sorted(get_args(annotation))
            kwargs.update(type=type(choices[0]), choices=choices)
        elif int in types:
            kwargs["type"] = human_readable_int if declaration.human_readable else int
            if declaration.human_readable:
                kwargs["help"] += f"\n\n{human_readable_int.__doc__}"
        elif float in types:
            kwargs["type"] = float
        elif str in types:
            kwargs["type"] = str
        else:
            raise ValueError(f"Unsupported CLI field {cls.__name__}.{field.name}")
        if type(None) in types and bool not in types:
            kwargs["type"] = optional_type(kwargs["type"])
        yield field.name, flags, kwargs


def add_cli_args(
    parser,
    cls: type,
    *,
    title: str,
    human_readable_int: Callable[[str], int],
    optional_type: Callable[[Callable[[str], Any]], Callable[[str], Any | None]],
    include_help: bool = True,
):
    prepared = list(
        arguments(
            cls,
            human_readable_int=human_readable_int,
            optional_type=optional_type,
            include_help=include_help,
        )
    )
    group = parser.add_argument_group(title=title, description=cls.__doc__)
    for _, flags, kwargs in prepared:
        group.add_argument(*flags, **kwargs)
    return group
