# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import re
from dataclasses import fields

from vllm.compilation.counter import CompilationCounter
from vllm.compilation.wrapper import reset_compile_wrapper

# `num_<field> = 0` style assignments performed by the reset helper.
ASSIGNMENT = re.compile(r"compilation_counter\.(\w+)\s*=\s*0")


def test_reset_compile_wrapper_clears_every_counter_field():
    """The reset helper must zero *all* CompilationCounter fields, otherwise a
    counter survives an elastic-EP rebalance and no longer describes the
    compilation that follows it."""
    source = inspect.getsource(reset_compile_wrapper)
    cleared = set(ASSIGNMENT.findall(source))
    declared = {f.name for f in fields(CompilationCounter)}

    assert declared - cleared == set(), (
        f"CompilationCounter fields never reset: {sorted(declared - cleared)}"
    )


def test_defaults_are_all_zero():
    """Guards the assumption above: every field's default is 0, so 'reset' means
    'back to default' for each of them."""
    for field in fields(CompilationCounter):
        assert field.default == 0, field.name
