# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest
import torch
from torch._inductor.codecache import BypassFxGraphCache

from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.compilation.pass_manager import PostGradPassManager
from vllm.config import CompilationConfig


def simple_callable(graph: torch.fx.Graph):
    pass


callable_decorated = CallableInductorPass(simple_callable,
                                          InductorPass.hash_source(__file__))


@pytest.mark.parametrize(
    "works, callable",
    [(False, simple_callable), (True, callable_decorated),
     (True, CallableInductorPass(simple_callable, "simple_callable"))])
def test_pass_manager(works: bool, callable):
    config = CompilationConfig().pass_config
    pass_manager = PostGradPassManager(
    )  # Create the pass manager without arguments
    pass_manager.configure(config)  # Adds default passes

    # Try to add the callable to the pass manager
    # For non-InductorPass callables, this should fail the assertion in add()
    if isinstance(callable, InductorPass):
        pass_manager.add(callable)
        # should succeed for proper InductorPass instances
        if works:
            pickle.dumps(pass_manager)
        else:
            with pytest.raises(BypassFxGraphCache):
                pickle.dumps(pass_manager)
    else:
        # For simple_callable, this should fail
        with pytest.raises(AssertionError):
            pass_manager.add(callable)
