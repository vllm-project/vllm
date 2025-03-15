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
    [
        (False, simple_callable),
        (True, callable_decorated),
        (True, CallableInductorPass(simple_callable, "simple_callable")),
    ],
)
def test_pass_manager(works: bool, callable):
    config = CompilationConfig().pass_config

    pass_manager = PostGradPassManager()  # pass manager without arguments
    pass_manager.configure(config)  # default passes

    # Try to add the callable to the pass manager
    try:
        pass_manager.add(callable)
        # If we got here, the add was successful (should be an InductorPass)
        # Now check pickling behavior based on the works parameter
        if works:
            pickle.dumps(pass_manager)
        else:
            with pytest.raises(BypassFxGraphCache):
                pickle.dumps(pass_manager)
    except AssertionError:
        # Should only get here for non-InductorPass callables
        assert not isinstance(callable, InductorPass)
