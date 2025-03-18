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
        (True, CallableInductorPass(simple_callable)),
    ],
)
def test_pass_manager(works: bool, callable):
    config = CompilationConfig().pass_config

    pass_manager = PostGradPassManager()  # pass manager without arguments
    pass_manager.configure(config)  # default passes

    # Try to add the callable to the pass manager
    if works:
        pass_manager.add(callable)
        pickle.dumps(pass_manager)
    else:
        with pytest.raises(AssertionError):
            pass_manager.add(callable)
