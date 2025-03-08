# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest
import torch
from torch._inductor.codecache import BypassFxGraphCache

from vllm.compilation.config import CompilationConfig
from vllm.compilation.inductor_pass import (CallableInductorPass,
                                            as_inductor_pass)
from vllm.compilation.pass_manager import PostGradPassManager


def simple_callable(graph: torch.fx.Graph):
    pass


@as_inductor_pass(files=(__file__, ))
def callable_decorated(graph: torch.fx.Graph):
    pass


@pytest.mark.parametrize(
    "works, callable",
    [(False, simple_callable), (True, callable_decorated),
     (True, CallableInductorPass(simple_callable, "simple_callable"))])
def test_pass_manager(works: bool, callable):
    config = CompilationConfig().pass_config
    pass_manager = PostGradPassManager([callable])
    pass_manager.configure(config)  # Adds default passes

    if works:
        pickle.dumps(pass_manager)
    else:
        with pytest.raises(BypassFxGraphCache):
            pickle.dumps(pass_manager)
