# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Union

from torch import fx

from vllm.compilation.inductor_pass import InductorPass


class TestBackend:
    """
    This class provides a simple Inductor backend that can be used for testing.
    It takes a list of custom passes and runs them after Inductor's passes.
    It also saves the graph before and after the custom passes for inspection.
    """

    def __init__(self, *passes: Union[InductorPass, Callable[[fx.Graph],
                                                             None]]):
        self.custom_passes = list(passes)
        from torch._inductor import config
        self.current_config = config.shallow_copy_dict()
        self.current_config['force_disable_caches'] = True
        self.current_config['post_grad_custom_post_pass'] = self.post_pass

    def __call__(self, graph: fx.GraphModule, example_inputs):
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(graph,
                          example_inputs,
                          config_patches=self.current_config)

    def post_pass(self, graph: fx.Graph):
        self.graph_pre_pass = deepcopy(graph)
        for pass_ in self.custom_passes:
            pass_(graph)

        self.graph_post_pass = deepcopy(graph)
        # assign by reference, will reflect the final state of the graph
        self.final_graph = graph
