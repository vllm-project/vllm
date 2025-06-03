# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Callable, Union

from torch import fx

from vllm.compilation.fx_utils import (find_specified_fn,
                                       find_specified_fn_maybe)
from vllm.compilation.inductor_pass import InductorPass
from vllm.config import get_current_vllm_config


class TestBackend:
    """
    This class provides a simple Inductor backend that can be used for testing.
    It takes a list of custom passes and runs them after Inductor's passes.
    It also saves the graph before and after the custom passes for inspection.

    Inductor config can be modified directly by editing the inductor_config
    property. This can be helpful for adding passes like the
    'pre_grad_custom_pass' and the 'post_grad_custom_pre_pass'.
    Inductor config is default-initialized from VllmConfig.CompilationConfig.
    """

    def __init__(self, *passes: Union[InductorPass, Callable[[fx.Graph],
                                                             None]]):
        self.custom_passes = list(passes)
        compile_config = get_current_vllm_config().compilation_config
        self.inductor_config = compile_config.inductor_compile_config
        self.inductor_config['force_disable_caches'] = True
        self.inductor_config['post_grad_custom_post_pass'] = self.post_pass

    def __call__(self, graph: fx.GraphModule, example_inputs):
        self.graph_pre_compile = deepcopy(graph)
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(graph,
                          example_inputs,
                          config_patches=self.inductor_config)

    def post_pass(self, graph: fx.Graph):
        self.graph_pre_pass = deepcopy(graph)
        for pass_ in self.custom_passes:
            pass_(graph)

        self.graph_post_pass = deepcopy(graph)
        # assign by reference, will reflect the final state of the graph
        self.final_graph = graph

    def check_before_ops(self, ops,
                         find_fn=find_specified_fn, \
                         find_fn_maybe=find_specified_fn_maybe, \
                        ops_fully_replaced=True):
        for op in ops:
            find_fn(self.graph_pre_pass.nodes, op)
            if ops_fully_replaced:
                assert find_fn_maybe(self.graph_post_pass.nodes, op) is None

    def check_after_ops(self, ops,
                        find_fn=find_specified_fn, \
                        find_fn_maybe=find_specified_fn_maybe):
        for op in ops:
            find_fn(self.graph_post_pass.nodes, op)
            assert find_fn_maybe(self.graph_pre_pass.nodes, op) is None
