# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Union

from torch import fx
from torch._ops import OpOverload

from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.inductor_pass import InductorPass
from vllm.compilation.pass_manager import with_pattern_match_debug
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig, get_current_vllm_config


class LazyInitPass(InductorPass):
    """
    If there's a pass that we want to initialize lazily in a test,
    we can wrap it in LazyInitPass, which will initialize the pass when invoked
    and then immediately invoke it.
    """

    def __init__(self, pass_cls: type[VllmInductorPass],
                 vllm_config: VllmConfig):
        self.pass_cls = pass_cls
        self.vllm_config = weakref.proxy(vllm_config)  # avoid cycle

    def __call__(self, graph: fx.Graph) -> None:
        self.pass_ = self.pass_cls(self.vllm_config)
        self.pass_(graph)


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

    @with_pattern_match_debug
    def post_pass(self, graph: fx.Graph):
        self.graph_pre_pass = deepcopy(graph)

        VllmInductorPass.dump_prefix = 0
        for pass_ in self.custom_passes:
            pass_(graph)
            VllmInductorPass.dump_prefix += 1

        VllmInductorPass.dump_prefix = None

        self.graph_post_pass = deepcopy(graph)
        # assign by reference, will reflect the final state of the graph
        self.final_graph = graph

    def check_before_ops(self, ops: Sequence[OpOverload], fully_replaced=True):
        for op in ops:
            num_pre = len(list(find_op_nodes(op, self.graph_pre_pass)))
            num_post = len(list(find_op_nodes(op, self.graph_post_pass)))
            assert num_pre > 0, f"Op {op.name()} not found in pre-pass graph"
            assert num_pre > num_post, f"All nodes remain for op {op.name()}"
            if fully_replaced:
                assert num_post == 0, \
                    f"Unexpected op {op.name()} in post-pass graph"

    def check_after_ops(self, ops: Sequence[OpOverload]):
        for op in ops:
            num_pre = len(list(find_op_nodes(op, self.graph_pre_pass)))
            num_post = len(list(find_op_nodes(op, self.graph_post_pass)))
            assert num_pre == 0, f"Unexpected op {op.name()} in pre-pass graph"
            assert num_post > 0, f"Op {op.name()} not found in post-pass graph"

    def op_count(self, op: OpOverload, before=False) -> int:
        graph = self.graph_pre_pass if before else self.graph_post_pass
        return len(list(find_op_nodes(op, graph)))
