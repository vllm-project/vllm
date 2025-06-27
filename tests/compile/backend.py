# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Optional, Union

from torch import fx
from torch._ops import OpOverload

from vllm.compilation.fx_utils import find_op_nodes
from vllm.compilation.inductor_pass import InductorPass
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
        self.pass_cls(self.vllm_config)(graph)


class TestPassManager(InductorPass):
    """
    TODO clean this up more
    """

    def __init__(
        self,
        *passes: Union[InductorPass, Callable[[fx.Graph], None]],
        check_fn: Optional[Callable[['TestPassManager'], None]] = None,
    ):
        self.custom_passes = list(passes)
        self.check_fn = check_fn

    def __call__(self, graph: fx.Graph):
        print(f"TestPassManager: Before pass: {self}")
        self.graph_pre_pass = deepcopy(graph)
        for pass_ in self.custom_passes:
            pass_(graph)

        self.graph_post_pass = deepcopy(graph)
        # assign by reference, will reflect the final state of the graph
        self.final_graph = graph

        # TestPassManager can get deepcopied,
        # so pass the current instance to the check function
        if self.check_fn:
            self.check_fn(self)

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

    def __init__(
        self,
        *passes: Union[InductorPass, Callable[[fx.Graph], None]],
        check_fn: Optional[Callable[['TestPassManager'], None]] = None,
    ):
        compile_config = get_current_vllm_config().compilation_config
        self.inductor_config = compile_config.inductor_compile_config
        self.inductor_config['force_disable_caches'] = True
        self.test_pass = TestPassManager(*passes, check_fn=check_fn)
        self.inductor_config['post_grad_custom_post_pass'] = self.test_pass

    def __call__(self, graph: fx.GraphModule, example_inputs):
        print("AAAAA")
        self.graph_pre_compile = deepcopy(graph)
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(graph,
                          example_inputs,
                          config_patches=self.inductor_config)

    @property
    def graph_post_pass(self):
        return self.test_pass.graph_post_pass

    @property
    def graph_pre_pass(self):
        return self.test_pass.graph_pre_pass

    @property
    def final_graph(self):
        return self.test_pass.final_graph
