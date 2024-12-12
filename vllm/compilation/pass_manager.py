from typing import List

from torch import fx as fx

from vllm.config import CompilationConfig
from vllm.logger import init_logger

from .fix_functionalization import FixFunctionalizationPass
from .fusion import FusionPass
from .inductor_pass import InductorPass
from .reshapes import RedundantReshapesPass

logger = init_logger(__name__)


class PostGradPassManager:
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It also supports pickling, which is used by the Inductor code cache.
    TODO(torch==2.6), use CustomGraphPass
    (torch._inductor.custom_graph_pass.CustomGraphPass)

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (RedundantReshapesPass, FusionPass)
    3. config["post_grad_custom_post_pass"] (if it exists)
    4. fix_functionalization
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self):
        self.passes: List[InductorPass] = []

    def __call__(self, graph: fx.Graph):
        for pass_ in self.passes:
            pass_(graph)

        # always run fix_functionalization last
        self.fix_functionalization(graph)

    def configure(self, pass_config: CompilationConfig.PassConfig):
        self.pass_config = pass_config
        if pass_config.enable_reshape:
            self.passes += [RedundantReshapesPass(pass_config)]

        if pass_config.enable_fusion:
            self.passes += [FusionPass.instance(pass_config)]

        self.fix_functionalization = FixFunctionalizationPass(pass_config)

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def __getstate__(self):
        """
        Custom pickling for the pass manager, as some passes cannot be pickled.
        Pickling occurs because the pass manager is set as the value of
        `config["post_grad_custom_post_pass"]` in the Inductor config.
        The config is pickled to act as a key in the Inductor code cache.
        Any other passes in the config are pickled as well.

        TODO(torch==2.6), use the `uuid` method in CustomGraphPass instead.
        """
        state = {"pass_config": self.pass_config.uuid(), "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.fix_functionalization.uuid())
        return state

    def __setstate__(self, state):
        """
        Do not allow unpickling of the pass manager.
        If this is needed in the future, it should properly pickle the passes.
        """
        raise ValueError("Cannot unpickle PostGradPassManager")
