# SPDX-License-Identifier: Apache-2.0

from typing import List

from torch import fx as fx

from vllm.config import CompilationConfig
from vllm.logger import init_logger

from .fix_functionalization import FixFunctionalizationPass
from .fusion import FusionPass
from .inductor_pass import CustomGraphPass, InductorPass
from .noop_elimination import NoOpEliminationPass

logger = init_logger(__name__)


class PostGradPassManager(CustomGraphPass):
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache. That includes torch<2.6
    support using pickling (in .inductor_pass.CustomGraphPass).

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass, FusionPass)
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
        if pass_config.enable_noop:
            self.passes += [NoOpEliminationPass(pass_config)]

        if pass_config.enable_fusion:
            self.passes += [FusionPass.instance(pass_config)]

        self.fix_functionalization = FixFunctionalizationPass(pass_config)

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        state = {"pass_config": self.pass_config.uuid(), "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.fix_functionalization.uuid())
        return InductorPass.hash_dict(state)
