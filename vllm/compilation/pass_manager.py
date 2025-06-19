# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import fx as fx

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .activation_quant_fusion import ActivationQuantFusionPass
from .collective_fusion import AsyncTPPass
from .fix_functionalization import FixFunctionalizationPass
from .fusion import FusionPass
from .fusion_attn import AttnFusionPass
from .inductor_pass import CustomGraphPass, InductorPass, get_pass_context
from .noop_elimination import NoOpEliminationPass
from .sequence_parallelism import SequenceParallelismPass
from .vllm_inductor_pass import VllmInductorPass

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
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)

        # always run fix_functionalization last
        self.fix_functionalization(graph)

    def configure(self, config: VllmConfig):
        self.pass_config = config.compilation_config.pass_config
        if self.pass_config.enable_noop:
            self.passes += [NoOpEliminationPass(config)]

        if self.pass_config.enable_fusion:
            self.passes += [FusionPass.instance(config)]
            self.passes += [ActivationQuantFusionPass(config)]

        if self.pass_config.enable_sequence_parallelism:
            self.passes += [SequenceParallelismPass(config)]
            if self.pass_config.enable_async_tp:
                self.passes += [AsyncTPPass(config)]

        if self.pass_config.enable_attn_fusion:
            self.passes += [AttnFusionPass(config)]

        self.fix_functionalization = FixFunctionalizationPass(config)

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
