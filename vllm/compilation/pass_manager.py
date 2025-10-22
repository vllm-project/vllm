# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools

from torch import fx as fx

from vllm import envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import set_env_var

from .post_cleanup import PostCleanupPass
from .vllm_inductor_pass import VllmInductorPass

if current_platform.is_cuda_alike():
    from .activation_quant_fusion import ActivationQuantFusionPass
    from .fusion import RMSNormQuantFusionPass
    from .fusion_attn import AttnFusionPass

if current_platform.is_rocm():
    from .rocm_aiter_rmsnorm_fusion import (
        RMSNormAiterQuantFusionPass,
        is_rocm_aiter_enabled,
    )

if current_platform.is_cuda():
    from .collective_fusion import AllReduceFusionPass, AsyncTPPass

from .fix_functionalization import FixFunctionalizationPass
from .inductor_pass import CustomGraphPass, InductorPass, get_pass_context
from .noop_elimination import NoOpEliminationPass
from .sequence_parallelism import SequenceParallelismPass

logger = init_logger(__name__)


def with_pattern_match_debug(fn):
    """
    Function decorator that turns on inductor pattern match debug
    for the duration of the call.
    Used to avoid logging builtin Inductor pattern matching.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if (debug_val := envs.VLLM_PATTERN_MATCH_DEBUG) is not None:
            # optionally check rank here
            with set_env_var("TORCHINDUCTOR_PATTERN_MATCH_DEBUG", debug_val):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper


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
        self.passes: list[InductorPass] = []

    @with_pattern_match_debug
    def __call__(self, graph: fx.Graph):
        VllmInductorPass.dump_prefix = 0  # reset dump index

        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable(shape):
                pass_(graph)
                VllmInductorPass.dump_prefix += 1
            else:
                logger.debug("Skipping %s with shape %s", pass_, shape)

        # post-cleanup goes before fix_functionalization
        # because it requires a functional graph
        self.post_cleanup(graph)
        VllmInductorPass.dump_prefix += 1

        # always run fix_functionalization last
        self.fix_functionalization(graph)
        VllmInductorPass.dump_prefix = None  # Cleanup index

    def configure(self, config: VllmConfig):
        self.pass_config = config.compilation_config.pass_config

        # Set the current vllm config to allow tracing CustomOp instances
        with set_current_vllm_config(config, check_compile=False):
            if self.pass_config.enable_noop:
                self.passes += [NoOpEliminationPass(config)]

            if self.pass_config.enable_sequence_parallelism:
                self.passes += [SequenceParallelismPass(config)]
                if self.pass_config.enable_async_tp:
                    self.passes += [AsyncTPPass(config)]

            if self.pass_config.enable_fi_allreduce_fusion:
                self.passes += [AllReduceFusionPass(config)]

            if self.pass_config.enable_fusion:
                if is_rocm_aiter_enabled():
                    self.passes += [RMSNormAiterQuantFusionPass(config)]
                self.passes += [RMSNormQuantFusionPass(config)]
                self.passes += [ActivationQuantFusionPass(config)]

            if self.pass_config.enable_attn_fusion:
                self.passes += [AttnFusionPass(config)]

            # needs a functional graph
            self.post_cleanup = PostCleanupPass(config)
            self.fix_functionalization = FixFunctionalizationPass(config)

        # [HACK: Bug with Inductor graph partition and torch.compile cache]
        # In PyTorch 2.9, torch.compile has a bug where the graph
        # partition is not taken into account during caching.
        # Because vLLM's Mode.VLLM_COMPILE is the only mode that uses
        # Inductor graph partition, and VLLM_COMPILE implies there
        # is a PostGradPassManager, we put the list of operators to graph
        # partition into the PostGradPassManager's uuid (which
        # then gets incorporated into Inductor's FX graph cache key).
        # Remove this hack whenever torch.compile fixes it.

        # This is the list of operators that vLLM asks Inductor to split.
        self.inductor_splitting_ops = []
        if (
            config.compilation_config.use_inductor_graph_partition
            and config.compilation_config.splitting_ops is not None
        ):
            # Sort them so we're not dependent on the ordering.
            self.inductor_splitting_ops = sorted(
                config.compilation_config.splitting_ops
            )

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
        """
        state = {
            "pass_config": self.pass_config.uuid(),
            "passes": [],
            "inductor_splitting_ops": [],
        }
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.fix_functionalization.uuid())

        # See [HACK: Bug with Inductor graph partition and torch.compile cache]
        state["inductor_splitting_ops"].extend(self.inductor_splitting_ops)

        return InductorPass.hash_dict(state)
