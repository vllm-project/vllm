# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch

from vllm.compilation.passes.inductor_pass import (
    CallableInductorPass,
    InductorPass,
    pass_context,
)
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.config import ModelConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.utils import Range
from vllm.platforms import current_platform


# dummy custom pass that doesn't inherit
def simple_callable(graph: torch.fx.Graph):
    pass


# Should fail to add directly to the pass manager
def test_bad_callable():
    config = VllmConfig()

    pass_manager = PostGradPassManager()
    pass_manager.configure(config)

    with pytest.raises(AssertionError):
        pass_manager.add(simple_callable)  # type: ignore[arg-type]


# Pass that inherits from InductorPass
class ProperPass(InductorPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        pass


@pytest.mark.parametrize(
    "callable",
    [
        ProperPass(),
        # Can also wrap callables in CallableInductorPass for compliance
        CallableInductorPass(simple_callable),
        CallableInductorPass(simple_callable, InductorPass.hash_source(__file__)),
    ],
)
def test_pass_manager_uuid(callable):
    # Set the pass context as PassManager uuid uses it
    with pass_context(Range(start=1, end=8)):
        # Some passes need dtype to be set
        config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))

        pass_manager = PostGradPassManager()
        pass_manager.configure(config)

        # Check that UUID is different if the same pass is added 2x
        pass_manager.add(callable)
        uuid1 = pass_manager.uuid()
        pass_manager.add(callable)
        uuid2 = pass_manager.uuid()
        assert uuid1 != uuid2

        # UUID should be the same as the original one,
        # as we constructed in the same way.
        pass_manager2 = PostGradPassManager()
        pass_manager2.configure(config)
        pass_manager2.add(callable)
        assert uuid1 == pass_manager2.uuid()

        # UUID should be different due to config change
        config2 = copy.deepcopy(config)
        config2.compilation_config.pass_config.fuse_norm_quant = (
            not config2.compilation_config.pass_config.fuse_norm_quant
        )
        config2.compilation_config.pass_config.fuse_act_quant = (
            not config2.compilation_config.pass_config.fuse_act_quant
        )
        pass_manager3 = PostGradPassManager()
        pass_manager3.configure(config2)
        pass_manager3.add(callable)
        assert uuid1 != pass_manager3.uuid()


def test_helion_rms_fusion_precedes_aiter_while_aiter_linear_stays_enabled():
    from vllm import envs
    from vllm._aiter_ops import rocm_aiter_ops

    if not (
        current_platform.is_rocm()
        and envs.VLLM_USE_HELION_KERNELS
        and rocm_aiter_ops.is_enabled()
    ):
        pytest.skip("requires ROCm with both Helion and AITER enabled")

    config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    config.compilation_config.pass_config.fuse_norm_quant = True

    pass_manager = PostGradPassManager()
    pass_manager.configure(config)
    pass_names = [type(pass_).__name__ for pass_ in pass_manager.passes]

    assert pass_names.index("RMSNormQuantFusionPass") < pass_names.index(
        "RocmAiterRMSNormQuantFusionPass"
    )
    assert pass_manager.helion_routing is not None
    assert rocm_aiter_ops.is_linear_fp8_enabled()
