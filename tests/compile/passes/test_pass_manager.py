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
from vllm.compilation.passes.pass_manager import PassManager, PostGradPassManager
from vllm.config import ModelConfig, VllmConfig
from vllm.config.utils import Range


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


def test_post_grad_is_pass_manager():
    """PostGradPassManager should be a thin subclass of the shared base."""
    assert issubclass(PostGradPassManager, PassManager)


def test_uuid_includes_always_on_passes():
    """The post-grad uuid must hash the always-on utility passes that run in
    __call__ (post_cleanup, ir_lowering, clone_elimination, post_cleanup,
    fix_functionalization), not just the configurable `self.passes`. This pins
    the contract that the base `_uuid_state` is extended by the subclass, so a
    future refactor that drops the always-on passes from the cache key fails
    here instead of silently reusing a stale Inductor cache entry."""
    with pass_context(Range(start=1, end=8)):
        config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))
        pass_manager = PostGradPassManager()
        pass_manager.configure(config)

        full_uuid = pass_manager.uuid()
        # The base-only state omits the always-on utility passes; the
        # subclass uuid must differ from hashing that base state alone.
        base_state = PassManager._uuid_state(pass_manager)
        assert full_uuid != InductorPass.hash_dict(base_state)

        # And the always-on pass UUIDs must each appear in the hashed state.
        full_state = pass_manager._uuid_state()
        for utility in (
            pass_manager.post_cleanup,
            pass_manager.ir_lowering,
            pass_manager.clone_elimination,
            pass_manager.fix_functionalization,
        ):
            assert utility.uuid() in full_state["passes"]
