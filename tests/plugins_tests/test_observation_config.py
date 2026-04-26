# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.engine.arg_utils import EngineArgs
from vllm.plugins.observation.interface import ObservationPlugin


class DummyPlugin(ObservationPlugin):
    def __init__(self, vllm_config=None):
        super().__init__(vllm_config=vllm_config)


# noqa: E501
def test_engine_args_observation_plugins():
    plugin = DummyPlugin()
    args = EngineArgs(
        model="facebook/opt-125m",
        observation_plugins=[plugin],
        enforce_eager=True,
    )
    assert len(args.observation_plugins) == 1
    assert args.observation_plugins[0] is plugin


def test_load_observation_plugins_types():
    from unittest.mock import MagicMock
    from vllm.config import VllmConfig
    from vllm.plugins.observation.interface import load_observation_plugins
    import pydantic

    pydantic.dataclasses.rebuild_dataclass(VllmConfig)
    vllm_config = MagicMock(spec=VllmConfig)

    # 1. Test passing an instance
    instance = DummyPlugin()
    manager_inst = load_observation_plugins([instance], vllm_config)
    assert len(manager_inst.plugins) == 1
    assert manager_inst.plugins[0] is instance

    # 2. Test passing a class type
    manager_cls = load_observation_plugins([DummyPlugin], vllm_config)
    assert len(manager_cls.plugins) == 1
    assert isinstance(manager_cls.plugins[0], DummyPlugin)
