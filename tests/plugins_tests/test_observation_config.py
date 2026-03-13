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
