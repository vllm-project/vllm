# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization.quark.utils import deep_compare


def test_cuda_empty_vs_unset_configs(monkeypatch: pytest.MonkeyPatch):
    """Test that configs created with normal (untouched) CUDA_VISIBLE_DEVICES
    and CUDA_VISIBLE_DEVICES="" are equivalent. This ensures consistent
    behavior regardless of whether GPU visibility is disabled via empty string
    or left in its normal state.
    """

    def create_config():
        engine_args = EngineArgs(
            model="deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True
        )
        return engine_args.create_engine_config()

    # Create config with CUDA_VISIBLE_DEVICES set normally
    normal_config = create_config()

    # Create config with CUDA_VISIBLE_DEVICES=""
    with monkeypatch.context() as m:
        m.setenv("CUDA_VISIBLE_DEVICES", "")
        empty_config = create_config()

    normal_config_dict = vars(normal_config)
    empty_config_dict = vars(empty_config)

    # Remove instance_id before comparison as it's expected to be different
    normal_config_dict.pop("instance_id", None)
    empty_config_dict.pop("instance_id", None)

    assert deep_compare(normal_config_dict, empty_config_dict), (
        'Configs with normal CUDA_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES=""'
        " should be equivalent"
    )


def test_ray_runtime_env(monkeypatch: pytest.MonkeyPatch):
    # In testing, this method needs to be nested inside as ray does not
    # see the test module.
    def create_config():
        engine_args = EngineArgs(
            model="deepseek-ai/DeepSeek-V2-Lite", trust_remote_code=True
        )
        return engine_args.create_engine_config()

    config = create_config()
    parallel_config = config.parallel_config
    assert parallel_config.ray_runtime_env is None

    import ray

    ray.init()

    runtime_env = {
        "env_vars": {
            "TEST_ENV_VAR": "test_value",
            # In future ray versions, this will be default, so when setting a
            # task or actor with num_gpus=None/0, the visible devices env var
            # won't be overridden resulting in no GPUs being visible on a gpu
            # machine.
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        },
    }

    config_ref = ray.remote(create_config).options(runtime_env=runtime_env).remote()

    config = ray.get(config_ref)
    parallel_config = config.parallel_config
    assert parallel_config.ray_runtime_env is not None
    assert (
        parallel_config.ray_runtime_env.env_vars().get("TEST_ENV_VAR") == "test_value"
    )

    ray.shutdown()


def test_unrecognized_env(monkeypatch):
    import os

    from vllm.envs import environment_variables

    # Remove any existing unrecognized VLLM env vars that might interfere
    for env in list(os.environ):
        if env.startswith("VLLM_") and env not in environment_variables:
            monkeypatch.delenv(env, raising=False)

    # Test that if fail_on_environ_validation is True, then an error
    # is raised when an unrecognized vLLM environment variable is set
    monkeypatch.setenv("VLLM_UNRECOGNIZED_ENV_VAR", "some_value")
    engine_args = EngineArgs(
        fail_on_environ_validation=True,
    )
    with pytest.raises(ValueError, match="Unknown vLLM environment variable detected"):
        engine_args.create_engine_config()

    # Test that if fail_on_environ_validation is False, then no error is raised
    engine_args = EngineArgs()
    engine_args.create_engine_config()

    # Test that when the unrecognized env var is removed, no error is raised
    monkeypatch.delenv("VLLM_UNRECOGNIZED_ENV_VAR")
    engine_args = EngineArgs(
        fail_on_environ_validation=True,
    )
    engine_args.create_engine_config()


def test_get_diff_sampling_param_includes_penalties():
    """``--override-generation-config`` must apply presence_penalty and
    frequency_penalty as server-side defaults, like the other sampling
    params. Regression test for the whitelist silently dropping them.

    Called unbound with a lightweight stand-in so no model is loaded:
    generation_config="vllm" makes the method skip try_get_generation_config
    and read only override_generation_config.
    """
    from types import SimpleNamespace

    from vllm.config.model import ModelConfig

    fake = SimpleNamespace(
        generation_config="vllm",
        override_generation_config={
            "presence_penalty": 1.5,
            "frequency_penalty": 0.5,
            "temperature": 0.6,
        },
    )

    diff = ModelConfig.get_diff_sampling_param(fake)

    assert diff["presence_penalty"] == 1.5
    assert diff["frequency_penalty"] == 0.5
    assert diff["temperature"] == 0.6


def test_generation_config_penalties_survive_diff_sampling_filter():
    """``generation_config.json`` penalties must survive the diff-sampling
    whitelist (regression for #50767, auto path).

    Covers the ``generation_config="auto"`` branch, where the override
    dict is empty and the values come from ``try_get_generation_config``.
    """
    from types import SimpleNamespace

    from vllm.config.model import ModelConfig

    fake = SimpleNamespace(
        generation_config="auto",
        override_generation_config={},
        try_get_generation_config=lambda: {
            "presence_penalty": 1.1,
            "frequency_penalty": 0.7,
            "temperature": 0.9,
        },
    )

    diff = ModelConfig.get_diff_sampling_param(fake)

    assert diff["presence_penalty"] == 1.1
    assert diff["frequency_penalty"] == 0.7
    assert diff["temperature"] == 0.9
