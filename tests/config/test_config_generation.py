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


def test_mamba2_without_architectures_field_runs_config_hook():
    """Checkpoints whose ``config.json`` omits the top-level ``architectures``
    field (e.g. ``AntonV/mamba2-130m-hf``) get the base HF arch name
    ``Mamba2Model`` injected by vLLM. The per-arch config hook that sets
    ``mamba_block_size`` is keyed by the resolved arch ``Mamba2ForCausalLM``, so
    dispatching it by the un-normalized ``Mamba2Model`` used to skip it and leave
    ``mamba_block_size`` unset, crashing engine core init with
    ``AssertionError: mamba_block_size is not None``.
    """
    engine_args = EngineArgs(
        model="AntonV/mamba2-130m-hf",
        max_model_len=2048,
        enforce_eager=True,
    )
    vllm_config = engine_args.create_engine_config()
    assert vllm_config.cache_config.mamba_block_size is not None
