# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import multiprocessing
import sys
import types
from unittest.mock import patch

from vllm.config import DeviceConfig, VllmConfig
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value


def test_mp_reducer():
    """
    Test that _reduce_config reducer is registered when AsyncLLM is instantiated
    without transformers_modules. This is a regression test for
    https://github.com/vllm-project/vllm/pull/18640.
    """

    # Ensure transformers_modules is not in sys.modules
    if "transformers_modules" in sys.modules:
        del sys.modules["transformers_modules"]

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    with patch("multiprocessing.reducer.register") as mock_register:
        engine_args = AsyncEngineArgs(
            model="facebook/opt-125m",
            max_model_len=32,
            gpu_memory_utilization=0.1,
            disable_log_stats=True,
        )

        async_llm = AsyncLLM.from_engine_args(
            engine_args,
            start_engine_loop=False,
        )

        assert mock_register.called, (
            "multiprocessing.reducer.register should have been called"
        )

        vllm_config_registered = False
        for call_args in mock_register.call_args_list:
            # Verify that a reducer for VllmConfig was registered
            if len(call_args[0]) >= 2 and call_args[0][0] == VllmConfig:
                vllm_config_registered = True

                reducer_func = call_args[0][1]
                assert callable(reducer_func), "Reducer function should be callable"
                break

        assert vllm_config_registered, (
            "VllmConfig should have been registered to multiprocessing.reducer"
        )

        async_llm.shutdown()


def test_mp_reducer_serializes_remote_config_with_config_mapping(tmp_path):
    module_name = "transformers_modules.synthetic.configuration_remote"
    module_path = tmp_path / "configuration_remote.py"
    module_path.write_text(
        "from transformers.models.auto import CONFIG_MAPPING\n"
        "class RemoteConfig:\n"
        "    def __init__(self):\n"
        "        self.value = 'original'\n"
        "    def mapping(self):\n"
        "        return CONFIG_MAPPING\n"
    )

    modules = {}
    for package_name in ("transformers_modules", "transformers_modules.synthetic"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        modules[package_name] = package

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    remote_module = importlib.util.module_from_spec(spec)
    modules[module_name] = remote_module

    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(remote_module)
        try:
            maybe_register_config_serialize_by_value()

            vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
            vllm_config.model_config = remote_module.RemoteConfig()  # type: ignore[assignment]
            payload = multiprocessing.reduction.ForkingPickler.dumps(vllm_config)
            restored = multiprocessing.reduction.ForkingPickler.loads(payload)

            assert restored.model_config.value == "original"
            assert restored.model_config.mapping()["llama"].model_type == "llama"
        finally:
            cloudpickle_module = sys.modules["cloudpickle"]
            cloudpickle_module.unregister_pickle_by_value(  # type: ignore[attr-defined]
                modules["transformers_modules"]
            )
