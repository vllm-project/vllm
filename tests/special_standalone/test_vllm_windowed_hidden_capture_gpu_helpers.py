# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_script():
    path = Path(__file__).with_name("vllm_windowed_hidden_capture_gpu.py")
    spec = importlib.util.spec_from_file_location("hidden_capture_gpu_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimePreflightTest(unittest.TestCase):
    def test_missing_native_extension_explains_editable_install(self) -> None:
        module = load_script()

        def failed_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="vllm=/checkout/vllm/__init__.py\n",
                stderr="ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'",
            )

        with self.assertRaisesRegex(RuntimeError, "VLLM_USE_PRECOMPILED"):
            module.validate_runtime_environment(run=failed_run)

    def test_dflash_checkpoint_is_rejected_for_dspark(self) -> None:
        module = load_script()

        with self.assertRaisesRegex(ValueError, "Qwen3DSparkModel"):
            module.validate_dspark_architectures(["DFlashDraftModel"])

    def test_qwen3_dspark_checkpoint_is_accepted(self) -> None:
        module = load_script()

        module.validate_dspark_architectures(["Qwen3DSparkModel"])

    def test_environment_uses_v2_without_removed_v1_switch(self) -> None:
        module = load_script()

        with patch.dict(os.environ, {}, clear=True):
            module.configure_environment()

            self.assertNotIn("VLLM_USE_V1", os.environ)
            self.assertEqual(os.environ["VLLM_USE_V2_MODEL_RUNNER"], "1")

    def test_engine_args_disable_async_scheduling_for_capture(self) -> None:
        module = load_script()
        args = SimpleNamespace(
            target_model="target",
            dtype="bfloat16",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=0.8,
            trust_remote_code=False,
            seed=7,
            max_model_len=None,
        )

        engine_kwargs = module.build_engine_kwargs(
            args, eager=False, speculative_config={"method": "dspark"}
        )

        self.assertIs(engine_kwargs["async_scheduling"], False)

    def test_aux_layer_ids_use_runner_dspark_semantics(self) -> None:
        module = load_script()
        speculative_config = SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(target_layer_ids=[7, 15, 23])
            )
        )
        engine = SimpleNamespace(
            vllm_config=SimpleNamespace(speculative_config=speculative_config)
        )

        layer_ids = module.resolve_aux_layer_ids(
            engine,
            None,
            get_aux_layers=lambda config: tuple(
                layer_id + 1
                for layer_id in config.draft_model_config.hf_config.target_layer_ids
            ),
        )

        self.assertEqual(layer_ids, (8, 16, 24))

    def test_cli_layout_maps_to_capture_result_contract(self) -> None:
        module = load_script()

        self.assertEqual(module.capture_layout("dflash_aux"), "dflash_aux")
        self.assertEqual(module.capture_layout("dflash_aux_plus_last"), "aux_final")

    def test_missing_result_metadata_fails_before_model_load(self) -> None:
        module = load_script()
        capture_module = SimpleNamespace(
            HiddenStateCaptureResult=SimpleNamespace(__dataclass_fields__={}),
            __file__="/old-checkout/vllm/v1/hidden_state_capture.py",
        )

        with self.assertRaisesRegex(RuntimeError, "out of sync.*layer_ids"):
            module.validate_capture_result_contract(capture_module)


if __name__ == "__main__":
    unittest.main()
