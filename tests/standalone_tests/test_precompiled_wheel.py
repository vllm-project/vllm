# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise setup-time wheel selection without importing torch or running setup."""

import ast
import os
import subprocess
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError

# setup.py executes setup() on import; load only the wheel utility class.
setup_path = Path(__file__).resolve().parents[2] / "setup.py"
node = next(
    node
    for node in ast.parse(setup_path.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name == "precompiled_wheel_utils"
)
namespace: dict[str, Any] = {
    "subprocess": subprocess,
    "os": os,
    "envs": SimpleNamespace(VLLM_DOCKER_BUILD_CONTEXT=False, VLLM_USE_PRECOMPILED=True),
}
exec(
    compile(ast.Module(body=[node], type_ignores=[]), str(setup_path), "exec"),
    namespace,
)
wheel_utils = namespace["precompiled_wheel_utils"]
WHEEL = {"package_name": "vllm", "platform_tag": "manylinux_x86_64"}


class TestCompatibleWheel(unittest.TestCase):
    def select(self):
        return wheel_utils.find_compatible_wheel("base", "cu130", "x86_64")

    def test_automatic_install_uses_compatible_wheel(self):
        wheel = dict(WHEEL, path="wheel.whl", filename="wheel.whl")
        with (
            patch.dict(
                os.environ, {"VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130"}, clear=True
            ),
            patch("platform.machine", return_value="x86_64"),
            patch.object(wheel_utils, "is_rocm_system", return_value=False),
            patch.object(
                wheel_utils, "get_base_commit_in_main_branch", return_value="base"
            ),
            patch.object(
                wheel_utils,
                "find_compatible_wheel",
                return_value=([wheel], "https://wheels.test/"),
            ) as select,
        ):
            self.assertEqual(
                wheel_utils.determine_wheel_url(),
                ("https://wheels.test/wheel.whl", "wheel.whl"),
            )
        select.assert_called_once_with("base", "cu130", "x86_64")

    def test_explicit_wheel_location_bypasses_selection(self):
        for location in (
            "/tmp/local.whl",
            "./wheels/local.whl",
            "../local.whl",
            "https://example.com/custom.whl",
            "https://wheels.vllm.ai/custom.whl",
        ):
            with (
                self.subTest(location=location),
                patch.dict(
                    os.environ,
                    {
                        "VLLM_USE_PRECOMPILED": "1",
                        "VLLM_PRECOMPILED_WHEEL_LOCATION": location,
                    },
                    clear=True,
                ),
                patch.object(wheel_utils, "find_compatible_wheel") as select,
            ):
                self.assertEqual(wheel_utils.determine_wheel_url(), (location, None))
            select.assert_not_called()

    def test_rust_only_install_bypasses_compiled_source_check(self):
        wheel = dict(WHEEL, path="wheel.whl", filename="wheel.whl")
        with (
            patch.dict(
                os.environ, {"VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130"}, clear=True
            ),
            patch.object(namespace["envs"], "VLLM_USE_PRECOMPILED", False),
            patch("platform.machine", return_value="x86_64"),
            patch.object(wheel_utils, "is_rocm_system", return_value=False),
            patch.object(
                wheel_utils, "get_base_commit_in_main_branch", return_value="base"
            ),
            patch.object(
                wheel_utils,
                "fetch_metadata_for_variant",
                return_value=([wheel], "https://wheels.test/"),
            ),
            patch.object(wheel_utils, "find_compatible_wheel") as select,
        ):
            self.assertEqual(
                wheel_utils.determine_wheel_url(),
                ("https://wheels.test/wheel.whl", "wheel.whl"),
            )
        select.assert_not_called()

    def test_publication_lag_uses_nearest_available_ancestor(self):
        with (
            patch.object(
                subprocess, "check_output", side_effect=["base\nolder\n", ""]
            ) as git,
            patch.object(
                wheel_utils,
                "fetch_metadata_for_variant",
                side_effect=[
                    HTTPError("url", 404, "missing", Message(), None),
                    ([WHEEL], "url"),
                ],
            ) as fetch,
        ):
            self.assertEqual(self.select(), ([WHEEL], "url"))
        self.assertEqual(fetch.call_args_list[-1].args, ("older", "cu130"))
        self.assertEqual(git.call_args_list[-1].args[0][3], "older")

    def test_compiled_changes_from_wheel_to_checkout_are_rejected(self):
        for path in ("csrc/ops.cpp", "rust/src/main.rs", "setup.py"):
            with (
                self.subTest(path=path),
                patch.object(subprocess, "check_output", side_effect=["base\n", path]),
                patch.object(
                    wheel_utils,
                    "fetch_metadata_for_variant",
                    return_value=([WHEEL], "url"),
                ),
                self.assertRaisesRegex(ValueError, "Build vLLM from source"),
            ):
                self.select()

    def test_wrong_architecture_is_skipped(self):
        with (
            patch.object(subprocess, "check_output", side_effect=["base\nolder\n", ""]),
            patch.object(
                wheel_utils,
                "fetch_metadata_for_variant",
                side_effect=[([], "wrong-arch"), ([WHEEL], "older")],
            ),
        ):
            self.assertEqual(self.select(), ([WHEEL], "older"))

    def test_exhausted_history_reports_no_published_wheel(self):
        with (
            patch.object(subprocess, "check_output", return_value="base\n"),
            patch.object(
                wheel_utils, "fetch_metadata_for_variant", return_value=([], "url")
            ),
            self.assertRaisesRegex(ValueError, "No published precompiled wheel"),
        ):
            self.select()

    def test_network_and_server_errors_do_not_trigger_walkback(self):
        for error in (
            URLError("offline"),
            HTTPError("url", 503, "busy", Message(), None),
        ):
            with (
                self.subTest(error=error),
                patch.object(subprocess, "check_output", return_value="base\nolder\n"),
                patch.object(
                    wheel_utils, "fetch_metadata_for_variant", side_effect=error
                ) as fetch,
                self.assertRaises(type(error)),
            ):
                self.select()
            fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
