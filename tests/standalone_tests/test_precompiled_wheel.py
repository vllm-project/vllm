# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise setup-time wheel selection without importing torch or running setup."""

import ast
import logging
import os
import subprocess
import sysconfig
import tempfile
import unittest
from contextlib import chdir
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from zipfile import ZipFile

# setup.py executes setup() on import; load only the wheel utility code.
setup_path = Path(__file__).resolve().parents[2] / "setup.py"
nodes = [
    node
    for node in ast.parse(setup_path.read_text()).body
    if (
        isinstance(node, ast.Import)
        and all(
            alias.name in {"os", "re", "shutil", "subprocess", "sysconfig"}
            for alias in node.names
        )
    )
    or (
        isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name
        in {
            "get_precompiled_rust_extension_paths",
            "has_compatible_rust_extension_suffix",
            "is_freethreaded",
            "precompiled_wheel_utils",
        }
    )
    or (
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "PRECOMPILED_RUST_EXTENSION_MEMBER_REGEX"
            for target in node.targets
        )
    )
]
namespace: dict[str, Any] = {
    "Path": Path,
    "logger": logging.getLogger("setup"),
    "envs": SimpleNamespace(VLLM_DOCKER_BUILD_CONTEXT=False, VLLM_USE_PRECOMPILED=True),
}
exec(
    compile(ast.Module(body=nodes, type_ignores=[]), str(setup_path), "exec"),
    namespace,
)
wheel_utils = namespace["precompiled_wheel_utils"]
WHEEL = {
    "package_name": "vllm",
    "platform_tag": "manylinux_x86_64",
    "path": "wheel.whl",
    "filename": "wheel.whl",
}
NOT_FOUND = HTTPError("url", 404, "missing", Message(), None)


class TestPublishedWheelSearch(unittest.TestCase):
    """Walk back from the merge-base in a real Git repo; only the index is mocked."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.repo)
        self.git("init", "-q")
        self.published = {self.commit("vllm/a.py")}

    def git(self, *args: str) -> str:
        identity = ["-c", "user.name=t", "-c", "user.email=t@t"]
        return subprocess.check_output(["git", *identity, *args], text=True).strip()

    def commit(self, path: str) -> str:
        file = self.repo / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(file.read_text() + "x" if file.exists() else "x")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", path)
        return self.git("rev-parse", "HEAD")

    def search(self, base: str, arch: str = "x86_64") -> str:
        def fetch(commit, variant):
            if commit not in self.published:
                raise NOT_FOUND
            return [WHEEL], "url"

        with patch.object(wheel_utils, "fetch_metadata_for_variant", fetch):
            return wheel_utils.find_published_wheel_commit(base, "cu130", arch)

    def test_published_base_is_used_despite_local_native_edits(self):
        base = self.commit("csrc/ops.cpp")
        self.published.add(base)
        self.commit("csrc/ops.cpp")
        (self.repo / "pyproject.toml").write_text("dirty")
        self.assertEqual(self.search(base), base)

    def test_publication_lag_uses_nearest_published_ancestor(self):
        published = self.commit("vllm/b.py")
        self.published.add(published)
        self.commit("vllm/c.py")
        self.assertEqual(self.search(self.commit("vllm/d.py")), published)

    def test_compiled_changes_since_wheel_are_rejected(self):
        (published,) = self.published
        for path in wheel_utils.WHEEL_BLOCKING_PATHS:
            with self.subTest(path=path):
                self.git("reset", "-q", "--hard", published)
                file = path if "." in path else f"{path}/file"
                with self.assertRaisesRegex(ValueError, file):
                    self.search(self.commit(file))

    def test_rust_changes_since_wheel_only_warn(self):
        (published,) = self.published
        base = self.commit("rust/src/main.rs")
        with self.assertLogs("setup", "WARNING") as logs:
            self.assertEqual(self.search(base), published)
        self.assertIn("rust/src/main.rs", logs.output[0])

    def test_wrong_architecture_is_not_a_match(self):
        with self.assertRaisesRegex(ValueError, "No published precompiled wheel"):
            self.search("HEAD", arch="aarch64")

    def test_search_stops_at_depth_limit(self):
        with patch.object(wheel_utils, "WHEEL_SEARCH_DEPTH", 2):
            self.commit("vllm/b.py")
            with self.assertRaisesRegex(ValueError, "within 2 commits"):
                self.search(self.commit("vllm/c.py"))

    def test_network_and_server_errors_do_not_trigger_walkback(self):
        for error in (
            URLError("offline"),
            HTTPError("url", 503, "busy", Message(), None),
        ):
            with (
                self.subTest(error=error),
                patch.object(
                    wheel_utils, "fetch_metadata_for_variant", side_effect=error
                ) as fetch,
                self.assertRaises(type(error)),
            ):
                wheel_utils.find_published_wheel_commit("HEAD", "cu130", "x86_64")
            fetch.assert_called_once()


class TestDetermineWheelUrl(unittest.TestCase):
    def determine(self, base: str, **env: str):
        with (
            patch.dict(
                os.environ,
                {"VLLM_PRECOMPILED_WHEEL_VARIANT": "cu130", **env},
                clear=True,
            ),
            patch("platform.machine", return_value="x86_64"),
            patch.object(wheel_utils, "is_rocm_system", return_value=False),
            patch.object(
                wheel_utils, "get_base_commit_in_main_branch", return_value=base
            ),
            patch.object(
                wheel_utils,
                "fetch_metadata_for_variant",
                return_value=([WHEEL], "https://wheels.test/"),
            ) as fetch,
            patch.object(
                wheel_utils, "find_published_wheel_commit", return_value="older"
            ) as search,
        ):
            url = wheel_utils.determine_wheel_url()
        return url, fetch, search

    def test_automatic_install_downloads_the_searched_commit(self):
        url, fetch, search = self.determine("base")
        self.assertEqual(url, ("https://wheels.test/wheel.whl", "wheel.whl"))
        search.assert_called_once_with("base", "cu130", "x86_64")
        fetch.assert_called_once_with("older", "cu130")

    def test_search_is_skipped_when_it_cannot_apply(self):
        cases = {
            "nightly fallback": ("nightly", {}, {}),
            "explicit commit": (
                "base",
                {"VLLM_PRECOMPILED_WHEEL_COMMIT": "a" * 40},
                {},
            ),
            "rust only": ("base", {}, {"VLLM_USE_PRECOMPILED": False}),
            "docker": ("base", {}, {"VLLM_DOCKER_BUILD_CONTEXT": True}),
        }
        for name, (base, env, flags) in cases.items():
            with self.subTest(name), patch.dict(vars(namespace["envs"]), flags):
                _, fetch, search = self.determine(base, **env)
                search.assert_not_called()
                commit = env.get("VLLM_PRECOMPILED_WHEEL_COMMIT", base)
                fetch.assert_called_once_with(commit, "cu130")

    def test_explicit_wheel_location_bypasses_selection(self):
        for location in ("/tmp/local.whl", "https://example.com/custom.whl"):
            with self.subTest(location=location):
                url, fetch, search = self.determine(
                    "base", VLLM_PRECOMPILED_WHEEL_LOCATION=location
                )
                self.assertEqual(url, (location, None))
                fetch.assert_not_called()
                search.assert_not_called()


class TestRustExtensionAbi(unittest.TestCase):
    def test_local_and_extracted_extensions_match_the_interpreter_abi(self):
        abi3 = "_rust_turn_detection.abi3.so"
        python_314t = "_rust_turn_detection.cpython-314t-x86_64-linux-gnu.so"
        python_313t = "_rust_turn_detection.cpython-313t-x86_64-linux-gnu.so"

        for gil_disabled, ext_suffix, compatible in [
            (False, ".cpython-314-x86_64-linux-gnu.so", abi3),
            (True, ".cpython-314t-x86_64-linux-gnu.so", python_314t),
        ]:
            config = {"Py_GIL_DISABLED": gil_disabled, "EXT_SUFFIX": ext_suffix}
            with (
                self.subTest(gil_disabled=gil_disabled),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(sysconfig, "get_config_var", side_effect=config.get),
            ):
                root = Path(tmp)
                package = root / "vllm"
                package.mkdir()
                for name in (abi3, python_314t, python_313t):
                    (package / name).touch()

                with self.subTest(source="local"), patch.dict(namespace, ROOT_DIR=root):
                    self.assertEqual(
                        namespace["get_precompiled_rust_extension_paths"](),
                        [package / compatible],
                    )

                wheel = root / "vllm.whl"
                with ZipFile(wheel, "w") as archive:
                    for name in ("vllm-rs", abi3, python_314t, python_313t):
                        archive.writestr(f"vllm/{name}", name)

                extracted = root / "extracted"
                extracted.mkdir()
                with chdir(extracted):
                    package_data = wheel_utils.extract_precompiled_and_patch_package(
                        str(wheel),
                        None,
                        extract_extensions=False,
                        extract_rust_frontend=True,
                    )
                with self.subTest(source="wheel"):
                    self.assertEqual(package_data, {"vllm": ["vllm-rs", compatible]})
                    self.assertEqual(
                        {path.name for path in (extracted / "vllm").iterdir()},
                        {"vllm-rs", compatible},
                    )


if __name__ == "__main__":
    unittest.main()
