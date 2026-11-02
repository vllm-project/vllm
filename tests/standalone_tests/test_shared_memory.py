# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Real stdlib shared-memory ownership, without importing the GPU runtime."""

import importlib.util
import subprocess
import sys
import unittest
from contextlib import ExitStack
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from unittest.mock import patch

HELPER = (
    Path(__file__).resolve().parents[2]
    / "vllm/distributed/device_communicators/shm_utils.py"
)
spec = importlib.util.spec_from_file_location("vllm_test_shm_utils", HELPER)
assert spec is not None and spec.loader is not None
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


class TestSharedMemoryAttachment(unittest.TestCase):
    def test_nonowner_can_close_without_unlinking_owner(self):
        owner = shared_memory.SharedMemory(create=True, size=1)
        try:
            owner.buf[0] = 41
            register = resource_tracker.register
            attached = helper.attach_shared_memory(owner.name)
            try:
                self.assertEqual(attached.buf[0], 41)
                attached.buf[0] = 42
                self.assertEqual(owner.buf[0], 42)
                self.assertIs(resource_tracker.register, register)
            finally:
                attached.close()
            again = helper.attach_shared_memory(owner.name)
            again.close()
        finally:
            owner.close()
            owner.unlink()
        with self.assertRaises(FileNotFoundError):
            helper.attach_shared_memory(owner.name)

    def test_independent_process_does_not_unlink_owner(self):
        owner = shared_memory.SharedMemory(create=True, size=1)
        try:
            owner.buf[0] = 17
            child = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('child_shm_utils', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
shm = module.attach_shared_memory(sys.argv[2])
assert shm.buf[0] == 17
shm.close()
"""
            done = subprocess.run(
                [sys.executable, "-I", "-B", "-c", child, str(HELPER), owner.name],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            self.assertNotIn("resource_tracker", done.stderr)
            attached = helper.attach_shared_memory(owner.name)
            self.assertEqual(attached.buf[0], 17)
            attached.close()
        finally:
            owner.close()
            owner.unlink()


class TestQueueBackendSelection(unittest.TestCase):
    """Policy tests mock runtime discovery, not queue or native operations."""

    def supported_runtime(self):
        stack = ExitStack()
        stack.enter_context(patch.object(helper.sys, "platform", "linux"))
        stack.enter_context(
            patch.object(helper.platform, "machine", return_value="aarch64")
        )
        stack.enter_context(
            patch.object(helper.sys, "_is_gil_enabled", return_value=True, create=True)
        )
        return stack

    def test_supported_implicit_mode_selects_futex(self):
        with (
            self.supported_runtime(),
            patch.object(helper.importlib, "import_module", return_value=object()),
        ):
            self.assertEqual(helper.resolve_shm_wait_backend("auto"), ("futex", None))

    def test_stock_opt_out_needs_no_native_import(self):
        with patch.object(
            helper.importlib, "import_module", side_effect=AssertionError
        ):
            self.assertEqual(helper.resolve_shm_wait_backend("stock"), ("stock", None))

    def test_missing_optional_extension_is_the_only_import_fallback(self):
        missing = ModuleNotFoundError(name="vllm._asa_shmwait")
        with (
            self.supported_runtime(),
            patch.object(helper.importlib, "import_module", side_effect=missing),
        ):
            backend, reason = helper.resolve_shm_wait_backend("auto")
            self.assertEqual(backend, "stock")
            self.assertIn("vllm._asa_shmwait", reason)
            with self.assertRaises(ModuleNotFoundError):
                helper.resolve_shm_wait_backend("futex")

    def test_broken_native_installations_are_not_hidden(self):
        errors = (
            ModuleNotFoundError(name="transitive_dependency"),
            ImportError("extension ABI mismatch"),
            OSError("loader failure"),
        )
        for error in errors:
            with (
                self.subTest(error=type(error).__name__),
                self.supported_runtime(),
                patch.object(helper.importlib, "import_module", side_effect=error),
                self.assertRaises(type(error)),
            ):
                helper.resolve_shm_wait_backend("auto")

    def test_unsupported_runtime_falls_back_only_for_implicit_mode(self):
        for target, name, value in (
            (helper.sys, "platform", "darwin"),
            (helper.platform, "machine", lambda: "s390x"),
            (helper.sys, "_is_gil_enabled", lambda: False),
        ):
            with (
                self.subTest(capability=name),
                self.supported_runtime(),
                patch.object(target, name, value),
                patch.object(
                    helper.importlib, "import_module", side_effect=AssertionError
                ),
            ):
                backend, reason = helper.resolve_shm_wait_backend("auto")
                self.assertEqual(backend, "stock")
                self.assertIn("requires", reason)
                with self.assertRaises(ValueError):
                    helper.resolve_shm_wait_backend("futex")

    def test_invalid_mode_is_not_a_fallback(self):
        with self.assertRaisesRegex(ValueError, "auto, stock, or futex"):
            helper.resolve_shm_wait_backend("typo")


if __name__ == "__main__":
    unittest.main()
