# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the depyf ``load_by_key_path`` patch in compilation/depyf_compat.py.

depyf replaces ``torch._inductor.codecache.PyCodeCache.load_by_key_path`` with
a hook that dumps each generated kernel before loading it. That hook's
signature is frozen at ``(key, path, linemap, attrs)`` while Inductor also
passes ``set_sys_modules``, so the patch re-wraps it to forward keywords depyf
does not know about.

End-to-end coverage lives in
``tests/compile/fullgraph/test_full_graph.py::test_depyf_integration``, which
needs a GPU and a cold compile cache. These tests pin the same behaviour by
driving ``PyCodeCache`` directly, so they catch the incompatibility in seconds
on CPU -- which is where it should have been caught in the first place.
"""

import sys

import depyf.explain.enable_debugging as enable_debugging
import torch._inductor.codecache as codecache

from vllm.compilation import depyf_compat

# Captured at import, which pytest does before any test can install the patch.
PRISTINE_HOOK = enable_debugging.patched_load_by_key_path


class TestPatchedHook:
    """The hook has to survive the call *and* keep both sides' behaviour."""

    def test_load_by_key_path_accepts_set_sys_modules(self, tmp_path):
        with depyf_compat.prepare_debug(str(tmp_path)):
            key, path = codecache.PyCodeCache.write("value = 1\n")
            module = codecache.PyCodeCache.load_by_key_path(
                key, path, set_sys_modules=False
            )

        assert module.value == 1

    def test_set_sys_modules_is_forwarded_not_swallowed(self, tmp_path):
        # Dropping the keyword would also make the call succeed, but it would
        # re-register benchmark modules in sys.modules and bring back the
        # exhaustion that pytorch#184285 fixed.
        with depyf_compat.prepare_debug(str(tmp_path)):
            key, path = codecache.PyCodeCache.write("value = 2\n")
            module = codecache.PyCodeCache.load_by_key_path(
                key, path, set_sys_modules=False
            )
            registered = module.__name__ in sys.modules

        sys.modules.pop(module.__name__, None)
        assert not registered

    def test_depyf_still_dumps_the_generated_kernel(self, tmp_path):
        # The patch reimplements depyf's hook body rather than wrapping it,
        # so this is what catches us drifting away from what depyf does.
        with depyf_compat.prepare_debug(str(tmp_path)):
            key, path = codecache.PyCodeCache.write("value = 3\n")
            codecache.PyCodeCache.load_by_key_path(key, path, set_sys_modules=False)

        dumped = list(tmp_path.rglob("*.kernel_*.py"))
        assert dumped, f"depyf wrote no kernel dump into {tmp_path}"
        assert "value = 3" in dumped[0].read_text()


class TestPatchApplication:
    def test_patch_is_still_needed(self):
        assert not depyf_compat._hook_accepts_inductor_keywords(PRISTINE_HOOK), (
            "depyf's load_by_key_path now accepts everything Inductor passes; "
            "drop vllm/compilation/depyf_compat.py and this test"
        )

    def test_patch_is_skipped_when_unnecessary(self):
        def accepts_anything(*args, **kwargs):
            pass

        assert depyf_compat._hook_accepts_inductor_keywords(accepts_anything)

    def test_patch_is_idempotent(self, tmp_path):
        with depyf_compat.prepare_debug(str(tmp_path)):
            first = enable_debugging.patched_load_by_key_path
        with depyf_compat.prepare_debug(str(tmp_path)):
            assert enable_debugging.patched_load_by_key_path is first
