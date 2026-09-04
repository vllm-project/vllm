# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Module-level (global) storage manifest.

Globals are the escape route neither copy-back nor the reload arena covers:
no walk rooted at the model reaches them, yet a captured graph bakes their
addresses. Both signals are exercised here, because each misses what the
other catches -- ``expired`` needs the old storage to actually die, and the
Machete reproduction on H200 showed 88/88 tensors rebound with 0/88
storages freed.

Runs on CPU: storage lifetime and identity are device-independent even
though the consumer (graph replay) is not.
"""

import sys
import types

import torch

from vllm.model_executor.reload_manifest import (
    GlobalStorageManifest, collect_module_level_tensors)

PREFIX = "vllm.model_executor.fake_global_holder"


def _install(**attrs) -> types.ModuleType:
    """A throwaway module under a scanned prefix, standing in for e.g.
    tokenspeed_mla's ``_g_workspace``."""
    module = types.ModuleType(PREFIX)
    for k, v in attrs.items():
        setattr(module, k, v)
    sys.modules[PREFIX] = module
    return module


def _record(**kwargs) -> GlobalStorageManifest:
    manifest = GlobalStorageManifest()
    manifest.record(prefixes=(PREFIX, ), require_cuda=False, **kwargs)
    return manifest


def _cleanup():
    sys.modules.pop(PREFIX, None)


class TestCollection:

    def teardown_method(self):
        _cleanup()

    def test_finds_bare_module_attribute(self):
        _install(_G_TENSOR=torch.ones(4))
        found = collect_module_level_tensors((PREFIX, ), require_cuda=False)
        assert f"{PREFIX}._G_TENSOR" in found

    def test_finds_tensors_inside_a_cache_dict(self):
        # the _g_workspace / _TRITON_BUFFER_CACHE shape
        _install(_CACHE={"cuda:0": torch.ones(4), "cuda:1": torch.ones(4)})
        found = collect_module_level_tensors((PREFIX, ), require_cuda=False)
        assert f"{PREFIX}._CACHE['cuda:0']" in found
        assert f"{PREFIX}._CACHE['cuda:1']" in found

    def test_paths_are_stable_across_scans(self):
        """`moved` is only computable if a path re-resolves to the same
        logical slot on the second scan."""
        _install(_CACHE={"k": torch.ones(4)}, _LIST=[torch.ones(2)])
        first = set(collect_module_level_tensors((PREFIX, ),
                                                 require_cuda=False))
        second = set(collect_module_level_tensors((PREFIX, ),
                                                  require_cuda=False))
        assert first == second

    def test_skips_classes_and_callables(self):
        class Holder:
            pass

        _install(SomeClass=Holder, some_fn=lambda: None,
                 _G_TENSOR=torch.ones(4))
        found = collect_module_level_tensors((PREFIX, ), require_cuda=False)
        assert list(found) == [f"{PREFIX}._G_TENSOR"]

    def test_require_cuda_filters_host_tensors(self):
        _install(_G_TENSOR=torch.ones(4))
        assert collect_module_level_tensors((PREFIX, ),
                                            require_cuda=True) == {}

    def test_reload_machinery_is_not_self_reported(self):
        found = collect_module_level_tensors(require_cuda=False)
        assert not any("reload_arena" in p or "reload_manifest" in p
                       for p in found)


class TestCheck:

    def teardown_method(self):
        _cleanup()

    def test_clean_when_nothing_changes(self):
        _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        assert len(manifest) == 1
        report = manifest.check()
        assert report.is_clean
        assert report.checked == 1

    def test_expired_when_old_storage_is_freed(self):
        """A cache that repopulates and drops its last reference: the
        captured address is dangling."""
        module = _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        module._CACHE["k"] = torch.ones(4)  # old storage loses its referent
        report = manifest.check()
        assert report.expired and not report.is_clean

    def test_moved_when_rebound_but_old_storage_kept_alive(self):
        """The Machete shape: rebound while a capture artifact still holds
        the previous storage, so expiry never fires and the model reads
        stale values in silence."""
        module = _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        keepalive = module._CACHE["k"]  # noqa: F841 - stands in for a capture artifact
        module._CACHE["k"] = torch.ones(4)
        report = manifest.check()
        assert not report.expired, "old storage should still be alive"
        assert report.moved and not report.is_clean

    def test_grow_in_place_is_clean(self):
        """A cache hit that reuses its buffer must not be flagged."""
        module = _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        module._CACHE["k"].fill_(7.0)  # same storage, new contents
        assert manifest.check().is_clean

    def test_growth_reallocation_is_caught(self):
        """The `_g_workspace` pattern: `if existing.numel() < needed:
        reallocate` silently rebinds under any graph that captured it."""
        module = _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        module._CACHE["k"] = torch.ones(64)  # grown
        report = manifest.check()
        assert not report.is_clean

    def test_vanished_path_with_live_storage_is_not_a_violation(self):
        module = _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        keepalive = module._CACHE.pop("k")  # noqa: F841
        report = manifest.check()
        assert report.vanished and report.is_clean

    def test_new_entries_after_capture_are_ignored(self):
        module = _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        module._CACHE["fresh"] = torch.ones(4)
        assert manifest.check().is_clean

    def test_report_names_the_offending_path(self):
        module = _install(_CACHE={"k": torch.ones(4)})
        manifest = _record()
        keepalive = module._CACHE["k"]  # noqa: F841
        module._CACHE["k"] = torch.ones(4)
        text = manifest.check().format()
        assert "_CACHE['k']" in text and "moved" in text
