# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import platform
import runpy
import sys
import types
from pathlib import Path

import pytest
import setuptools


ROOT = Path(__file__).resolve().parents[1]


def load_precompiled_wheel_utils(monkeypatch):
    for key in (
        "VLLM_USE_PRECOMPILED",
        "VLLM_USE_PRECOMPILED_RUST",
        "VLLM_PRECOMPILED_WHEEL_LOCATION",
        "VLLM_PRECOMPILED_WHEEL_VARIANT",
        "VLLM_PRECOMPILED_WHEEL_COMMIT",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(sys, "argv", ["setup.py", "egg_info"])
    monkeypatch.setattr(setuptools, "setup", lambda *args, **kwargs: None)

    rust = types.ModuleType("setuptools_rust")
    rust_build = types.ModuleType("setuptools_rust.build")

    class Binding:
        Exec = object()
        PyO3 = object()

    class RustExtension:
        def __init__(self, *, target=None, binding=None, **kwargs):
            self.target = {"default": target}
            self.binding = binding

    rust.Binding = Binding
    rust.RustExtension = RustExtension
    rust_build.build_rust = type("build_rust", (), {})

    scm = types.ModuleType("setuptools_scm")
    scm.get_version = lambda *args, **kwargs: "0.0.0"

    monkeypatch.setitem(sys.modules, "setuptools_rust", rust)
    monkeypatch.setitem(sys.modules, "setuptools_rust.build", rust_build)
    monkeypatch.setitem(sys.modules, "setuptools_scm", scm)

    saved_modules = {
        name: sys.modules.get(name)
        for name in ("envs", "rust_build")
    }

    try:
        namespace = runpy.run_path(
            str(ROOT / "setup.py"),
            run_name="vllm_setup_test",
        )
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    return namespace["precompiled_wheel_utils"]


@pytest.mark.parametrize(
    "explicit_variant",
    [True, False],
    ids=["explicit", "auto-detected"],
)
def test_cuda_variant_metadata_failure_does_not_fallback_to_default(
    monkeypatch,
    explicit_variant,
):
    utils = load_precompiled_wheel_utils(monkeypatch)

    commit = "a" * 40
    monkeypatch.setenv("VLLM_PRECOMPILED_WHEEL_COMMIT", commit)
    monkeypatch.setattr(utils, "is_rocm_system", lambda: False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    if explicit_variant:
        monkeypatch.setenv("VLLM_PRECOMPILED_WHEEL_VARIANT", "cu129")
    else:
        monkeypatch.setattr(
            utils,
            "detect_system_cuda_variant",
            lambda: "cu129",
        )

    calls = []

    def fetch_metadata(commit_arg, variant):
        calls.append((commit_arg, variant))

        if variant == "cu129":
            raise RuntimeError("simulated missing cu129 metadata")

        if variant is None:
            return (
                [
                    {
                        "package_name": "vllm",
                        "platform_tag": "manylinux_2_28_x86_64",
                        "filename": "vllm-default.whl",
                        "path": "../vllm-default.whl",
                    }
                ],
                "https://example.invalid/commit/vllm/",
            )

        raise AssertionError(f"unexpected variant: {variant}")

    monkeypatch.setattr(
        utils,
        "fetch_metadata_for_variant",
        fetch_metadata,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated missing cu129 metadata",
    ):
        utils.determine_wheel_url()

    assert calls == [(commit, "cu129")]
