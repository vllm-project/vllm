# SPDX-License-Identifier: Apache-2.0
"""Default vLLM path is not affected by Quest backend code being on disk."""
from __future__ import annotations

import sys


def test_quest_packages_not_imported_by_vllm_attention_module():
    # Importing the vLLM attention machinery must not eagerly drag in any
    # quest module. Other tests in this package may have already imported
    # quest submodules, so snapshot+restore sys.modules to avoid leaving
    # later tests with dangling module references.
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name.startswith("vllm.v1.attention.backends.quest")
    }
    for name in saved:
        del sys.modules[name]
    try:
        import vllm.v1.attention.selector  # noqa: F401
        import vllm.v1.attention.backends.flash_attn  # noqa: F401
        import vllm.v1.attention.backends.registry  # noqa: F401

        bad = [
            m
            for m in sys.modules
            if m.startswith("vllm.v1.attention.backends.quest")
        ]
        assert bad == [], (
            f"Quest packages were eagerly imported by vLLM core: {bad}. "
            "The Quest backend must remain opt-in."
        )
    finally:
        # Restore quest modules so subsequent tests see the same module
        # objects they already imported.
        for name, mod in saved.items():
            sys.modules[name] = mod


def test_vllm_config_can_be_built_without_quest_config():
    from vllm.config import VllmConfig

    cfg = VllmConfig.__new__(VllmConfig)
    # Just make sure the field has a None default and is not required.
    import dataclasses

    field = next(f for f in dataclasses.fields(VllmConfig) if f.name == "quest_config")
    # default OR default_factory must produce None.
    if field.default is not dataclasses.MISSING:
        assert field.default is None
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        assert field.default_factory() is None
    else:
        raise AssertionError("quest_config has neither default nor default_factory")
