# SPDX-License-Identifier: Apache-2.0
"""Subprocess assertion: with Quest disabled, no quest module is loaded."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_quest_module_not_imported_when_env_unset():
    script = textwrap.dedent(
        """
        import os, sys, json
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        # Touch the parts of vLLM that a real engine init would.
        from vllm.config import VllmConfig  # noqa
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )  # noqa
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
        )  # noqa
        loaded = sorted(
            m for m in sys.modules if m.startswith("vllm.v1.attention.backends.quest")
        )
        print(json.dumps(loaded))
        """
    )
    env = dict(os.environ)
    env.pop("VLLM_ATTENTION_BACKEND", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, (
        f"subprocess failed: stdout={res.stdout!r} stderr={res.stderr!r}"
    )
    loaded = res.stdout.strip().splitlines()[-1]
    import json

    assert json.loads(loaded) == [], (
        f"Quest modules leaked into default vLLM path: {loaded}"
    )
