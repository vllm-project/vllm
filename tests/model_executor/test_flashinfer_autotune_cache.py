# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from vllm.model_executor.warmup import kernel_warmup


def test_resolve_flashinfer_autotune_file_default_layout(
    monkeypatch, tmp_path: Path
) -> None:
    fake_jit = SimpleNamespace(
        env=SimpleNamespace(
            FLASHINFER_WORKSPACE_DIR=Path("/flashinfer-cache/0.6.11.post2/103a")
        )
    )
    fake_flashinfer = SimpleNamespace(jit=fake_jit)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.jit", fake_jit)
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", None)

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == (
        tmp_path
        / "flashinfer_autotune_cache"
        / "0.6.11.post2"
        / "103a"
        / cache_hash
        / "autotune_configs.json"
    )
    assert path.parent.is_dir()


def test_resolve_flashinfer_autotune_file_uses_override_dir(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        kernel_warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path)
    )
    monkeypatch.setattr(
        kernel_warmup, "aot_compile_hash_factors", lambda _: ["env-hash", "config-hash"]
    )

    runner = SimpleNamespace(vllm_config=SimpleNamespace())
    cache_hash = sha256(str(["env-hash", "config-hash"]).encode()).hexdigest()

    path = kernel_warmup._resolve_flashinfer_autotune_file(runner)

    assert path == tmp_path / cache_hash / "autotune_configs.json"
