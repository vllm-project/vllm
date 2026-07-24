# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import builtins
import importlib
import sys
from pathlib import Path

import pytest
import torch

from vllm.kernels.helion_generated import dispatcher
from vllm.kernels.helion_generated.manifests import MANIFESTS


def test_exact_matching_and_token_bucketing():
    dispatcher._select_module.cache_clear()
    assert dispatcher._select_module("nvidia_h100", 2048, 128, 1).endswith("_t1")
    assert dispatcher._select_module("nvidia_h100", 2048, 128, 1).endswith("_t1")
    assert dispatcher._select_module.cache_info().hits == 1
    assert dispatcher._select_module("nvidia_h100", 2048, 128, 3).endswith("_t4")
    assert dispatcher._select_module("nvidia_h100", 2048, 128, 8193).endswith("_t8192")
    assert dispatcher._select_module("nvidia_h100", 2049, 128, 1) is None
    assert dispatcher._select_module("nvidia_h100", 2048, 64, 1) is None
    assert dispatcher._select_module("nvidia_h200", 2048, 128, 1) is None


def test_launcher_import_is_cached(monkeypatch):
    dispatcher._load_launcher.cache_clear()
    calls = 0
    real_import = importlib.import_module

    def counting_import(name: str):
        nonlocal calls
        calls += 1
        return real_import(name)

    monkeypatch.setattr(dispatcher.importlib, "import_module", counting_import)
    module = dispatcher._select_module("nvidia_h100", 2048, 128, 2)
    assert module is not None
    assert dispatcher._load_launcher(module) is dispatcher._load_launcher(module)
    assert calls == 1


def test_warmup_bucket_selection(monkeypatch):
    monkeypatch.setattr(dispatcher, "_runtime_platform", lambda: "nvidia_h100")
    assert dispatcher.selected_token_buckets([1, 3, 64, 65, 9000]) == (
        1,
        4,
        64,
        128,
        8192,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_layout_matching(monkeypatch):
    monkeypatch.setattr(dispatcher, "_runtime_platform", lambda: "nvidia_h100")
    x = torch.empty((5, 2048), device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    row = torch.empty((5, 16), device="cuda", dtype=torch.float32)
    column = torch.empty((16, 5), device="cuda", dtype=torch.float32).t()
    tma = torch.empty_strided((5, 16), (1, 8), device="cuda", dtype=torch.float32)

    assert dispatcher._eligible_module(x, q, row, 128, False, False)
    assert dispatcher._eligible_module(x, q, column, 128, True, False)
    assert dispatcher._eligible_module(x, q, tma, 128, True, True)
    assert dispatcher._eligible_module(x, q, row, 128, True, False) is None
    assert dispatcher._eligible_module(x, q, row, 64, False, False) is None
    assert (
        dispatcher._eligible_module(x[:, :1024], q[:, :1024], row, 128, False, False)
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("platform,hidden_size", [(None, 2048), ("nvidia_h100", 1024)])
def test_unsupported_case_uses_native_fallback(
    monkeypatch, platform: str | None, hidden_size: int
):
    calls = 0

    def fallback(*args, **kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(dispatcher, "_runtime_platform", lambda: platform)
    monkeypatch.setattr(dispatcher, "_native_fallback", fallback)
    x = torch.empty((2, hidden_size), device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((2, hidden_size // 128), device="cuda", dtype=torch.float32)
    dispatcher.per_token_group_fp8_quant(x, q, s, 128, 1e-10, -448, 448, False)
    assert calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_generated_execution_does_not_import_helion(monkeypatch):
    module_path = dispatcher._select_module("nvidia_h100", 2048, 128, 2)
    assert module_path is not None
    sys.modules.pop(module_path, None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "helion" or name.startswith("helion."):
            raise AssertionError(f"unexpected Helion import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    launcher = importlib.import_module(module_path).call
    x = torch.randn((2, 2048), device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((2, 16), device="cuda", dtype=torch.float32)
    launcher(x, q, s, 128, 1e-10, -448, 448, False, False, False)
    torch.accelerator.synchronize()


def test_generated_artifacts_have_stable_runtime_contract():
    root = Path(__file__).parents[3]
    expected_args = [
        "input",
        "output_q",
        "output_s",
        "group_size",
        "eps",
        "fp8_min",
        "fp8_max",
        "scale_ue8m0",
        "dummy_is_scale_transposed",
        "dummy_is_tma_aligned",
    ]
    module_paths = {path for kernels in MANIFESTS.values() for path in kernels.values()}
    assert len(module_paths) == 84
    assert all(
        ".kernels.per_token_group_fp8_quant.nvidia_" in path for path in module_paths
    )

    for module_path in module_paths:
        path = root / (module_path.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text())
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(name.name.split(".", 1)[0] for name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_roots.add((node.module or "").split(".", 1)[0])
        assert imported_roots <= {"__future__", "torch", "triton"}
        call = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "call"
        )
        assert [arg.arg for arg in call.args.args] == expected_args
