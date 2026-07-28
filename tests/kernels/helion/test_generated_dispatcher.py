# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import builtins
import importlib
import sys
from pathlib import Path

import pytest
import torch

from vllm.kernels.helion_generated import (
    dispatcher,
    warm_up_helion_kernels,
)
from vllm.kernels.helion_generated.manifests import (
    GENERATED_KERNEL_MANIFESTS,
    PRESERVED_SPECIALIZATION_MANIFESTS,
)
from vllm.kernels.helion_generated.ops import (
    fused_qk_norm_rope,
    per_token_group_fp8_quant,
    rms_norm_per_block_quant,
    silu_and_mul_per_block_quant,
)


def test_generator_registry_covers_source_kernels():
    pytest.importorskip("helion")
    from scripts.generate_helion_kernels import KERNEL_REGISTRY

    assert set(KERNEL_REGISTRY) == {
        "dynamic_per_token_scaled_fp8_quant",
        "fused_qk_norm_rope",
        "per_token_group_fp8_quant",
        "rms_norm_dynamic_per_token_quant",
        "rms_norm_per_block_quant",
        "silu_and_mul_per_block_quant",
        "silu_mul_fp8",
    }


def test_nearest_config_matching_and_token_bucketing():
    assert per_token_group_fp8_quant._select_module(
        "nvidia_h100", 2048, 128, 1
    ).endswith("_t1")
    assert per_token_group_fp8_quant._select_module(
        "nvidia_h100", 2048, 128, 3
    ).endswith("_t4")
    assert per_token_group_fp8_quant._select_module(
        "nvidia_h100", 2048, 128, 8193
    ).endswith("_t8192")
    assert per_token_group_fp8_quant._select_module(
        "nvidia_h100", 2049, 128, 1
    ).endswith("_h2048_g128_t1")
    assert per_token_group_fp8_quant._select_module(
        "nvidia_h100", 2048, 64, 1
    ).endswith("_h2048_g128_t1")
    assert per_token_group_fp8_quant._select_module("nvidia_b200", 2049, 128, 1) is None
    assert per_token_group_fp8_quant._select_module("nvidia_b200", 2048, 64, 1) is None
    assert per_token_group_fp8_quant._select_module("nvidia_h200", 2048, 128, 1) is None


def test_fusion_kernel_token_bucketing():
    dispatcher._select_bucketed_module.cache_clear()
    assert dispatcher._select_bucketed_module(
        "rms_norm_per_block_quant", "nvidia_h100", (2048, 128), 3
    ).endswith("_t4")
    assert dispatcher._select_bucketed_module(
        "silu_and_mul_per_block_quant", "nvidia_h100", (6144, 128), 20000
    ).endswith("_t16384")
    assert (
        dispatcher._select_bucketed_module(
            "fused_qk_norm_rope", "nvidia_b200", (16, 8), 1
        )
        is None
    )


def test_launcher_import_is_cached(monkeypatch):
    dispatcher._load_launcher.cache_clear()
    calls = 0
    real_import = importlib.import_module

    def counting_import(name: str):
        nonlocal calls
        calls += 1
        return real_import(name)

    monkeypatch.setattr(dispatcher.importlib, "import_module", counting_import)
    module = per_token_group_fp8_quant._select_module("nvidia_h100", 2048, 128, 2)
    assert module is not None
    assert dispatcher._load_launcher(module) is dispatcher._load_launcher(module)
    assert calls == 1


def test_warmup_bucket_selection(monkeypatch):
    monkeypatch.setattr(
        per_token_group_fp8_quant, "_runtime_platform", lambda: "nvidia_h100"
    )
    assert per_token_group_fp8_quant.selected_token_buckets([1, 3, 64, 65, 9000]) == (
        1,
        4,
        64,
        128,
        8192,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_layout_matching(monkeypatch):
    monkeypatch.setattr(
        per_token_group_fp8_quant, "_runtime_platform", lambda: "nvidia_h100"
    )
    x = torch.empty((5, 2048), device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    row = torch.empty((5, 16), device="cuda", dtype=torch.float32)
    row64 = torch.empty((5, 32), device="cuda", dtype=torch.float32)
    column = torch.empty((16, 5), device="cuda", dtype=torch.float32).t()
    tma = torch.empty_strided((5, 16), (1, 8), device="cuda", dtype=torch.float32)

    assert per_token_group_fp8_quant._eligible_module(x, q, row, 128, False, False)
    assert per_token_group_fp8_quant._eligible_module(x, q, column, 128, True, False)
    assert per_token_group_fp8_quant._eligible_module(x, q, tma, 128, True, True)
    assert (
        per_token_group_fp8_quant._eligible_module(x, q, row, 128, True, False) is None
    )
    assert per_token_group_fp8_quant._eligible_module(x, q, row64, 64, False, False)
    x1024 = torch.empty((5, 1024), device="cuda", dtype=torch.bfloat16)
    q1024 = torch.empty_like(x1024, dtype=torch.float8_e4m3fn)
    row1024 = torch.empty((5, 8), device="cuda", dtype=torch.float32)
    assert per_token_group_fp8_quant._eligible_module(
        x1024, q1024, row1024, 128, False, False
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fusion_kernel_exact_matching(monkeypatch):
    for module in (
        fused_qk_norm_rope,
        rms_norm_per_block_quant,
        silu_and_mul_per_block_quant,
    ):
        monkeypatch.setattr(module, "_runtime_platform", lambda: "nvidia_h100")
    device = "cuda"

    input = torch.empty((5, 2048), device=device, dtype=torch.bfloat16)
    result = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    weight = torch.empty(2048, device=device, dtype=input.dtype)
    scale = torch.empty((16, 5), device=device, dtype=torch.float32).t()
    residual = torch.empty_like(input)
    assert rms_norm_per_block_quant._eligible_module(
        result, input, weight, scale, None, None, 128, True
    )
    assert (
        rms_norm_per_block_quant._eligible_module(
            result, input, weight, scale, None, residual, 128, True
        )
        is None
    )

    silu_input = torch.empty((5, 12288), device=device, dtype=torch.bfloat16)
    silu_out = torch.empty((5, 6144), device=device, dtype=torch.float8_e4m3fn)
    silu_scales = torch.empty((48, 5), device=device, dtype=torch.float32).t()
    assert silu_and_mul_per_block_quant._eligible_module(
        silu_out, silu_input, silu_scales, 128, None, True
    )
    assert (
        silu_and_mul_per_block_quant._eligible_module(
            silu_out,
            silu_input,
            torch.empty_like(silu_scales.contiguous()),
            128,
            None,
            False,
        )
        is None
    )

    qkv = torch.empty((5, 4096), device=device, dtype=torch.bfloat16)
    q_weight = torch.empty(128, device=device, dtype=qkv.dtype)
    k_weight = torch.empty_like(q_weight)
    cache = torch.empty((40960, 128), device=device, dtype=qkv.dtype)
    positions = torch.arange(5, device=device, dtype=torch.int64)
    assert fused_qk_norm_rope._eligible_module(
        qkv, 16, 8, 8, 128, q_weight, k_weight, cache, True, positions, -1
    )
    assert fused_qk_norm_rope._eligible_module(
        qkv, 16, 8, 8, 128, q_weight, k_weight, cache, False, positions, -1
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("platform,hidden_size", [(None, 2048), ("nvidia_h100", 1000)])
def test_unsupported_case_uses_native_fallback(
    monkeypatch, platform: str | None, hidden_size: int
):
    calls = 0

    def fallback(*args, **kwargs):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        per_token_group_fp8_quant, "_runtime_platform", lambda: platform
    )
    monkeypatch.setattr(per_token_group_fp8_quant, "_native_fallback", fallback)
    x = torch.empty((2, hidden_size), device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((2, hidden_size // 128), device="cuda", dtype=torch.float32)
    per_token_group_fp8_quant.per_token_group_fp8_quant(
        x, q, s, 128, 1e-10, -448, 448, False
    )
    assert calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_generated_execution_does_not_import_helion(monkeypatch):
    module_path = per_token_group_fp8_quant._select_module("nvidia_h100", 3072, 64, 2)
    assert module_path is not None
    sys.modules.pop(module_path, None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "helion" or name.startswith("helion."):
            raise AssertionError(f"unexpected Helion import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    launcher = importlib.import_module(module_path).call
    x = torch.randn((2, 3072), device="cuda", dtype=torch.bfloat16)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((2, 48), device="cuda", dtype=torch.float32)
    launcher(x, q, s, 64, 1e-10, -448, 448, False, False, False)
    torch.accelerator.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_untuned_shapes_match_native_ops(monkeypatch):
    for module in (
        fused_qk_norm_rope,
        per_token_group_fp8_quant,
        rms_norm_per_block_quant,
        silu_and_mul_per_block_quant,
    ):
        monkeypatch.setattr(module, "_runtime_platform", lambda: "nvidia_h100")

    tokens = 3
    hidden_size = 3072
    group_size = 64
    input = torch.randn((tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    output_ref = torch.empty_like(output)
    scales = torch.empty(
        (tokens, hidden_size // group_size), device="cuda", dtype=torch.float32
    )
    scales_ref = torch.empty_like(scales)
    per_token_group_fp8_quant.per_token_group_fp8_quant(
        input, output, scales, group_size, 1e-10, -448, 448, False
    )
    torch.ops._C.per_token_group_fp8_quant(
        input, output_ref, scales_ref, group_size, 1e-10, -448, 448, False, False, False
    )
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=0.2, atol=0.2)
    torch.testing.assert_close(scales, scales_ref, rtol=2e-2, atol=1e-6)

    weight = torch.randn(hidden_size, device="cuda", dtype=input.dtype)
    result = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    result_ref = torch.empty_like(result)
    block_scales = torch.empty(
        (hidden_size // group_size, tokens), device="cuda", dtype=torch.float32
    ).t()
    block_scales_ref = torch.empty_like(block_scales)
    rms_norm_per_block_quant.rms_norm_per_block_quant(
        result,
        input,
        weight,
        block_scales,
        1e-6,
        None,
        None,
        group_size,
        True,
    )
    torch.ops._C.rms_norm_per_block_quant(
        result_ref,
        input,
        weight,
        block_scales_ref,
        1e-6,
        None,
        None,
        group_size,
        True,
    )
    torch.testing.assert_close(result.float(), result_ref.float(), rtol=0.2, atol=0.2)
    torch.testing.assert_close(block_scales, block_scales_ref, rtol=2e-2, atol=1e-6)

    intermediate_size = 8192
    silu_input = torch.randn(
        (tokens, 2 * intermediate_size), device="cuda", dtype=torch.bfloat16
    )
    silu_output = torch.empty(
        (tokens, intermediate_size), device="cuda", dtype=torch.float8_e4m3fn
    )
    silu_output_ref = torch.empty_like(silu_output)
    silu_scales = torch.empty(
        (intermediate_size // group_size, tokens),
        device="cuda",
        dtype=torch.float32,
    ).t()
    silu_scales_ref = torch.empty_like(silu_scales)
    silu_and_mul_per_block_quant.silu_and_mul_per_block_quant(
        silu_output, silu_input, silu_scales, group_size, None, True
    )
    torch.ops._C.silu_and_mul_per_block_quant(
        silu_output_ref, silu_input, silu_scales_ref, group_size, None, True
    )
    torch.testing.assert_close(
        silu_output.float(), silu_output_ref.float(), rtol=0.2, atol=0.2
    )
    torch.testing.assert_close(silu_scales, silu_scales_ref, rtol=2e-2, atol=1e-6)

    q_heads = 24
    kv_heads = 4
    head_dim = 64
    qkv = torch.randn(
        (tokens, (q_heads + 2 * kv_heads) * head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    qkv_ref = qkv.clone()
    q_weight = torch.randn(head_dim, device="cuda", dtype=qkv.dtype)
    k_weight = torch.randn_like(q_weight)
    cos_sin_cache = torch.randn((1024, 64), device="cuda", dtype=qkv.dtype)
    position_ids = torch.arange(tokens, device="cuda", dtype=torch.int64)
    fused_qk_norm_rope.fused_qk_norm_rope(
        qkv,
        q_heads,
        kv_heads,
        kv_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_sin_cache,
        False,
        position_ids,
    )
    torch.ops._C.fused_qk_norm_rope(
        qkv_ref,
        q_heads,
        kv_heads,
        kv_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_sin_cache,
        False,
        position_ids,
        -1,
    )
    torch.testing.assert_close(qkv, qkv_ref, rtol=3e-2, atol=3e-2)


def test_generated_artifacts_have_stable_runtime_contract():
    root = Path(__file__).parents[3]
    expected_args = {
        "fused_qk_norm_rope": [
            "qkv",
            "num_heads_q",
            "num_heads_k",
            "num_heads_v",
            "head_dim",
            "eps",
            "q_weight",
            "k_weight",
            "cos_sin_cache",
            "is_neox",
            "position_ids",
            "forced_token_heads_per_warp",
        ],
        "per_token_group_fp8_quant": [
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
        ],
        "rms_norm_per_block_quant": [
            "result",
            "input",
            "weight",
            "scale",
            "epsilon",
            "scale_ub",
            "residual",
            "group_size",
            "is_scale_transposed",
        ],
        "silu_and_mul_per_block_quant": [
            "out",
            "input",
            "scales",
            "group_size",
            "scale_ub",
            "is_scale_transposed",
        ],
    }
    modules_by_kernel = {
        kernel_name: {
            path for kernels in platform_manifests.values() for path in kernels.values()
        }
        for kernel_name, platform_manifests in GENERATED_KERNEL_MANIFESTS.items()
    }
    assert sum(map(len, modules_by_kernel.values())) == 270
    assert {
        ("fused_qk_norm_rope", "nvidia_h100"),
        ("per_token_group_fp8_quant", "nvidia_h100"),
        ("rms_norm_per_block_quant", "nvidia_h100"),
        ("silu_and_mul_per_block_quant", "nvidia_h100"),
    } == PRESERVED_SPECIALIZATION_MANIFESTS
    module_paths = {path for paths in modules_by_kernel.values() for path in paths}
    assert all(
        any(f".kernels.{kernel_name}.nvidia_" in path for kernel_name in expected_args)
        for path in module_paths
    )

    for kernel_name, module_paths in modules_by_kernel.items():
        for module_path in module_paths:
            path = root / (module_path.replace(".", "/") + ".py")
            tree = ast.parse(path.read_text())
            imported_roots: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(
                        name.name.split(".", 1)[0] for name in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    imported_roots.add((node.module or "").split(".", 1)[0])
            assert imported_roots <= {"__future__", "torch", "triton"}
            call = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "call"
            )
            assert [arg.arg for arg in call.args.args] == expected_args[kernel_name]
            if (
                "nvidia_h100" in module_path
                and (kernel_name, "nvidia_h100") in PRESERVED_SPECIALIZATION_MANIFESTS
            ):
                assert "tl.constexpr" in path.read_text()


def test_generated_fusion_op_map_preserves_mutations():
    op_map = dispatcher.build_compiled_generated_op_map()
    assert len(op_map) == 3
    for native_op, routed_op in op_map.items():
        native_mutations = tuple(
            (arg.name, bool(arg.alias_info and arg.alias_info.is_write))
            for arg in native_op._schema.arguments
        )
        routed_mutations = tuple(
            (arg.name, bool(arg.alias_info and arg.alias_info.is_write))
            for arg in routed_op._schema.arguments
        )
        assert routed_mutations == native_mutations


def test_generated_fusion_warmup_case_selection(monkeypatch):
    monkeypatch.setattr(dispatcher, "_runtime_platform", lambda: "nvidia_h100")
    cases = dispatcher._selected_cases("silu_and_mul_per_block_quant", [3, 20000])
    assert (6144, 128, 4) in cases
    assert (6144, 128, 16384) in cases
    assert len(cases) == 6


def test_warm_up_helion_kernels_combines_generated_warmups(monkeypatch):
    calls = []
    modules = (
        fused_qk_norm_rope,
        per_token_group_fp8_quant,
        rms_norm_per_block_quant,
        silu_and_mul_per_block_quant,
    )
    for module in modules:
        monkeypatch.setattr(
            module,
            "warmup",
            lambda token_counts, device: calls.append((tuple(token_counts), device)),
        )

    warm_up_helion_kernels(iter([3, 17]), "cuda:1")

    assert calls == [((3, 17), "cuda:1")] * len(modules)
