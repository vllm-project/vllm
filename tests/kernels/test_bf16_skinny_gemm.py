# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BF16 skinny GEMMs and model-specific selectors."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import regex as re
import torch
from torch import nn

from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
)
from vllm.models.deepseek_v32.nvidia import glm52_low_latency_gemm as glm52_gemm
from vllm.models.kimi_k3.nvidia import low_latency_gemm as k3_gemm
from vllm.models.kimi_k3.nvidia.low_latency_gemm import KIMI_K3_PROJECTIONS
from vllm.models.qwen4_exp.nvidia import low_latency_gemm as qwen4_exp_gemm

# Keyed by local (N, K): (cute token counts, dsv3 token counts). 1536x7168 is
# the unified shared_gate_up_proj/mla_g_proj entry (dsv3 M1..16).
EXPECTED_SELECTIONS = {
    (1536, 128): (set(), set(range(1, 17))),
    (3072, 128): (set(), set(range(1, 17))),
    (1536, 7168): (set(), set(range(1, 17))),
    (3072, 7168): (set(range(1, 6)), set()),
    (2112, 7168): (set(), set(range(1, 17))),
    (2304, 1536): (set(), set(range(1, 17))),
    (4608, 1536): (set(), set(range(1, 17))),
    (3584, 7168): ({1}, set(range(2, 9))),
    (6288, 7168): (set(range(1, 5)), set()),
    (12448, 7168): (set(range(1, 4)), set()),
    (7168, 768): (set(), set(range(1, 17))),
    (7168, 1536): ({1}, set()),
    (7168, 3072): ({1, 2}, set()),
    (7168, 3584): ({1, 2}, set()),
    (7168, 4224): ({1}, set()),
    (7168, 8448): (set(range(1, 4)), set()),
    (8448, 7168): ({1, 2}, set()),
    (16896, 7168): ({1, 2}, set()),
    (7168, 14336): ({1, 2}, set()),
    (20480, 7168): (set(range(1, 5)), set()),
    (40960, 7168): (set(range(1, 5)), set()),
    # TP16.
    (3216, 7168): (set(range(1, 6)), set(range(9, 16))),
    (768, 7168): (set(range(1, 5)), set(range(5, 17))),
    (1152, 1536): ({1}, set(range(2, 17))),
    (768, 128): (set(), set(range(1, 17))),
    (7168, 384): (set(), set(range(1, 9))),
    (4224, 7168): (set(range(1, 4)), set(range(4, 9))),
    (10240, 7168): (set(range(1, 5)), set()),
}

K3ProjectionTable = dict[tuple[int, int], k3_gemm.ProjectionSpec]
K3_GPU_TABLES = (
    ((10, 3), k3_gemm.KIMI_K3_PROJECTIONS),
    ((9, 0), k3_gemm.KIMI_K3_PROJECTIONS_SM90),
)

CUTE_CASES = [
    (capability, spec.n, spec.k, num_tokens)
    for capability, table in K3_GPU_TABLES
    for spec in table.values()
    for num_tokens, _ in spec.cute_configs
]

RESIDUAL_CUTE_CASES = [
    (spec.n, spec.k, num_tokens)
    for spec in k3_gemm.KIMI_K3_PROJECTIONS.values()
    for num_tokens, _ in spec.residual_configs
]

GLM_CUTE_CASES = [
    (spec, config)
    for spec in glm52_gemm.GLM52_PROJECTIONS.values()
    for _, config in spec.cute_configs
]

QWEN4_EXP_SM90_CASES = [
    (n, k, num_tokens, config)
    for (n, k), plans in qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS.items()
    for num_tokens, config in plans.items()
]

EXPECTED_CUTE_CONFIGS = {
    (3072, 7168, 1): (224, 3, 4, 8),
    (3072, 7168, 2): (128, 3, 2, 8),
    (3072, 7168, 3): (128, 2, 1, 8),
    (3072, 7168, 4): (64, 2, 2, 8),
    (3072, 7168, 5): (128, 3, 1, 8),
    (3584, 7168, 1): (224, 2, 4, 8),
    (6288, 7168, 1): (224, 3, 4, 8),
    (6288, 7168, 2): (64, 3, 2, 8),
    (6288, 7168, 3): (32, 3, 4, 8),
    (6288, 7168, 4): (128, 6, 1, 8),
    (12448, 7168, 1): (224, 4, 2, 8),
    (12448, 7168, 2): (64, 4, 2, 8),
    (12448, 7168, 3): (64, 2, 2, 8),
    (7168, 1536, 1): (96, 4, 2, 8),
    (7168, 3072, 1): (96, 2, 4, 8),
    (7168, 3072, 2): (32, 4, 4, 8),
    (7168, 3584, 1): (224, 4, 2, 8),
    (7168, 3584, 2): (64, 4, 2, 8),
    (7168, 4224, 1): (96, 4, 2, 4),
    (7168, 8448, 1): (32, 4, 4, 8),
    (7168, 8448, 2): (96, 4, 1, 8),
    (7168, 8448, 3): (96, 4, 1, 8),
    (8448, 7168, 1): (224, 3, 4, 8),
    (8448, 7168, 2): (32, 4, 4, 8),
    (16896, 7168, 1): (224, 6, 4, 8),
    (16896, 7168, 2): (32, 4, 4, 8),
    (7168, 14336, 1): (256, 2, 1, 4),
    (7168, 14336, 2): (224, 4, 2, 8),
    (20480, 7168, 1): (224, 4, 2, 8),
    (20480, 7168, 2): (64, 4, 2, 8),
    (20480, 7168, 3): (64, 2, 2, 8),
    (20480, 7168, 4): (64, 4, 1, 8),
    (40960, 7168, 1): (128, 4, 2, 8),
    (40960, 7168, 2): (64, 4, 2, 8),
    (40960, 7168, 3): (64, 2, 2, 8),
    (40960, 7168, 4): (64, 4, 1, 8),
    # TP16.
    (3216, 7168, 1): (224, 3, 4, 8),
    (3216, 7168, 2): (128, 4, 2, 8),
    (3216, 7168, 3): (128, 2, 1, 8),
    (3216, 7168, 4): (64, 2, 2, 8),
    (3216, 7168, 5): (128, 3, 1, 8),
    (768, 7168, 1): (224, 2, 4, 8),
    (768, 7168, 2): (224, 2, 2, 8),
    (768, 7168, 3): (224, 2, 2, 8),
    (768, 7168, 4): (224, 2, 2, 8),
    (1152, 1536, 1): (192, 3, 4, 8),
    (4224, 7168, 1): (224, 3, 4, 8),
    (4224, 7168, 2): (128, 2, 1, 8),
    (4224, 7168, 3): (64, 2, 2, 8),
    (10240, 7168, 1): (224, 4, 2, 8),
    (10240, 7168, 2): (32, 2, 4, 8),
    (10240, 7168, 3): (64, 4, 1, 8),
    (10240, 7168, 4): (64, 4, 1, 8),
}

EXPECTED_RESIDUAL_CUTE_CONFIGS = {
    (7168, 3584, 1): (64, 4, 2, 8),
    (7168, 3584, 2): (64, 7, 2, 8),
    (7168, 3584, 3): (64, 2, 1, 8),
    (7168, 3584, 4): (64, 2, 1, 8),
}


def _config_tuple(config) -> tuple[int, int, int, int]:
    return (
        config.block_size,
        config.outputs_per_block,
        config.k_unroll,
        config.vector_width,
    )


def test_table_is_keyed_by_shape() -> None:
    for (n, k), spec in k3_gemm.KIMI_K3_PROJECTIONS.items():
        assert (spec.n, spec.k) == (n, k)


def test_every_dsv3_routed_shape_is_instantiated() -> None:
    """dsv3_fused_a_gemm specializes on (K, N); an unlisted shape raises.

    The table routes by shape while the kernel is built per shape, so a missing
    instantiation only shows up at the token counts that route to dsv3. Checking
    it here needs no GPU, which is the point -- a GPU-only check is exactly what
    let (3216, 7168) ship without its DISPATCH_DSV3_SHAPE(7168, 3216).
    """
    source = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "libtorch_stable"
        / "dsv3_fused_a_gemm.cu"
    ).read_text(encoding="utf-8")
    # Benchmark-only shapes live behind VLLM_K3_BENCH_SHAPES and are not built
    # by default, so they must not count as available.
    production_macros = source.split("#ifdef VLLM_K3_BENCH_SHAPES")[0]
    explicit = source.split("#undef DISPATCH_DSV3_SHAPE")[1].split(
        "#ifdef VLLM_K3_BENCH_SHAPES"
    )[0]
    compiled = {
        (int(hd_in), int(hd_out))
        for hd_in, hd_out in re.findall(
            r"DISPATCH_DSV3_SHAPE\((\d+),\s*(\d+)\)", production_macros
        )
    } | {
        (int(hd_in), int(hd_out))
        for hd_in, hd_out in re.findall(
            r"hd_in == (\d+) && hd_out == (\d+)",
            production_macros + explicit,
        )
    }
    assert compiled, "failed to parse the dispatch list"

    specs = [
        *KIMI_K3_PROJECTIONS.values(),
        *k3_gemm.KIMI_K3_PROJECTIONS_SM100.values(),
        *k3_gemm.KIMI_K3_PROJECTIONS_SM90.values(),
        glm52_gemm.GLM52_QKV_A_PROJECTION,
        glm52_gemm.GLM52_Q_B_PROJECTION,
    ]
    missing = sorted(
        (spec.n, spec.k)
        for spec in specs
        if spec.dsv3_tokens and (spec.k, spec.n) not in compiled
    )
    assert not missing, (
        f"routed to dsv3 with no instantiation: {missing}; add "
        "DISPATCH_DSV3_SHAPE(K, N) for each"
    )


def test_packed_row_major_rejects_single_row_slice() -> None:
    packed = torch.empty(1, 128)
    sliced = torch.empty(1, 144)[:, :128]

    assert packed.is_contiguous()
    assert sliced.is_contiguous()
    assert k3_gemm._is_packed_row_major(packed)
    assert not k3_gemm._is_packed_row_major(sliced)


def test_cute_configs_match_measured_table() -> None:
    configs = [
        (spec.n, spec.k, num_tokens, config)
        for spec in k3_gemm.KIMI_K3_PROJECTIONS.values()
        for num_tokens, config in spec.cute_configs
    ]
    actual = {
        (n, k, num_tokens): _config_tuple(config)
        for n, k, num_tokens, config in configs
    }
    assert actual == EXPECTED_CUTE_CONFIGS
    static_k_configs = {
        (n, k, num_tokens, config.static_k)
        for n, k, num_tokens, config in configs
        if config.static_k is not None
    }
    assert static_k_configs == {(7168, 14336, 1, 14336)}


def test_residual_cute_configs_match_measured_table() -> None:
    actual = {
        (spec.n, spec.k, num_tokens): _config_tuple(config)
        for spec in k3_gemm.KIMI_K3_PROJECTIONS.values()
        for num_tokens, config in spec.residual_configs
    }
    assert actual == EXPECTED_RESIDUAL_CUTE_CONFIGS


def test_kda_overlap_configs_match_measured_table() -> None:
    assert k3_gemm.KDA_PROJECTION_OVERLAP_MAX_TOKENS == 14
    assert k3_gemm.KDA_SKINNY_N_MAX_TOKENS == 14
    assert k3_gemm.KDA_SKINNY_K_MAX_TOKENS == 14
    assert _config_tuple(k3_gemm.KDA_M1_QKVG_CONFIG) == (64, 4, 2, 8)
    assert _config_tuple(k3_gemm.KDA_M1_FAB_CONFIG) == (224, 1, 2, 8)
    assert {
        num_tokens: _config_tuple(config)
        for num_tokens, config in k3_gemm.KDA_QKVG_CONFIGS.items()
    } == {
        1: (64, 4, 2, 8),
        2: (64, 3, 2, 8),
    }


@pytest.mark.parametrize("tp_size", [1, 2, 4, 16])
def test_kda_projection_overlap_is_tp8_only(tp_size: int) -> None:
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    kda = KimiK3DeltaAttention.__new__(KimiK3DeltaAttention)
    nn.Module.__init__(kda)
    kda.tp_size = tp_size
    kda._projection_overlap_max_tokens = 0

    assert not k3_gemm._enable_kda_projection_overlap(kda)
    assert kda._projection_overlap_max_tokens == 0


def test_kda_qkvg_autotune_enables_full_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flashinfer.gemm

    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    kda = KimiK3DeltaAttention.__new__(KimiK3DeltaAttention)
    nn.Module.__init__(kda)
    kda._projection_overlap_max_tokens = max(k3_gemm.KDA_QKVG_CONFIGS)
    kda.in_proj_qkvgfab = nn.Module()
    kda.in_proj_qkvgfab.weight = nn.Parameter(
        torch.empty(6144, 8, dtype=torch.bfloat16),
        requires_grad=False,
    )
    calls = []

    def fake_mm_bf16(a, b, *, pdl, backend):
        calls.append((a.shape, b.shape, pdl, backend))
        return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype)

    monkeypatch.setattr(flashinfer.gemm, "mm_bf16", fake_mm_bf16)

    k3_gemm.autotune_kda_qkvg(kda)

    assert calls == [(torch.Size([14, 8]), torch.Size([8, 6144]), True, "cute-dsl")]
    assert (
        kda._projection_overlap_max_tokens == k3_gemm.KDA_PROJECTION_OVERLAP_MAX_TOKENS
    )


def test_glm52_projection_plans_are_separate() -> None:
    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION

    qkv_a_plan = qkv_a.build_plan()
    q_b_plan = q_b.build_plan()

    assert (qkv_a.n, qkv_a.k, set(qkv_a_plan)) == (
        2624,
        6144,
        set(range(1, 17)),
    )
    assert (q_b.n, q_b.k, set(q_b_plan)) == (
        2048,
        2048,
        set(range(1, 17)),
    )
    assert {
        num_tokens
        for num_tokens, (backend, _) in qkv_a_plan.items()
        if backend == "cute"
    } == {1, 2}
    assert {
        num_tokens
        for num_tokens, (backend, _) in qkv_a_plan.items()
        if backend == "dsv3_fused_a"
    } == set(range(3, 17))
    assert {
        num_tokens for num_tokens, (backend, _) in q_b_plan.items() if backend == "cute"
    } == {1, 2}
    assert {
        num_tokens
        for num_tokens, (backend, _) in q_b_plan.items()
        if backend == "dsv3_fused_a"
    } == set(range(3, 17))

    eh = glm52_gemm.GLM52_EH_PROJECTION
    eh_plan = eh.build_plan()
    assert (eh.n, eh.k) == (6144, 12288)
    # The MTP eh_proj has no dsv3 winners; M >= 4 falls back to cuBLAS.
    assert set(eh_plan) == {1, 2, 3}
    assert all(backend == "cute" for backend, _ in eh_plan.values())


def test_glm52_layout_rejects_nonpacked_single_row_view() -> None:
    single_row = torch.empty(1, 144)[:, :128]
    multiple_rows = torch.empty(2, 144)[:, :128]

    assert single_row.stride() == multiple_rows.stride() == (144, 1)
    assert not glm52_gemm._is_supported_row_major(single_row)
    assert not glm52_gemm._is_supported_row_major(multiple_rows)


def test_glm52_installer_maps_only_selected_unquantized_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearBase(nn.Module):
        def __init__(
            self,
            n: int,
            k: int,
            quant_method: object,
        ) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(
                    n,
                    k,
                    dtype=torch.bfloat16,
                    device="meta",
                )
            )
            self.quant_method = quant_method

    qkv_a = glm52_gemm.GLM52_QKV_A_PROJECTION
    q_b = glm52_gemm.GLM52_Q_B_PROJECTION
    root = nn.Module()
    root.attn = nn.Module()
    root.attn.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.attn.q_b_proj = FakeLinearBase(
        q_b.n, q_b.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.same_shape_other_name = FakeLinearBase(
        qkv_a.n, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    root.quantized = nn.Module()
    quantized_method = object()
    root.quantized.q_b_proj = FakeLinearBase(q_b.n, q_b.k, quantized_method)
    root.wrong_shape = nn.Module()
    root.wrong_shape.fused_qkv_a_proj = FakeLinearBase(
        qkv_a.n + 1, qkv_a.k, glm52_gemm.UnquantizedLinearMethod()
    )
    monkeypatch.setattr(glm52_gemm, "LinearBase", FakeLinearBase)
    monkeypatch.setattr(glm52_gemm, "_is_sm103", lambda: True)
    monkeypatch.setattr(
        glm52_gemm.shape_dynamic_skinny_gemm,
        "is_available",
        lambda: False,
    )

    glm52_gemm.enable_glm52_low_latency_gemm(
        root,
        torch.bfloat16,
    )

    assert isinstance(
        root.attn.fused_qkv_a_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert isinstance(
        root.attn.q_b_proj.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.attn.fused_qkv_a_proj.quant_method._plan == qkv_a.build_plan()
    assert root.attn.q_b_proj.quant_method._plan == q_b.build_plan()
    assert isinstance(
        root.same_shape_other_name.quant_method,
        glm52_gemm.GLM52LowLatencyLinearMethod,
    )
    assert root.quantized.q_b_proj.quant_method is quantized_method
    assert (
        type(root.wrong_shape.fused_qkv_a_proj.quant_method)
        is glm52_gemm.UnquantizedLinearMethod
    )


@pytest.mark.parametrize("key", EXPECTED_SELECTIONS)
def test_sm103_selector_table(key: tuple[int, int]) -> None:
    n, k = key
    cute_tokens, dsv3_tokens = EXPECTED_SELECTIONS[key]
    for num_tokens in range(1, 17):
        backend = k3_gemm.select_kimi_k3_backend(num_tokens, n, k)
        if num_tokens in cute_tokens:
            assert backend == "cute"
        elif num_tokens in dsv3_tokens:
            assert backend == "dsv3_fused_a"
        else:
            assert backend is None


@pytest.mark.parametrize("key", EXPECTED_SELECTIONS)
def test_selector_requires_supported_shape_and_tokens(key: tuple[int, int]) -> None:
    n, k = key
    assert k3_gemm.select_kimi_k3_backend(0, n, k) is None
    assert k3_gemm.select_kimi_k3_backend(17, n, k) is None
    assert k3_gemm.select_kimi_k3_backend(1, n + 1, k) is None
    assert k3_gemm.select_kimi_k3_backend(1, n, k + 1) is None


def test_unlisted_shape_and_unselected_tokens_fall_back() -> None:
    # Shape absent from the table.
    assert k3_gemm.select_kimi_k3_backend(1, 1000, 1000) is None
    # o_proj (7168,1536) is CuTe M1 only; M2+ falls back.
    assert k3_gemm.select_kimi_k3_backend(2, 7168, 1536) is None


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_sm103_residual_selector_table(num_tokens: int) -> None:
    backend = k3_gemm.select_kimi_k3_backend(num_tokens, 7168, 3584, has_residual=True)
    assert backend == ("cute" if num_tokens <= 4 else None)


def test_build_plan_matches_selector() -> None:
    for spec in k3_gemm.KIMI_K3_PROJECTIONS.values():
        plan = k3_gemm._build_plan(spec)
        for num_tokens in range(1, 17):
            backend = k3_gemm.select_kimi_k3_backend(num_tokens, spec.n, spec.k)
            if backend is None:
                assert num_tokens not in plan
            else:
                assert plan[num_tokens][0] == backend


# Keyed by local (N, K): (cute token counts, dsv3 token counts). Mirrors
# KIMI_K3_PROJECTIONS_SM100, measured on B200; intentionally different from
# EXPECTED_SELECTIONS (e.g. 3584x7168 routes dsv3 at M2..8 on SM103 but only
# wins with CuTe at M1..3 on SM100).
EXPECTED_SELECTIONS_SM100 = {
    (1536, 128): (set(), {1, 16}),
    (3072, 128): (set(), {8}),
    (1536, 7168): ({1, 2}, {4, 8}),
    (3072, 7168): ({1, 2}, set()),
    (2112, 7168): ({1, 2}, {4, 16}),
    (2304, 1536): (set(), set(range(1, 17))),
    (4608, 1536): (set(), {1, 2, 4}),
    (6288, 7168): ({1}, set()),
    (12448, 7168): ({1, 2, 3, 4}, set()),
    (7168, 768): (set(), {1}),
    # SM103-tuned config wins on B200 where the heuristic config tied.
    (7168, 1536): ({1}, set()),
    # M2 config lifted from the SM103 table (wins on B200).
    (7168, 3072): ({1, 2}, set()),
    (7168, 3584): ({1, 2}, set()),
    (7168, 8448): ({1, 2, 3}, set()),
    (8448, 7168): ({1, 2}, set()),
    (16896, 7168): ({1}, set()),
    (20480, 7168): ({1, 2, 3, 4}, set()),
    (40960, 7168): ({1, 2, 3, 4}, set()),
    (10240, 7168): ({1, 2, 3, 4}, set()),
    (3584, 7168): ({1, 2, 3}, set()),
    # TP16 shapes measured on B200.
    (768, 7168): ({1, 2, 3, 4}, set()),
    (1152, 1536): ({1}, set()),
    (3216, 7168): ({1, 2}, set()),
    (4224, 7168): ({1, 2, 3}, set()),
}


@pytest.mark.parametrize(
    "table",
    [k3_gemm.KIMI_K3_PROJECTIONS_SM100, k3_gemm.KIMI_K3_PROJECTIONS_SM90],
)
def test_device_table_is_keyed_by_shape(table: K3ProjectionTable) -> None:
    for (n, k), spec in table.items():
        assert (spec.n, spec.k) == (n, k)


@pytest.mark.parametrize("key", EXPECTED_SELECTIONS_SM100)
def test_sm100_selector_table(key: tuple[int, int]) -> None:
    n, k = key
    spec = k3_gemm.KIMI_K3_PROJECTIONS_SM100[key]
    cute_tokens, dsv3_tokens = EXPECTED_SELECTIONS_SM100[key]
    for num_tokens in range(1, 17):
        backend = k3_gemm._backend_for(spec, num_tokens, has_residual=False)
        if num_tokens in cute_tokens:
            assert backend == "cute"
        elif num_tokens in dsv3_tokens:
            assert backend == "dsv3_fused_a"
        else:
            assert backend is None


@pytest.mark.parametrize(
    "table",
    [k3_gemm.KIMI_K3_PROJECTIONS_SM100, k3_gemm.KIMI_K3_PROJECTIONS_SM90],
)
def test_device_build_plan_matches_table(table: K3ProjectionTable) -> None:
    for spec in table.values():
        plan = k3_gemm._build_plan(spec)
        for num_tokens in range(1, 17):
            backend = k3_gemm._backend_for(spec, num_tokens, has_residual=False)
            if backend is None:
                assert num_tokens not in plan
            else:
                assert plan[num_tokens][0] == backend


def test_sm90_table_covers_measured_points() -> None:
    specs = k3_gemm.KIMI_K3_PROJECTIONS_SM90.values()
    assert len(k3_gemm.KIMI_K3_PROJECTIONS_SM90) == 18
    assert sum(len(spec.cute_configs) + len(spec.dsv3_tokens) for spec in specs) == 90


def test_low_latency_table_capability_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: True)
    assert k3_gemm._low_latency_table() is k3_gemm.KIMI_K3_PROJECTIONS

    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: False)
    monkeypatch.setattr(
        k3_gemm.current_platform,
        "is_device_capability",
        lambda cc: cc == (10, 0),
    )
    assert k3_gemm._low_latency_table() is k3_gemm.KIMI_K3_PROJECTIONS_SM100

    monkeypatch.setattr(
        k3_gemm.current_platform,
        "is_device_capability",
        lambda cc: cc == (9, 0),
    )
    assert k3_gemm._low_latency_table() is k3_gemm.KIMI_K3_PROJECTIONS_SM90

    monkeypatch.setattr(
        k3_gemm.current_platform, "is_device_capability", lambda cc: False
    )
    assert k3_gemm._low_latency_table() is None


def test_qwen4_exp_hopper_plans_are_valid() -> None:
    plans = qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS

    assert len(plans) == 9
    assert sum(map(len, plans.values())) == 31
    assert (320, 10240) in plans
    assert (10240, 320) not in plans
    for (n, k), shape_plans in plans.items():
        for num_tokens, config in shape_plans.items():
            assert config.num_rows == num_tokens
            assert n % config.outputs_per_block == 0
            assert k % (config.block_size * config.vector_width) == 0
            assert config.static_k in (None, k)


@pytest.mark.parametrize(
    "capability,expected_plans",
    [
        ((10, 3), qwen4_exp_gemm.QWEN4_EXP_GEMM_PLANS),
        ((9, 0), qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS),
        ((8, 0), {}),
    ],
)
def test_qwen4_exp_gemm_capability_routing(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected_plans: dict[tuple[int, int], dict[int, SkinnyGemmConfig]],
) -> None:
    monkeypatch.setattr(
        qwen4_exp_gemm.current_platform,
        "is_device_capability",
        lambda target: capability == target,
    )

    assert qwen4_exp_gemm._gemm_plans() == expected_plans


def test_installation_is_shape_specific_and_unquantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinear(nn.Module):
        def __init__(self, quant_method: object, n: int, k: int) -> None:
            super().__init__()
            self.quant_method = quant_method
            self.weight = torch.empty(n, k)

    class FakeHead(nn.Module):
        def __init__(self, n: int, k: int) -> None:
            super().__init__()
            self.quant_method = k3_gemm.UnquantizedEmbeddingMethod()
            self.weight = torch.empty(n, k)

    root = nn.Module()
    # dsv3-only shape (no cute warmup contribution).
    root.dsv3_only = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 2304, 1536)
    # quantized: must be left untouched.
    quantized_method = object()
    root.quantized = FakeLinear(quantized_method, 6288, 7168)
    # cute shape.
    root.cute = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 6288, 7168)
    # cute + residual shape.
    root.residual = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 7168, 3584)
    # shape absent from the table: must be left untouched.
    root.unlisted = FakeLinear(k3_gemm.UnquantizedLinearMethod(), 1234, 5678)
    root.lm_head = FakeHead(20480, 7168)

    monkeypatch.setattr(k3_gemm, "LinearBase", FakeLinear)
    monkeypatch.setattr(k3_gemm, "ParallelLMHead", FakeHead)
    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: True)
    warmup_configs: set[SkinnyGemmConfig] = set()
    residual_warmup_configs: set[SkinnyGemmConfig] = set()
    monkeypatch.setattr(k3_gemm.shape_dynamic_skinny_gemm, "is_available", lambda: True)

    def request_warmup_configs(dtype, configs, *, has_residual=False):
        target = residual_warmup_configs if has_residual else warmup_configs
        target.update(configs)

    monkeypatch.setattr(
        k3_gemm.shape_dynamic_skinny_gemm,
        "request_warmup_configs",
        request_warmup_configs,
    )

    k3_gemm.enable_kimi_k3_low_latency_gemm(root, torch.bfloat16)

    assert isinstance(root.dsv3_only.quant_method, k3_gemm.KimiK3LowLatencyLinearMethod)
    assert isinstance(root.cute.quant_method, k3_gemm.KimiK3LowLatencyLinearMethod)
    assert isinstance(root.residual.quant_method, k3_gemm.KimiK3LowLatencyLinearMethod)
    assert root.quantized.quant_method is quantized_method
    assert type(root.unlisted.quant_method) is k3_gemm.UnquantizedLinearMethod
    assert isinstance(
        root.lm_head.quant_method, k3_gemm.KimiK3LowLatencyEmbeddingMethod
    )
    # Warmup covers only the installed modules' local (N, K).
    assert warmup_configs == {
        config
        for key in ((6288, 7168), (7168, 3584), (20480, 7168))
        for _, config in k3_gemm.KIMI_K3_PROJECTIONS[key].cute_configs
    }
    assert residual_warmup_configs == {
        config
        for _, config in k3_gemm.KIMI_K3_PROJECTIONS[(7168, 3584)].residual_configs
    }


@pytest.mark.parametrize(
    "dtype,platform_enabled",
    [(torch.float16, True), (torch.bfloat16, False)],
)
def test_installation_requires_bf16_sm103(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    platform_enabled: bool,
) -> None:
    class FakeLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = k3_gemm.UnquantizedLinearMethod()
            self.weight = torch.empty(2304, 1536)

    root = nn.Module()
    root.projection = FakeLinear()
    monkeypatch.setattr(k3_gemm, "LinearBase", FakeLinear)
    monkeypatch.setattr(k3_gemm, "_is_sm103", lambda: platform_enabled)
    # Pin the rest of the capability probe so platform_enabled=False stays a
    # no-op even when the test itself runs on SM100 hardware.
    monkeypatch.setattr(
        k3_gemm.current_platform,
        "is_device_capability",
        lambda cc: platform_enabled and cc == (10, 3),
    )

    k3_gemm.enable_kimi_k3_low_latency_gemm(root, dtype)

    assert type(root.projection.quant_method) is k3_gemm.UnquantizedLinearMethod


def _require_capability_and_dsv3(capability: tuple[int, int]) -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != capability
    ):
        pytest.skip(f"Kimi-K3 selection requires SM{capability[0]}{capability[1]}")
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        pytest.skip("dsv3_fused_a_gemm was not built")


def _require_capability_and_cute(capability: tuple[int, int]) -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != capability
    ):
        pytest.skip(f"CuTe DSL selection requires SM{capability[0]}{capability[1]}")
    if not k3_gemm.shape_dynamic_skinny_gemm.is_available():
        pytest.skip("CuTe DSL is not available")


def _require_sm103_and_dsv3() -> None:
    _require_capability_and_dsv3((10, 3))


def _require_sm103_and_cute() -> None:
    _require_capability_and_cute((10, 3))


@pytest.mark.parametrize("spec,config", GLM_CUTE_CASES)
def test_glm_cute_selected_shapes(
    spec: glm52_gemm.GLM52ProjectionSpec,
    config: SkinnyGemmConfig,
) -> None:
    _require_sm103_and_cute()
    torch.manual_seed(42)
    x = torch.randn(
        config.num_rows,
        spec.k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    plan = spec.build_plan()

    output = glm52_gemm.run_glm52_plan(plan, x, weight)

    assert output is not None
    reference = x.float() @ weight.float().t()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("n,k,num_tokens,config", QWEN4_EXP_SM90_CASES)
def test_qwen4_exp_sm90_selected_shapes(
    n: int,
    k: int,
    num_tokens: int,
    config: SkinnyGemmConfig,
) -> None:
    _require_capability_and_cute((9, 0))
    torch.manual_seed(42 + num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    selected = qwen4_exp_gemm.QWEN4_EXP_SM90_GEMM_PLANS[(n, k)][num_tokens]
    assert selected == config
    output = qwen4_exp_gemm._qwen4_exp_low_latency_gemm(x, weight)

    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def test_glm52_q_b_nonpacked_single_row_falls_back() -> None:
    _require_sm103_and_cute()
    spec = glm52_gemm.GLM52_Q_B_PROJECTION
    storage = torch.randn(
        1,
        glm52_gemm.GLM52_QKV_A_PROJECTION.n,
        dtype=torch.bfloat16,
        device="cuda",
    )
    x = storage[:, : spec.k]
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")

    assert x.stride() == (glm52_gemm.GLM52_QKV_A_PROJECTION.n, 1)
    assert not glm52_gemm._runtime_ok(x, weight)
    output = glm52_gemm.run_glm52_plan(spec.build_plan(), x, weight)

    assert output is None


@pytest.mark.parametrize("spec,config", GLM_CUTE_CASES)
def test_glm_cute_selected_shapes_cuda_graph_capture(
    spec: glm52_gemm.GLM52ProjectionSpec,
    config: SkinnyGemmConfig,
) -> None:
    _require_sm103_and_cute()
    x = torch.randn(
        config.num_rows,
        spec.k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    plan = spec.build_plan()
    glm52_gemm.run_glm52_plan(plan, x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = glm52_gemm.run_glm52_plan(plan, x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert output is not None
    reference = x.float() @ weight.float().t()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("capability,n,k,num_tokens", CUTE_CASES)
def test_cute_selected_shapes(
    capability: tuple[int, int], n: int, k: int, num_tokens: int
) -> None:
    _require_capability_and_cute(capability)
    torch.manual_seed(42)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    output = k3_gemm.try_low_latency_gemm(x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14])
def test_kda_projection_overlap_cuda_graph(num_tokens: int) -> None:
    """The TP8 split must preserve projection results across both streams."""
    _require_sm103_and_cute()
    _require_sm103_and_dsv3()
    torch.manual_seed(43)
    hidden_states = torch.randn(num_tokens, 7168, dtype=torch.bfloat16, device="cuda")
    packed_weight = torch.randn(6288, 7168, dtype=torch.bfloat16, device="cuda")
    f_b_weight = torch.randn(1536, 128, dtype=torch.bfloat16, device="cuda")
    aux_stream = torch.cuda.Stream()
    events = (torch.cuda.Event(), torch.cuda.Event())

    k3_gemm.run_kda_projection_overlap(
        hidden_states,
        packed_weight,
        f_b_weight,
        aux_stream,
        events,
    )
    torch.accelerator.synchronize()

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        projected_qkvg, g1, beta = k3_gemm.run_kda_projection_overlap(
            hidden_states,
            packed_weight,
            f_b_weight,
            aux_stream,
            events,
        )
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph.replay()
    torch.accelerator.synchronize()

    assert beta.stride() == ((140 if num_tokens == 1 else 144), 1)

    expected_qkvg = torch.nn.functional.linear(
        hidden_states.float(), packed_weight[:6144].float()
    )
    expected_fab = torch.nn.functional.linear(
        hidden_states.float(), packed_weight[6144:6284].float()
    )
    expected_g1 = torch.nn.functional.linear(
        expected_fab[:, :128].to(torch.bfloat16).float(),
        f_b_weight.float(),
    )
    for actual, expected in (
        (projected_qkvg, expected_qkvg),
        (g1, expected_g1),
        (beta, expected_fab[:, 128:]),
    ):
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.flatten(), dim=0
        ).item()
        assert cosine > 0.999


def _dsv3_probe_tokens(tokens: frozenset[int]) -> set[int]:
    """Extremes, plus both sides of the kernel's num_tokens<=8 tile_n branch."""
    if not tokens:
        return set()
    return {min(tokens), max(tokens)} | ({8, 9} & set(tokens))


# Derived from the table rather than hand-listed, so a shape routed to dsv3
# cannot be added without being exercised here.
DSV3_CASES = sorted(
    (capability, num_tokens, spec.n, spec.k)
    for capability, table in K3_GPU_TABLES
    for spec in table.values()
    for num_tokens in _dsv3_probe_tokens(spec.dsv3_tokens)
)

GLM_DSV3_CASES = [
    (num_tokens, spec)
    for spec in (
        glm52_gemm.GLM52_QKV_A_PROJECTION,
        glm52_gemm.GLM52_Q_B_PROJECTION,
    )
    for num_tokens in sorted(_dsv3_probe_tokens(spec.dsv3_tokens))
]


@pytest.mark.parametrize("capability,num_tokens,n,k", DSV3_CASES)
def test_dsv3_selected_shapes(
    capability: tuple[int, int], num_tokens: int, n: int, k: int
) -> None:
    _require_capability_and_dsv3(capability)
    torch.manual_seed(42)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")

    output = k3_gemm.try_low_latency_gemm(x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("num_tokens,spec", GLM_DSV3_CASES)
def test_glm_dsv3_selected_shapes(
    num_tokens: int,
    spec: glm52_gemm.GLM52ProjectionSpec,
) -> None:
    _require_sm103_and_dsv3()
    torch.manual_seed(42)
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")

    output = glm52_gemm.run_glm52_plan(spec.build_plan(), x, weight)

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


def test_nonpacked_single_token_dsv3_falls_back() -> None:
    _require_sm103_and_dsv3()
    n, k = 1536, 128
    storage = torch.randn(1, k + 16, dtype=torch.bfloat16, device="cuda")
    x = storage[:, :k]
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(n, k)]
    method = k3_gemm.KimiK3LowLatencyLinearMethod(
        k3_gemm._build_plan(spec), k3_gemm._build_residual_plan(spec)
    )

    assert x.is_contiguous()
    assert x.stride() == (k + 16, 1)
    assert not k3_gemm._runtime_ok(x, weight)  # strict guard rejects the view
    output = method.apply(SimpleNamespace(weight=weight), x)

    reference = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(output, reference)


@pytest.mark.parametrize("capability,table", K3_GPU_TABLES)
def test_selected_kernels_cuda_graph_capture(
    capability: tuple[int, int], table: K3ProjectionTable
) -> None:
    _require_capability_and_cute(capability)
    _require_capability_and_dsv3(capability)
    cute_spec = table[(6288, 7168)]
    dsv3_spec = table[(1536, 128)]
    cute_x = torch.randn(1, cute_spec.k, dtype=torch.bfloat16, device="cuda")
    cute_weight = torch.randn(
        cute_spec.n, cute_spec.k, dtype=torch.bfloat16, device="cuda"
    )
    dsv3_x = torch.randn(1, dsv3_spec.k, dtype=torch.bfloat16, device="cuda")
    dsv3_weight = torch.randn(
        dsv3_spec.n, dsv3_spec.k, dtype=torch.bfloat16, device="cuda"
    )
    k3_gemm.try_low_latency_gemm(cute_x, cute_weight)
    k3_gemm.try_low_latency_gemm(dsv3_x, dsv3_weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute_output = k3_gemm.try_low_latency_gemm(cute_x, cute_weight)
        dsv3_output = k3_gemm.try_low_latency_gemm(dsv3_x, dsv3_weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert cute_output is not None
    assert dsv3_output is not None
    for output, activation, weight in (
        (cute_output, cute_x, cute_weight),
        (dsv3_output, dsv3_x, dsv3_weight),
    ):
        reference = torch.nn.functional.linear(activation, weight)
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        ).item()
        assert cosine > 0.999


@pytest.mark.parametrize("num_tokens", [1, 8, 9, 16])
def test_dsv3_cuda_graph_capture_tile_branches(num_tokens: int) -> None:
    """Capture DSV3 across the num_tokens<=8 vs >8 tile_n branch."""
    _require_sm103_and_dsv3()
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(1536, 128)]
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    k3_gemm.try_low_latency_gemm(x, weight)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = k3_gemm.try_low_latency_gemm(x, weight)
    graph.replay()
    torch.accelerator.synchronize()

    assert output is not None
    reference = torch.nn.functional.linear(x, weight)
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("n,k,num_tokens", RESIDUAL_CUTE_CASES)
def test_cute_residual_epilogue(n: int, k: int, num_tokens: int) -> None:
    _require_sm103_and_cute()
    torch.manual_seed(42 + num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, n, dtype=torch.bfloat16, device="cuda")
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(n, k)]
    config = spec.residual_config(num_tokens)
    assert config is not None

    output = k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)

    reference = x.float() @ weight.float().t() + residual.float()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.flatten(), dim=0
    ).item()
    assert cosine > 0.999


@pytest.mark.parametrize("num_tokens", range(1, 17))
def test_cute_residual_epilogue_all_supported_token_counts(num_tokens: int) -> None:
    _require_sm103_and_cute()
    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
        ShapeDynamicSkinnyGemm,
    )

    n, k = 64, 512
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, n, dtype=torch.bfloat16, device="cuda")
    config = ShapeDynamicSkinnyGemm._config(num_tokens, n, k)

    output = k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)

    reference = x.float() @ weight.float().t() + residual.float()
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-1)


@pytest.mark.parametrize("num_tokens", range(1, 5))
def test_cute_residual_epilogue_cuda_graph_capture(num_tokens: int) -> None:
    _require_sm103_and_cute()
    spec = k3_gemm.KIMI_K3_PROJECTIONS[(7168, 3584)]
    config = spec.residual_config(num_tokens)
    assert config is not None
    x = torch.randn(num_tokens, spec.k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(spec.n, spec.k, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(num_tokens, spec.n, dtype=torch.bfloat16, device="cuda")
    k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = k3_gemm.shape_dynamic_skinny_gemm(x, weight, config, residual)
    graph.replay()
    torch.accelerator.synchronize()

    reference = x.float() @ weight.float().t() + residual.float()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.flatten(), dim=0
    ).item()
    assert cosine > 0.999


class _SkinnyGemmSpy:
    """Wraps the skinny-GEMM singleton to record whether CuTe was invoked."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.calls: list[int] = []

    def __call__(self, a, b, config=None, residual=None):
        self.calls.append(a.shape[0])
        return self._real(a, b, config, residual)

    def is_available(self) -> bool:
        return self._real.is_available()


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4])
def test_latent_moe_production_layout_residual(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
) -> None:
    """The real Latent-MoE residual is a non-packed slice of a cat buffer.

    The strict packed-row-major guard rejects such a slice at every token count
    (a size-1 leading dim reads as contiguous but its stride is not packed), so
    the CuTe residual epilogue never fires for this production layout and the
    method falls back to addmm. Output is correct regardless of the path.
    """
    _require_sm103_and_cute()
    latent_dim, shared_dim = 3584, 7168  # routed_expert_up_proj K, N
    torch.manual_seed(7 + num_tokens)
    buf = torch.randn(
        num_tokens, latent_dim + shared_dim, dtype=torch.bfloat16, device="cuda"
    )
    latent = buf[:, :latent_dim]  # non-contiguous view (row stride = full width)
    residual = buf[:, latent_dim:]  # non-contiguous view
    weight = torch.randn(shared_dim, latent_dim, dtype=torch.bfloat16, device="cuda")

    spec = k3_gemm.KIMI_K3_PROJECTIONS[(shared_dim, latent_dim)]
    method = k3_gemm.KimiK3LowLatencyLinearMethod(
        k3_gemm._build_plan(spec), k3_gemm._build_residual_plan(spec)
    )
    spy = _SkinnyGemmSpy(k3_gemm.shape_dynamic_skinny_gemm)
    monkeypatch.setattr(k3_gemm, "shape_dynamic_skinny_gemm", spy)

    layer = SimpleNamespace(weight=weight)
    output = method.apply_with_residual(layer, latent, residual)

    reference = latent.float() @ weight.float().t() + residual.float()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(), reference.flatten(), dim=0
    ).item()
    assert cosine > 0.999  # correct regardless of the path taken
    assert not spy.calls, (
        "non-packed buf-slice residual must fall back to addmm at every M"
    )


def test_residual_dispatch_falls_back_to_addmm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback = torch.randn(2, 3)
    residual = torch.randn(2, 3)
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    monkeypatch.setattr(torch, "addmm", lambda *args: fallback)
    # CPU tensors fail the runtime check, forcing the addmm fallback.
    method = k3_gemm.KimiK3LowLatencyLinearMethod({}, {})

    output = method.apply_with_residual(SimpleNamespace(weight=weight), x, residual)

    assert output is fallback


def test_fallback_preserves_default_method() -> None:
    x = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    # CPU tensors fail the runtime check, forcing the base-method fallback.
    method = k3_gemm.KimiK3LowLatencyLinearMethod({}, {})

    output = method.apply(SimpleNamespace(weight=weight), x)

    torch.testing.assert_close(output, torch.nn.functional.linear(x, weight))
