# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MEGA_MOE_NVFP4,
)
from vllm.model_executor.warmup import deep_gemm_warmup


def _block_fp8_layer(n: int, k: int, scale_name: str = "weight_scale"):
    layer = SimpleNamespace(
        weight=torch.empty((n, k), dtype=torch.float8_e4m3fn),
        weight_block_size=[128, 128],
        deep_gemm_warmup_provider=object.__new__(DeepGemmFp8BlockScaledMMKernel),
    )
    setattr(layer, scale_name, torch.empty((n // 128, k // 128), dtype=torch.float32))
    return layer


def _run_warmup(monkeypatch, layers) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        deep_gemm_warmup,
        "get_mk_alignment_for_contiguous_layout",
        lambda: [128, 128],
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_deepgemm_fp8_gemm_nt_warmup",
        lambda **kwargs: calls.append(kwargs),
    )
    model = SimpleNamespace(modules=lambda: iter(layers))
    deep_gemm_warmup.deepgemm_fp8_gemm_nt_warmup(model, max_tokens=16)
    return calls


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_registered_deep_gemm_layer_is_warmed_up(monkeypatch, scale_name) -> None:
    """Any layer stamped by the DeepGEMM kernel is warmed, regardless of the
    quantization method that owns it or the name of its scale parameter."""
    layer = _block_fp8_layer(256, 128, scale_name)

    calls = _run_warmup(monkeypatch, [layer])

    assert len(calls) == 1
    assert calls[0]["w"] is layer.weight
    assert calls[0]["ws"] is getattr(layer, scale_name)


def test_n_multiple_of_64_matches_kernel_selection(monkeypatch) -> None:
    """DeepGEMM accepts N % 64 == 0 (e.g. DeepSeek kv_a_proj_with_mqa, N=576),
    so warmup must not require N % 128 == 0."""
    assert len(_run_warmup(monkeypatch, [_block_fp8_layer(576, 256)])) == 1


def test_unstamped_and_mismatched_block_layers_are_skipped(monkeypatch) -> None:
    mismatched = _block_fp8_layer(256, 128)
    mismatched.weight_block_size = [1, 32]
    layers = [SimpleNamespace(), mismatched, _block_fp8_layer(256, 128)]

    calls = _run_warmup(monkeypatch, layers)

    assert len(calls) == 1
    assert calls[0]["w"] is layers[-1].weight


@pytest.mark.parametrize("is_bmm", [False, True])
def test_kernel_registers_itself_as_warmup_provider(is_bmm) -> None:
    """Bmm layers bypass the kernel at runtime, so they must not be warmed."""
    kernel = object.__new__(DeepGemmFp8BlockScaledMMKernel)
    kernel.is_deep_gemm_supported = False
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((256, 128), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((2, 1), dtype=torch.float32), requires_grad=False
    )
    layer.weight_block_size = [128, 128]
    layer.is_bmm = is_bmm

    kernel.process_weights_after_loading(layer)

    provider = getattr(layer, "deep_gemm_warmup_provider", None)
    assert provider is (None if is_bmm else kernel)


def test_mega_moe_predicate_matches_only_native_mega_moe_layers(monkeypatch) -> None:
    """Without the MegaMoE model modules loaded, nothing matches and nothing is
    imported; with them loaded, only DeepSeek-V4, Kimi K3 and MiniMax M3 MegaMoE
    layers match."""
    name = "vllm.models.deepseek_v4.nvidia.model"
    kimi_name = "vllm.models.kimi_k3.nvidia.model"
    minimax_name = "vllm.models.minimax_m3.nvidia.model"
    for model_module in (name, kimi_name, minimax_name):
        monkeypatch.delitem(sys.modules, model_module, raising=False)
    other = SimpleNamespace(use_native_mega_moe=True, use_mega_moe=True)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(other)
    assert not {name, kimi_name, minimax_name} & set(sys.modules)

    moe_cls = type("DeepseekV4MoE", (), {})
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(DeepseekV4MoE=moe_cls))
    native, fused = moe_cls(), moe_cls()
    native.use_native_mega_moe, fused.use_native_mega_moe = True, False
    assert deep_gemm_warmup._mega_moe_may_use_deep_gemm(native)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(fused)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(other)

    kimi_cls = type("KimiMoE", (), {})
    monkeypatch.setitem(sys.modules, kimi_name, SimpleNamespace(KimiMoE=kimi_cls))
    kimi_mega, kimi_fused = kimi_cls(), kimi_cls()
    kimi_mega.use_mega_moe, kimi_fused.use_mega_moe = True, False
    assert deep_gemm_warmup._mega_moe_may_use_deep_gemm(kimi_mega)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(kimi_fused)

    minimax_cls = type("MiniMaxM3MoE", (), {})
    monkeypatch.setitem(
        sys.modules, minimax_name, SimpleNamespace(MiniMaxM3MoE=minimax_cls)
    )
    minimax_mega, minimax_fused = minimax_cls(), minimax_cls()
    minimax_mega.use_mega_moe, minimax_fused.use_mega_moe = True, False
    assert deep_gemm_warmup._mega_moe_may_use_deep_gemm(minimax_mega)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(minimax_fused)


@pytest.mark.parametrize(
    ("weight_format", "expected"),
    [(MEGA_MOE_MXFP4, [2, 5]), (MEGA_MOE_NVFP4, [1, 2, 3, 5])],
)
def test_mega_moe_m_values_cover_nvfp4_launch_variants(
    monkeypatch, weight_format, expected
) -> None:
    """NVFP4 switches from the decode-shaped to the bulk kernel inside one BLOCK_M,
    so both ends are warmed; MX kernels depend on BLOCK_M alone."""
    monkeypatch.setattr(
        deep_gemm_warmup, "_import_deep_gemm", lambda: SimpleNamespace()
    )
    module = SimpleNamespace(
        gate=SimpleNamespace(),
        experts=SimpleNamespace(
            get_block_m=lambda num_tokens: 64 if num_tokens <= 2 else 128,
            weight_format=weight_format,
        ),
        use_sequence_parallel=False,
    )
    assert deep_gemm_warmup._get_fp8_fp4_mega_moe_m_values(module, 5) == expected


def test_mega_moe_m_values_reach_mega_gate(monkeypatch) -> None:
    """Each bf16_mega_gate config is warmed with an M above the GateLinear
    threshold (1 for 128 experts), so the forward actually reaches the kernel."""
    dg = SimpleNamespace(
        get_bf16_mega_gate_config=lambda m, *_: {"num_sms": -(-m // 16)},
    )
    monkeypatch.setattr(deep_gemm_warmup, "_import_deep_gemm", lambda: dg)
    monkeypatch.setattr(deep_gemm_warmup, "_mega_moe_uses_mega_gate", lambda _: True)
    module = SimpleNamespace(
        gate=SimpleNamespace(input_size=4096, output_size=128),
        experts=SimpleNamespace(
            top_k=6,
            get_block_m=lambda num_tokens: 16,
            weight_format=MEGA_MOE_MXFP4,
        ),
        use_sequence_parallel=False,
    )
    assert deep_gemm_warmup._get_fp8_fp4_mega_moe_m_values(module, 40) == [16, 32, 40]


@pytest.fixture
def clear_fp8_einsum_warmup_cache():
    deep_gemm_warmup.FP8_EINSUM_WARMUP_CACHE.clear()
    yield
    deep_gemm_warmup.FP8_EINSUM_WARMUP_CACHE.clear()


def _dsv4_attention(
    *,
    weight_dtype: torch.dtype = torch.float8_e4m3fn,
    is_bmm: bool = True,
    recipe: tuple[int, int, int] | None = (1, 128, 128),
    tma_aligned_scales: bool = False,
    quantize_o_proj: bool = False,
):
    """Minimal stand-in for a DeepSeek-V4 attention layer's o_proj inputs."""
    n_groups, o_lora_rank, d = 2, 128, 256
    wo_a = SimpleNamespace(
        weight=torch.empty((n_groups, o_lora_rank, d), dtype=weight_dtype),
        weight_scale_inv=torch.empty(
            (n_groups, o_lora_rank // 128, d // 128), dtype=torch.float32
        ),
        is_bmm=is_bmm,
    )
    return SimpleNamespace(
        wo_a=wo_a,
        _einsum_recipe=recipe,
        _tma_aligned_scales=tma_aligned_scales,
        _can_quantize_o_proj=quantize_o_proj,
    )


def test_fp8_einsum_data_matches_only_fp8_bmm_o_proj() -> None:
    assert deep_gemm_warmup._fp8_einsum_warmup_data(_dsv4_attention()) is not None
    assert (
        deep_gemm_warmup._fp8_einsum_warmup_data(
            _dsv4_attention(weight_dtype=torch.bfloat16)
        )
        is None
    )
    assert (
        deep_gemm_warmup._fp8_einsum_warmup_data(_dsv4_attention(is_bmm=False)) is None
    )
    assert (
        deep_gemm_warmup._fp8_einsum_warmup_data(_dsv4_attention(recipe=None)) is None
    )
    assert deep_gemm_warmup._fp8_einsum_warmup_data(SimpleNamespace()) is None


def test_fp8_einsum_relax_m_values_cover_block_m_boundaries(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "relax")
    seen_n = []

    def fake_optimal(max_tokens, n, device):
        seen_n.append(n)
        return [1, 64, 513, 641, 1025, max_tokens]

    monkeypatch.setattr(
        deep_gemm_warmup, "_generate_optimal_warmup_m_values", fake_optimal
    )
    weight = torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)

    m_values = deep_gemm_warmup._get_fp8_einsum_m_values(weight, 2048)

    # N for the boundary heuristic is the einsum output width (o_lora_rank).
    assert seen_n == [128]
    assert m_values == sorted({*range(1, 2049, 16), 64, 2048})


def test_fp8_einsum_relax_m_values_cover_measured_boundaries(monkeypatch) -> None:
    """Every M DeepGEMM was measured to switch einsum config at, SM90 and SM100."""
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "relax")
    monkeypatch.setattr(
        deep_gemm_warmup, "_generate_optimal_warmup_m_values", lambda *args: []
    )
    weight = torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)

    m_values = set(deep_gemm_warmup._get_fp8_einsum_m_values(weight, 8192))

    sm90 = {1, 17, 33, 65, 129, 257, 513, 641, 1025}
    sm100 = {1, 33, 65, 81, 97, 129, 161, 193, 641, 2161, *range(225, 482, 32)}
    assert sm90 | sm100 <= m_values


def test_fp8_einsum_full_m_values_cover_every_m(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "full")
    weight = torch.empty((2, 128, 256), dtype=torch.float8_e4m3fn)
    assert deep_gemm_warmup._get_fp8_einsum_m_values(weight, 5) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    ("recipe", "tma_aligned_scales", "scale_dtype", "scale_inner"),
    [
        ((1, 128, 128), False, torch.float32, 2),  # SM90: FP32 block scales
        ((1, 1, 128), True, torch.int32, 1),  # SM100: INT32-packed UE8M0
    ],
)
def test_fp8_einsum_warmup_matches_runtime_layout(
    monkeypatch,
    clear_fp8_einsum_warmup_cache,
    recipe,
    tma_aligned_scales,
    scale_dtype,
    scale_inner,
) -> None:
    monkeypatch.setattr(
        deep_gemm_warmup, "_get_fp8_einsum_m_values", lambda weight, max_tokens: [1, 3]
    )
    calls = []
    monkeypatch.setattr(
        deep_gemm_warmup,
        "fp8_einsum",
        lambda eq, a, b, out, *, recipe, use_128x4_sfd_layout: calls.append(
            (eq, a, b, out, recipe, use_128x4_sfd_layout)
        ),
    )
    attn = _dsv4_attention(recipe=recipe, tma_aligned_scales=tma_aligned_scales)
    weight, weight_scale = attn.wo_a.weight, attn.wo_a.weight_scale_inv

    for _ in range(2):  # the second call is a cache hit
        deep_gemm_warmup._deepgemm_fp8_einsum_warmup(
            weight, weight_scale, recipe, tma_aligned_scales, False, max_tokens=3
        )

    assert len(calls) == 2
    for num_tokens, (eq, (a, a_scale), b, out, actual_recipe, sfd) in zip(
        [1, 3], calls
    ):
        aligned_t = deep_gemm_warmup.get_tma_aligned_size(num_tokens, 4)
        assert eq == "bhr,hdr->bhd"
        # fused_inv_rope_fp8_quant output: [T, G, D] over a [G, T, D] buffer.
        assert a.shape == (num_tokens, 2, 256)
        assert a.dtype == torch.float8_e4m3fn
        assert a.stride() == (256, num_tokens * 256, 1)
        assert a_scale.shape == (num_tokens, 2, scale_inner)
        assert a_scale.dtype == scale_dtype
        assert a_scale.stride() == (1, scale_inner * aligned_t, aligned_t)
        assert b[0] is weight and b[1] is weight_scale
        assert out.shape == (num_tokens, 2, 128) and out.dtype == torch.bfloat16
        assert actual_recipe == recipe and sfd is False


def test_fp8_einsum_data_reports_the_quantized_o_proj_variant() -> None:
    bf16_out = deep_gemm_warmup._fp8_einsum_warmup_data(_dsv4_attention())
    fp8_out = deep_gemm_warmup._fp8_einsum_warmup_data(
        _dsv4_attention(
            recipe=(1, 1, 32), tma_aligned_scales=True, quantize_o_proj=True
        )
    )
    assert bf16_out is not None and fp8_out is not None
    assert bf16_out[4] is False and fp8_out[4] is True
    # The two variants are different kernels, so they must not share a key.
    assert deep_gemm_warmup._get_fp8_einsum_warmup_key(
        *bf16_out
    ) != deep_gemm_warmup._get_fp8_einsum_warmup_key(*fp8_out)


def test_fp8_einsum_warmup_matches_quantized_o_proj_layout(
    monkeypatch, clear_fp8_einsum_warmup_cache
) -> None:
    """DSv4.1 mega attention takes an FP8 (out, scale) pair in the 128x4 layout."""
    monkeypatch.setattr(
        deep_gemm_warmup, "_get_fp8_einsum_m_values", lambda weight, max_tokens: [200]
    )
    calls = []
    monkeypatch.setattr(
        deep_gemm_warmup,
        "fp8_einsum",
        lambda eq, a, b, out, *, recipe, use_128x4_sfd_layout: calls.append(
            (out, use_128x4_sfd_layout)
        ),
    )
    attn = _dsv4_attention(
        recipe=(1, 1, 32), tma_aligned_scales=True, quantize_o_proj=True
    )

    data = deep_gemm_warmup._fp8_einsum_warmup_data(attn)
    assert data is not None
    weight, weight_scale, recipe, tma_aligned_scales, quantize_output = data

    deep_gemm_warmup._deepgemm_fp8_einsum_warmup(
        weight,
        weight_scale,
        recipe,
        tma_aligned_scales,
        quantize_output,
        max_tokens=200,
    )

    ((out, use_128x4_sfd_layout),) = calls
    assert use_128x4_sfd_layout is True
    z, z_scale = out
    n_groups, o_lora_rank = 2, 128
    assert z.shape == (200, n_groups, o_lora_rank)
    assert z.dtype == torch.float8_e4m3fn
    # DeepGEMM asserts d.stride(0) == o_lora_rank on the permuted output.
    assert z.permute(1, 0, 2).stride(0) == o_lora_rank
    assert z_scale.shape == (2, n_groups * o_lora_rank // 128, 128)
    assert z_scale.dtype == torch.int32 and z_scale.is_contiguous()


def test_fp8_einsum_warmup_covers_both_output_variants(
    monkeypatch, clear_fp8_einsum_warmup_cache
) -> None:
    """A target and a draft layer can disagree on the quantized o_proj."""
    monkeypatch.setattr(
        deep_gemm_warmup, "_get_fp8_einsum_m_values", lambda weight, max_tokens: [1]
    )
    seen = []
    monkeypatch.setattr(
        deep_gemm_warmup,
        "fp8_einsum",
        lambda eq, a, b, out, *, recipe, use_128x4_sfd_layout: seen.append(
            use_128x4_sfd_layout
        ),
    )
    layers = [
        _dsv4_attention(recipe=(1, 1, 32), tma_aligned_scales=True),
        _dsv4_attention(
            recipe=(1, 1, 32), tma_aligned_scales=True, quantize_o_proj=True
        ),
    ]
    model = SimpleNamespace(modules=lambda: iter(layers))

    deep_gemm_warmup.deepgemm_fp8_einsum_warmup(model, 1)

    assert seen == [False, True]


def test_count_warmup_iterations_counts_each_fp8_einsum_shape_once(
    monkeypatch, clear_fp8_einsum_warmup_cache
) -> None:
    layers = [_dsv4_attention(), _dsv4_attention()]  # e.g. two decoder layers

    def fake_data(m):
        if not hasattr(m, "wo_a"):
            return None
        return (m.wo_a.weight, m.wo_a.weight_scale_inv, m._einsum_recipe, False, False)

    monkeypatch.setattr(deep_gemm_warmup, "_fp8_einsum_warmup_data", fake_data)
    monkeypatch.setattr(deep_gemm_warmup, "_deep_gemm_linear_data", lambda m: None)
    monkeypatch.setattr(
        deep_gemm_warmup, "_fused_moe_grouped_gemm_may_use_deep_gemm", lambda m: False
    )
    monkeypatch.setattr(
        deep_gemm_warmup, "_mega_moe_may_use_deep_gemm", lambda m: False
    )
    monkeypatch.setattr(
        deep_gemm_warmup, "_get_fp8_einsum_m_values", lambda w, max_tokens: [1, 2, 3]
    )
    model = SimpleNamespace(modules=lambda: iter(layers))

    assert deep_gemm_warmup._count_warmup_iterations(model, 3) == 3
