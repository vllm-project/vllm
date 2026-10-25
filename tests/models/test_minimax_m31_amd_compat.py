# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for MiniMax-M3.1 AMD compatibility: config, DSpark, NVFP4→MXFP4."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.models.minimax_m3.common.dspark_mapper import map_dspark_weight_name
from vllm.models.minimax_m3.common.nvfp4_mxfp4 import (
    FP4_PACK_FACTOR,
    MINIMAX_NVFP4_EXPERT_TENSORS,
    MINIMAX_ROUTED_EXPERT_CKPT,
    MXFP4_GROUP_SIZE,
    NVFP4_GROUP_SIZE,
    ROCM_AITER_MXFP4_ALIGN,
    aiter_mxfp4_padded_dims,
    copy_padded_w13_w2,
    down_w2_shape,
    fused_w13_shape,
    mxfp4_packed_cols,
    mxfp4_scale_cols,
    nvfp4_packed_cols,
    nvfp4_scale_cols,
    pad_fused_expert_shapes,
    split_gate_up_rows,
)
from vllm.models.minimax_m3.common.rocm_compat import (
    MINIMAX_DSPARK_ARCHITECTURES,
    MINIMAX_M31_ROCM_RECIPE,
    default_minimax_indexer_kv_dtype,
    is_minimax_draft_weight,
    is_minimax_m3_hf_config,
    minimax_m3_sparse_attention_is_full,
    minimax_m31_rocm_nvfp4_unsupported,
    nested_minimax_dspark_model,
    preserves_minimax_dspark_architecture,
)

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _m3_freq(num_layers: int = 6) -> list[int]:
    return [0] * min(3, num_layers) + [1] * max(0, num_layers - 3)


def _m31_freq(num_layers: int = 6) -> list[int]:
    return [1] * num_layers


def _text_config(
    *, num_layers: int, freq: list[int], model_type: str = "minimax_m3_text"
):
    return SimpleNamespace(
        model_type=model_type,
        num_hidden_layers=num_layers,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_index_dim": 128,
            "sparse_num_index_heads": 4,
            "sparse_topk_blocks": 16,
            "sparse_block_size": 128,
            "sparse_disable_index_value": freq,
            "sparse_score_type": "max",
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_attention_freq": freq,
        },
    )


@pytest.mark.cpu_test
def test_m31_config_routes_every_layer_through_msa() -> None:
    m3 = _text_config(num_layers=6, freq=_m3_freq(6))
    m31 = _text_config(num_layers=6, freq=_m31_freq(6))

    def _ids(config: SimpleNamespace) -> set[int]:
        freq = config.sparse_attention_config["sparse_attention_freq"]
        return {i for i, flag in enumerate(freq) if flag != 0}

    assert _ids(m3) == {3, 4, 5}
    assert _ids(m31) == {0, 1, 2, 3, 4, 5}
    assert not minimax_m3_sparse_attention_is_full(m3)
    assert minimax_m3_sparse_attention_is_full(m31)
    assert is_minimax_m3_hf_config(m31)
    assert is_minimax_m3_hf_config(SimpleNamespace(model_type="minimax_m3_vl"))


@pytest.mark.cpu_test
def test_nested_dspark_prefers_local_config(tmp_path: Path) -> None:
    dspark = tmp_path / "dspark"
    dspark.mkdir()
    (dspark / "config.json").write_text("{}", encoding="utf-8")
    m3 = _text_config(num_layers=6, freq=_m3_freq(6))
    assert nested_minimax_dspark_model(str(tmp_path), m3) == str(dspark)


@pytest.mark.cpu_test
def test_nested_dspark_hub_path_only_for_all_sparse() -> None:
    m3 = _text_config(num_layers=6, freq=_m3_freq(6))
    m31 = _text_config(num_layers=6, freq=_m31_freq(6))
    assert nested_minimax_dspark_model("MiniMaxAI/MiniMax-M3", m3) is None
    assert (
        nested_minimax_dspark_model("MiniMaxAI/MiniMax-M3.1-Flash", m31)
        == "MiniMaxAI/MiniMax-M3.1-Flash/dspark"
    )


@pytest.mark.cpu_test
def test_hf_config_keeps_minimax_dspark_architecture() -> None:
    assert preserves_minimax_dspark_architecture(["DSparkMiniMaxDraftModel"])
    assert preserves_minimax_dspark_architecture(["MiniMaxM3DSparkModel"])
    assert not preserves_minimax_dspark_architecture(
        ["MiniMaxM3SparseForConditionalGeneration"]
    )
    assert MINIMAX_DSPARK_ARCHITECTURES == (
        "DSparkMiniMaxDraftModel",
        "MiniMaxM3DSparkModel",
    )


@pytest.mark.cpu_test
def test_dspark_weight_mapper_strips_nested_prefix() -> None:
    mapped, shard = map_dspark_weight_name(
        "language_model.model.dspark.layers.0.decoder_layer.self_attn.q_proj.weight"
    )
    assert mapped == "layers.0.self_attn.qkv_proj.weight"
    assert shard == "q"
    mapped, shard = map_dspark_weight_name(
        "language_model.model.dspark.layers.2.decoder_layer.mlp.gate_proj.weight_scale_inv"
    )
    assert mapped == "layers.2.mlp.gate_up_proj.weight_scale"
    assert shard == 0
    mapped, shard = map_dspark_weight_name(
        "language_model.model.dspark.final_norm.weight"
    )
    assert mapped == "norm.weight"
    assert shard is None


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "name,skipped",
    [
        ("model.layers.3.block_sparse_moe.experts.0.w1.weight", False),
        ("model.mtp.layers.0.self_attn.qkv_proj.weight", True),
        ("language_model.model.dspark.fc.weight", True),
        ("model.layers.0.self_attn.qkv_proj.weight", False),
    ],
)
def test_target_skips_mtp_and_nested_dspark_weights(name: str, skipped: bool) -> None:
    assert is_minimax_draft_weight(name) is skipped


@pytest.mark.cpu_test
def test_rocm_nvfp4_error_points_at_mxfp4_fp8_recipe() -> None:
    message = minimax_m31_rocm_nvfp4_unsupported("KV cache")
    assert "not supported on ROCm" in message
    assert MINIMAX_M31_ROCM_RECIPE in message
    assert "NVFP4→MXFP4" in message
    recipe_script = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "online_serving"
        / "minimax_m31_flash_rocm.sh"
    )
    script = recipe_script.read_text(encoding="utf-8")
    assert "--kv-cache-dtype fp8" in script
    assert '"indexer_kv_dtype": "fp8"' in script
    assert "--quantization-config.moe mxfp4" in script
    assert "monokernel" in script
    assert "not native NVFP4" in script
    recipe = (
        Path(__file__).resolve().parents[2]
        / "recipes"
        / "models"
        / "MiniMaxAI"
        / "MiniMax-M3.1-Flash.yaml"
    )
    yaml_text = recipe.read_text(encoding="utf-8")
    amd_override = yaml_text.split("  amd:\n    extra_args:", 1)[1].split(
        "    extra_env:", 1
    )[0]
    assert "nvfp4" not in amd_override
    assert "fp8" in amd_override
    assert "mxfp4" in yaml_text.split("amd:\n        extra_args:", 1)[1][:200]


@pytest.mark.cpu_test
def test_default_indexer_dtype_fp8_on_gfx950(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.rocm_compat.current_platform.is_rocm",
        lambda: True,
    )
    fake_rocm = SimpleNamespace(on_gfx950=lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "vllm.platforms.rocm", fake_rocm)
    assert default_minimax_indexer_kv_dtype() == "fp8"


@pytest.mark.cpu_test
def test_default_indexer_dtype_bf16_off_gfx950(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm.models.minimax_m3.common.rocm_compat.current_platform.is_rocm",
        lambda: True,
    )
    fake_rocm = SimpleNamespace(on_gfx950=lambda: False)
    monkeypatch.setitem(__import__("sys").modules, "vllm.platforms.rocm", fake_rocm)
    assert default_minimax_indexer_kv_dtype() == "bf16"


@pytest.mark.cpu_test
def test_specialized_decode_refuses_all_sparse_m31() -> None:
    text = _text_config(num_layers=8, freq=_m31_freq(8))
    assert minimax_m3_sparse_attention_is_full(text)
    DecodeConfig = pytest.importorskip(
        "vllm.models.minimax_m3.amd.specialized_decode.runtime.spec"
    ).DecodeConfig
    from unittest.mock import MagicMock

    config = MagicMock()
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        use_ubatching=False,
        enable_expert_parallel=False,
        enable_eplb=False,
    )
    config.cache_config = SimpleNamespace(
        cache_dtype="fp8", block_size=128, kv_offloading_size=None
    )
    config.attention_config.resolve_indexer_kv_dtype.return_value = "fp8"
    config.lora_config = None
    config.kv_transfer_config = None
    config.use_v2_model_runner = True
    config.speculative_config = None
    config.model_config.enable_sleep_mode = False
    config.model_config.hf_text_config = text
    config.weight_transfer_config = None
    config.offload_config = SimpleNamespace(
        uva=SimpleNamespace(cpu_offload_gb=0),
        prefetch=SimpleNamespace(offload_group_size=0),
    )
    config.compilation_config = SimpleNamespace(mode=None)
    config.aux_output_config = None
    config.kernel_config = SimpleNamespace(moe_backend="aiter")
    reasons = DecodeConfig(last_layer=7).refusal_reasons(config)
    assert any("MiniMax-M3.1" in reason for reason in reasons)


@pytest.mark.cpu_test
def test_nvfp4_mxfp4_scale_and_pack_layouts() -> None:
    cols = 256
    assert nvfp4_packed_cols(cols) == cols // FP4_PACK_FACTOR
    assert nvfp4_scale_cols(cols) == cols // NVFP4_GROUP_SIZE
    assert mxfp4_packed_cols(cols) == cols // FP4_PACK_FACTOR
    assert mxfp4_scale_cols(cols) == cols // MXFP4_GROUP_SIZE
    assert nvfp4_scale_cols(cols) == 2 * mxfp4_scale_cols(cols)
    assert MINIMAX_ROUTED_EXPERT_CKPT == {"gate": "w1", "down": "w2", "up": "w3"}
    assert "input_scale" in MINIMAX_NVFP4_EXPERT_TENSORS


@pytest.mark.cpu_test
def test_nvfp4_experts_pad_then_requantize_to_mxfp4() -> None:
    num_experts = 2
    hidden, intermediate = 48, 64
    padded_h, padded_i = aiter_mxfp4_padded_dims(
        hidden, intermediate, alignment=ROCM_AITER_MXFP4_ALIGN
    )
    assert padded_h == ROCM_AITER_MXFP4_ALIGN
    assert padded_i == ROCM_AITER_MXFP4_ALIGN
    src_w13 = fused_w13_shape(num_experts, intermediate, hidden)
    src_w2 = down_w2_shape(num_experts, hidden, intermediate)
    dst_w13, dst_w2 = pad_fused_expert_shapes(num_experts, hidden, intermediate)
    assert src_w13 == (2, 128, 48)
    assert src_w2 == (2, 48, 64)
    assert dst_w13 == (2, 512, 256)
    assert dst_w2 == (2, 256, 256)

    w13 = [
        [[float(expert + row + col) for col in range(hidden)] for row in range(128)]
        for expert in range(num_experts)
    ]
    w2 = [
        [
            [float(expert + 10 + row + col) for col in range(intermediate)]
            for row in range(hidden)
        ]
        for expert in range(num_experts)
    ]
    padded_w13, padded_w2 = copy_padded_w13_w2(
        w13, w2, target_hidden=padded_h, target_intermediate=padded_i
    )
    gate_rows, up_rows = split_gate_up_rows(len(padded_w13[0]))
    assert list(gate_rows) == list(range(padded_i))
    assert list(up_rows) == list(range(padded_i, 2 * padded_i))
    assert padded_w13[0][0][:hidden] == w13[0][0]
    assert padded_w13[0][padded_i][:hidden] == w13[0][intermediate]
    assert padded_w13[0][0][hidden:] == [0.0] * (padded_h - hidden)
    assert padded_w2[1][0][:intermediate] == w2[1][0]
    assert all(value == 0.0 for value in padded_w2[1][0][intermediate:])

    packed, scales = _mxfp4_quantize_row(padded_w13[0][0])
    assert len(packed) == mxfp4_packed_cols(padded_h)
    assert len(scales) == mxfp4_scale_cols(padded_h)
    # Padding must stay exact zeros after MXFP4 requant of a zero block.
    zero_row = [0.0] * padded_h
    zero_packed, zero_scales = _mxfp4_quantize_row(zero_row)
    assert _mxfp4_dequantize_row(zero_packed, zero_scales) == zero_row
    # Conditioned values cover the e2m1 codebook so requant is a round trip.
    codebook_row = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * (padded_h // 8)
    packed, scales = _mxfp4_quantize_row(codebook_row)
    assert _mxfp4_dequantize_row(packed, scales) == codebook_row


@pytest.mark.cpu_test
def test_nvfp4_input_scales_are_consumed_not_copied() -> None:
    """Online MXFP4 uses dynamic activations; NVFP4 input_scale is dropped."""
    assert "input_scale" in MINIMAX_NVFP4_EXPERT_TENSORS
    assert "weight_scale_2" in MINIMAX_NVFP4_EXPERT_TENSORS


def _quantize_e2m1(value: float) -> int:
    sign = 8 if value < 0 else 0
    mag = abs(value)
    best = min(range(8), key=lambda code: abs(mag - _E2M1[code]))
    return sign | best


def _mxfp4_quantize_row(values: list[float]) -> tuple[list[int], list[int]]:
    packed: list[int] = []
    scales: list[int] = []
    for start in range(0, len(values), MXFP4_GROUP_SIZE):
        block = values[start : start + MXFP4_GROUP_SIZE]
        max_abs = max((abs(value) for value in block), default=0.0)
        if max_abs == 0.0:
            exp = 127
        else:
            import math

            exp = int(math.floor(math.log2(max_abs / 6.0))) + 127
            exp = min(max(exp, 0), 255)
        scale = 2.0 ** (exp - 127)
        scales.append(exp)
        codes = [_quantize_e2m1(value / scale) for value in block]
        for lo, hi in zip(codes[0::2], codes[1::2], strict=True):
            packed.append(lo | (hi << 4))
    return packed, scales


def _mxfp4_dequantize_row(packed: list[int], scales: list[int]) -> list[float]:
    out: list[float] = []
    for group, exp in enumerate(scales):
        scale = 2.0 ** (exp - 127)
        for byte in packed[
            group * (MXFP4_GROUP_SIZE // 2) : (group + 1) * (MXFP4_GROUP_SIZE // 2)
        ]:
            lo = byte & 0xF
            hi = (byte >> 4) & 0xF
            for nibble in (lo, hi):
                mag = _E2M1[nibble & 7]
                out.append((-mag if nibble & 8 else mag) * scale)
    return out
