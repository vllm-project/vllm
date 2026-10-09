# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("MiniMax-M3 fused decode is ROCm-only", allow_module_level=True)

from vllm.model_executor.models.config import MiniMaxM3SparseConfig
from vllm.models.minimax_m3.amd import fused_decode
from vllm.models.minimax_m3.amd.fused_decode import fused_decode_unsupported_reason


def _config(**overrides):
    values = dict(
        tp=4,
        pp=1,
        dp=1,
        ep=False,
        cache_dtype="fp8",
        block_size=128,
        indexer_kv_dtype="fp8",
        max_model_len=133120,
        use_index_cache=False,
        fused=True,
        quantization_config=None,
    )
    values.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=values["tp"],
            pipeline_parallel_size=values["pp"],
            data_parallel_size=values["dp"],
            enable_expert_parallel=values["ep"],
        ),
        cache_config=SimpleNamespace(
            cache_dtype=values["cache_dtype"], block_size=values["block_size"]
        ),
        attention_config=SimpleNamespace(
            minimax_m3_fused_decode=values["fused"],
            resolve_indexer_kv_dtype=lambda default: values["indexer_kv_dtype"],
        ),
        model_config=SimpleNamespace(
            max_model_len=values["max_model_len"],
            quantization_config=values["quantization_config"],
            hf_text_config=SimpleNamespace(
                use_index_cache=values["use_index_cache"],
                sparse_attention_config={"sparse_attention_freq": [0, 0, 1, 1]},
            ),
        ),
    )


@pytest.fixture(autouse=True)
def _gfx950(monkeypatch):
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: True)


def test_supported_deployment():
    assert fused_decode_unsupported_reason(_config()) is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"tp": 8}, "tensor parallel size 4"),
        ({"pp": 2}, "pipeline or data parallelism"),
        ({"dp": 2}, "pipeline or data parallelism"),
        ({"ep": True}, "expert parallelism"),
        ({"cache_dtype": "auto"}, "--kv-cache-dtype fp8"),
        ({"block_size": 64}, "--block-size 128"),
        ({"indexer_kv_dtype": "bf16"}, "indexer_kv_dtype"),
        ({"max_model_len": (1 << 20) + 1}, "--max-model-len"),
        ({"use_index_cache": True}, "use_index_cache"),
    ],
)
def test_unsupported_deployment(overrides, reason):
    assert reason in fused_decode_unsupported_reason(_config(**overrides))


def test_requires_gfx950(monkeypatch):
    import vllm.platforms.rocm as rocm

    monkeypatch.setattr(rocm, "on_gfx950", lambda: False)
    assert "gfx950" in fused_decode_unsupported_reason(_config())


def test_tensor_parallel_size_matches_kernel():
    from aiter.ops.flydsl.kernels.minimax_m3_mono.config import TP

    assert fused_decode._TP == TP


def test_quantizes_only_sparse_attention_projections():
    config = _config()
    MiniMaxM3SparseConfig.verify_and_update_config(config)
    targets = config.model_config.quantization_config.targets
    assert set(targets.values()) == {"fp8_per_channel"}
    assert sorted(targets) == [
        "re:.*layers\\.2\\.self_attn\\.(qkv|o)_proj$",
        "re:.*layers\\.3\\.self_attn\\.(qkv|o)_proj$",
    ]


@pytest.mark.parametrize(
    "overrides", [{"fused": False}, {"quantization_config": object()}]
)
def test_leaves_quantization_alone(overrides):
    config = _config(**overrides)
    before = config.model_config.quantization_config
    MiniMaxM3SparseConfig.verify_and_update_config(config)
    assert config.model_config.quantization_config is before
