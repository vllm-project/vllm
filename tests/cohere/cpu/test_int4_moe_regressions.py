# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the int4 MoE / vision regressions fixed during the v0.21 rebase.

These are fast, GPU-free tests that pin the pure-logic helpers behind three
nightly bee_eval failures so the fixes cannot silently regress again when
upstream reshuffles files:

1. ``compressed_tensors_moe_wna16_marlin`` w2-scale TP sharding for GPTQ MoE
   (``Invalid thread config ... group_size=16, is_k_full=0``).
2. ``compressed_tensors_moe`` actorder normalization for mixed AWQ+GPTQ MoE
   (``All MoE projections need to have same quantization scheme``).
3. ``cohere2_vision`` skipping stray ``lm_head.*`` weights on tied-embedding
   checkpoints (``no module or parameter named 'lm_head'``).

CPU-safe: none of these paths touch CUDA. Lives under ``tests/cohere/cpu`` so it
is auto-discovered by ``run_cpu_tests`` in ``run_tests.sh`` (the ``cpu_check``
group run by daily CI). Run locally with:
``pytest tests/cohere/cpu/test_int4_moe_regressions.py``
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("compressed_tensors")

from compressed_tensors.quantization import (  # noqa: E402
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)


def _w4_group_args(actorder, group_size: int = 32) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
        symmetric=True,
        actorder=actorder,
    )


# ---------------------------------------------------------------------------
# Failure 1: WNA16 Marlin MoE w2-scale sharding (GPTQ int4, TP>1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "actorder,group_size,part,full,expected",
    [
        # actorder="group" with real grouping: must load full-K w2 scales and,
        # when sharded (part != full), report is_k_full=False.
        (ActivationOrdering.GROUP, 32, 64, 128, (True, 128, False)),
        # actorder="group" but unsharded (part == full): full scales, k_full.
        (ActivationOrdering.GROUP, 32, 128, 128, (True, 128, True)),
        # actorder="group" with channel-wise (group_size == -1): no full load.
        (ActivationOrdering.GROUP, -1, 64, 128, (False, 64, False)),
        # "static"/"weight" reorder at quant time -> shard normally + k_full.
        ("static", 32, 64, 128, (False, 64, True)),
        ("weight", 32, 64, 128, (False, 64, True)),
        (None, 32, 64, 128, (False, 64, True)),
    ],
)
def test_w2_scale_sharding(actorder, group_size, part, full, expected):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (  # noqa: E501
        CompressedTensorsWNA16MarlinMoEMethod,
    )

    result = CompressedTensorsWNA16MarlinMoEMethod._w2_scale_sharding(
        actorder, group_size, part, full
    )
    assert result == expected


def test_w2_scale_sharding_static_never_invalid_k_full():
    """Regression: static actorder under TP must keep is_k_full=True so the
    Marlin kernel does not get the invalid (group_size=16, is_k_full=0) config.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16_marlin import (  # noqa: E501
        CompressedTensorsWNA16MarlinMoEMethod,
    )

    load_full_w2, w2_scales_size, is_k_full = (
        CompressedTensorsWNA16MarlinMoEMethod._w2_scale_sharding("static", 32, 64, 128)
    )
    assert load_full_w2 is False
    assert w2_scales_size == 64  # sharded per-partition, not full-K
    assert is_k_full is True


# ---------------------------------------------------------------------------
# Failure 2: actorder normalization for mixed AWQ+GPTQ MoE schemes
# ---------------------------------------------------------------------------


def test_normalize_weight_actorder_maps_weight_and_static_to_none():
    """``weight`` and (its alias) ``static`` are layout-equivalent to ``None``
    and must normalize to the same value so mixed-scheme MoE ckpts compare equal.
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
        _normalize_weight_actorder,
    )

    for actorder in (ActivationOrdering.WEIGHT, "weight", "static"):
        sd = {"weights": _w4_group_args(actorder), "input_activations": None}
        out = _normalize_weight_actorder(sd)
        assert out["weights"].actorder is None


def test_normalize_weight_actorder_preserves_group_and_none():
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
        _normalize_weight_actorder,
    )

    group_sd = {"weights": _w4_group_args(ActivationOrdering.GROUP)}
    assert _normalize_weight_actorder(group_sd)["weights"].actorder == "group"

    none_sd = {"weights": _w4_group_args(None)}
    assert _normalize_weight_actorder(none_sd)["weights"].actorder is None

    assert _normalize_weight_actorder(None) is None


def test_normalize_weight_actorder_makes_mixed_schemes_compare_equal():
    """A GPTQ (``static``) and an AWQ (``weight``) projection must be treated as
    the same scheme after normalization (the equality check in get_moe_method).
    """
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
        _normalize_weight_actorder,
    )

    gptq = {"weights": _w4_group_args("static"), "input_activations": None}
    awq = {"weights": _w4_group_args("weight"), "input_activations": None}

    gptq_n = _normalize_weight_actorder(gptq)
    awq_n = _normalize_weight_actorder(awq)
    assert gptq_n == awq_n


# ---------------------------------------------------------------------------
# Failure 3: cohere2_vision skips stray lm_head weights (tied embeddings)
# ---------------------------------------------------------------------------


def test_cohere2_vision_load_weights_skips_lm_head(monkeypatch):
    import vllm.model_executor.models.cohere2_vision as mod

    captured: dict[str, object] = {}

    class _FakeLoader:
        def __init__(self, model, skip_prefixes=None):
            captured["skip_prefixes"] = skip_prefixes

        def load_weights(self, weights, mapper=None):
            captured["mapper"] = mapper
            return {name for name, _ in weights}

    monkeypatch.setattr(mod, "AutoWeightsLoader", _FakeLoader)

    dummy_self = SimpleNamespace(hf_to_vllm_mapper="sentinel-mapper")
    weights = [("lm_head.weight", object()), ("language_model.embed.weight", object())]

    loaded = mod.Cohere2VisionForConditionalGeneration.load_weights(dummy_self, weights)

    assert captured["skip_prefixes"] == ["lm_head"]
    assert captured["mapper"] == "sentinel-mapper"
    assert loaded == {"lm_head.weight", "language_model.embed.weight"}
