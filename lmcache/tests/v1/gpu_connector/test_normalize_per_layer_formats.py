# SPDX-License-Identifier: Apache-2.0
"""Tests for ``normalize_and_discover_per_layer_formats``.

Uniform models return the whole normalized structure (so a format that isn't a
per-layer list -- a cross-layer fused tensor, or a ``[keys, values]`` K/V-split
-- stays intact); mixed models report each engine group's own per-layer format.
The K/V-split path is the SGLang regression: it must not be sliced per layer.
"""

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import normalize_and_discover_per_layer_formats
import lmcache.c_ops as lmc_ops

NB, NL, BS, NH, HS = 7, 5, 3, 2, 4
DT = torch.float16
F = lmc_ops.EngineKVFormat

# The vLLM CPU-HND safeguard forces HND regardless of hint on a CPU host; bypass
# it so the hint-driven NHD branch is exercised on any host.
_VLLM_DEV = "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type"


def _t(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=DT)


def test_sglang_kv_list_uniform():
    # Regression: a flat list of 2*NL tensors that detection regroups into a
    # length-2 [key_layers, value_layers]. ``is_layer_list`` is False for the
    # K/V-split format, so it is returned whole -- never sliced per layer.
    flat = [_t(NB * BS, NH, HS) for _ in range(2 * NL)]
    normalized, formats = normalize_and_discover_per_layer_formats(
        flat, (), EngineType.SGLANG, {"tokens_per_block": BS}
    )
    assert len(normalized) == 2 and len(normalized[0]) == NL
    assert tuple(normalized[0][0].shape) == (NB, BS, NH, HS)
    assert formats == [F.TWO_X_NL_X_NB_BS_NH_HS] * NL


def test_vllm_layer_list_uniform(monkeypatch):
    # Per-layer-list format, every layer the same: whole list back, format per layer.
    monkeypatch.setattr(_VLLM_DEV, "cuda")
    kv = [_t(2, NB, BS, NH, HS) for _ in range(NL)]
    normalized, formats = normalize_and_discover_per_layer_formats(
        kv, (), EngineType.VLLM, {"kv_layout": "NHD"}
    )
    assert len(normalized) == NL
    assert formats == [F.NL_X_TWO_NB_BS_NH_HS] * NL


def test_vllm_heterogeneous_groups(monkeypatch):
    # Mixed-format model: a full K/V group beside a key-only MLA index group.
    monkeypatch.setattr(_VLLM_DEV, "cuda")
    kv = [_t(2, NB, BS, NH, HS) for _ in range(3)] + [_t(NB, BS, HS) for _ in range(2)]
    normalized, formats = normalize_and_discover_per_layer_formats(
        kv, [[0, 1, 2], [3, 4]], EngineType.VLLM, {"kv_layout": "NHD"}
    )
    assert len(normalized) == 5
    assert formats[:3] == [F.NL_X_TWO_NB_BS_NH_HS] * 3
    assert formats[3:] == [F.NL_X_NB_BS_HS] * 2


def test_vllm_mixed_rank4_fused_groups(monkeypatch):
    # Regression for #4263: blocks-first fused rank-4 groups whose trailing
    # dims differ (hybrid sliding + full attention under the vLLM 0.26 V2
    # model runner). v0.5.2's whole-set detection applied the first tensor's
    # fused_dim to every layer and died in reshape; mixed-shape sets must be
    # detected per shape bucket instead, each layer keeping its own raw
    # [NB, BS, NH, CS] tensor and reporting the fused CS format.
    monkeypatch.setattr(_VLLM_DEV, "cuda")
    kv = [_t(NB, BS, NH, 2 * HS) for _ in range(3)] + [
        _t(NB, BS, NH, 4 * HS) for _ in range(2)
    ]
    normalized, formats = normalize_and_discover_per_layer_formats(
        kv, [[0, 1, 2], [3, 4]], EngineType.VLLM, {"kv_layout": "NHD"}
    )
    assert len(normalized) == 5
    assert tuple(normalized[0].shape) == (NB, BS, NH, 2 * HS)
    assert tuple(normalized[3].shape) == (NB, BS, NH, 4 * HS)
    assert formats == [F.NL_X_NB_BS_NH_CS] * 5
