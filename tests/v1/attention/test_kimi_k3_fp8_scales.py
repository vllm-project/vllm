# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from vllm.model_executor.layers.attention import kimi_k3_fp8_scales


class _FakeGroup:
    def __init__(self, rank: int, size: int):
        self.rank_in_group = rank
        self.ranks = list(range(size))


class _FakeLayer:
    def __init__(self):
        self._kimi_k3_fp8_calibration_amax = torch.arange(
            1,
            37,
            dtype=torch.float32,
        ).view(3, 12)
        self._kimi_k3_fp8_static_descale = torch.empty((0, 12))
        self._kimi_k3_fp8_calibration_state = {"armed": True}


def _config(
    layer: _FakeLayer,
    *,
    save_path: str | None = None,
    load_path: str | None = None,
):
    attention_config = SimpleNamespace(
        rocm_kimi_k3_fp8_prefill_scale_save_path=save_path,
        rocm_kimi_k3_fp8_prefill_scale_path=load_path,
        rocm_kimi_k3_fp8_prefill_scale_margin=1.1,
    )
    return SimpleNamespace(
        attention_config=attention_config,
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.3.self_attn.attn": layer}
        ),
        model_config=SimpleNamespace(
            model="moonshotai/Kimi-K3",
            revision=None,
        ),
    )


def test_save_kimi_k3_fp8_calibration(tmp_path: Path, monkeypatch) -> None:
    layer = _FakeLayer()
    config = _config(layer, save_path=str(tmp_path))
    monkeypatch.setattr(
        kimi_k3_fp8_scales,
        "get_tp_group",
        lambda: _FakeGroup(rank=1, size=2),
    )
    monkeypatch.setattr(
        kimi_k3_fp8_scales,
        "get_pp_group",
        lambda: _FakeGroup(rank=0, size=1),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    kimi_k3_fp8_scales.save_kimi_k3_fp8_calibration(config)

    path = tmp_path / "kimi-k3-fp8-scales-pp00-tp01.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tp_rank"] == 1
    assert payload["layers"]["3"]["q_amax"] == list(range(1, 13))
    assert payload["layers"]["3"]["v_amax"] == list(range(25, 37))


def test_load_kimi_k3_fp8_static_scales(tmp_path: Path, monkeypatch) -> None:
    layer = _FakeLayer()
    path = tmp_path / "scales.safetensors"
    tensors = {
        f"layers.3.{name}_descale": torch.arange(
            offset,
            offset + 24,
            dtype=torch.float32,
        )
        for name, offset in (("q", 1), ("k", 101), ("v", 201))
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "schema": "1",
            "model": "moonshotai/Kimi-K3",
            "revision": "",
            "tp_size": "2",
            "pp_size": "1",
            "num_layers": "1",
            "fp8_dtype": "float8_e4m3fnuz",
            "cache_mode": "bf16_latent_cache",
            "qk_head_dim": "192",
            "v_head_dim": "128",
        },
    )
    config = _config(layer, load_path=str(path))
    monkeypatch.setattr(
        kimi_k3_fp8_scales,
        "get_tp_group",
        lambda: _FakeGroup(rank=1, size=2),
    )
    monkeypatch.setattr(
        kimi_k3_fp8_scales,
        "get_pp_group",
        lambda: _FakeGroup(rank=0, size=1),
    )

    kimi_k3_fp8_scales.prepare_kimi_k3_fp8_scales(config)

    assert layer._kimi_k3_fp8_static_descale.shape == (3, 12)
    torch.testing.assert_close(
        layer._kimi_k3_fp8_static_descale[0],
        torch.arange(13, 25, dtype=torch.float32),
    )
    torch.testing.assert_close(
        layer._kimi_k3_fp8_static_descale[2],
        torch.arange(213, 225, dtype=torch.float32),
    )
