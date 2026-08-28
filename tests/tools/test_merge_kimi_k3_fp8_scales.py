# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import runpy
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def test_merge_kimi_k3_fp8_scales(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "shards"
    input_dir.mkdir()
    for tp_rank in range(2):
        values = [float(tp_rank * 12 + index + 1) for index in range(12)]
        shard = {
            "schema": 1,
            "model": "moonshotai/Kimi-K3",
            "revision": "revision",
            "checkpoint_id": "abcdef123456",
            "calibration_id": "calibration-test",
            "tp_size": 2,
            "tp_rank": tp_rank,
            "pp_size": 1,
            "pp_rank": 0,
            "fp8_dtype": "float8_e4m3fnuz",
            "cache_mode": "bf16_latent_cache",
            "local_heads": 12,
            "qk_head_dim": 192,
            "v_head_dim": 128,
            "margin": 1.1,
            "layers": {
                "3": {
                    "runtime_name": "model.layers.3.self_attn.attn",
                    "q_amax": values,
                    "k_amax": values,
                    "v_amax": values,
                }
            },
        }
        path = input_dir / f"kimi-k3-fp8-scales-pp00-tp{tp_rank:02d}.json"
        path.write_text(json.dumps(shard), encoding="utf-8")

    output = tmp_path / "scales.safetensors"
    script = Path(__file__).parents[2] / "tools" / "merge_kimi_k3_fp8_scales.py"
    monkeypatch.setattr(sys, "argv", [str(script), str(input_dir), str(output)])
    runpy.run_path(str(script), run_name="__main__")

    with safe_open(output, framework="pt", device="cpu") as artifact:
        scale = artifact.get_tensor("layers.3.q_descale")
        assert scale.shape == (24,)
        expected = torch.arange(1, 25, dtype=torch.float32)
        expected *= 1.1 / torch.finfo(torch.float8_e4m3fnuz).max
        torch.testing.assert_close(scale, expected)
        assert artifact.metadata()["num_layers"] == "1"
