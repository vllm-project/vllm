# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint KV-cache scales reach MLA attention layers, and FlashInfer MLA
decode follows them across a weight reload under FULL CUDA graphs."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api

# compressed-tensors FP8 DeepSeek-V3 (MLA), one decoder layer.
MODEL = "inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC"
KV_SCALE = 0.5
PROMPTS = ["The capital of France is", "1 2 3 4 5 6"]

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.supports_fp8(),
    reason="Requires an FP8 KV cache on CUDA",
)


@pytest.fixture(scope="module")
def kv_scale_checkpoint(tmp_path_factory) -> str:
    """`MODEL` with a `kv_cache_scheme` and per-layer `self_attn.{k,v}_scale`
    tensors, the names llm-compressor writes for an attention module."""
    src = Path(hf_api().snapshot_download(MODEL))
    dst = tmp_path_factory.mktemp("mla-fp8-kv")
    for path in src.iterdir():
        if path.name != "config.json":
            (dst / path.name).symlink_to(path)
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"]["kv_cache_scheme"] = {
        "num_bits": 8,
        "type": "float",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
    }
    (dst / "config.json").write_text(json.dumps(config))
    assert not (dst / "model.safetensors.index.json").exists()
    scales = {
        f"model.layers.{i}.self_attn.{name}": torch.tensor(KV_SCALE)
        for i in range(config["num_hidden_layers"])
        for name in ("k_scale", "v_scale")
    }
    save_file(scales, dst / "kv-scales.safetensors")
    return str(dst)


def _mla_kv_scales(worker) -> list[tuple[float, float, float]]:
    from vllm.model_executor.layers.attention import MLAAttention

    return [
        (m._k_scale_float, m._v_scale_float, m._k_scale.item())
        for m in worker.model_runner.get_model().modules()
        if isinstance(m, MLAAttention)
    ]


def test_mla_loads_checkpoint_kv_scales(vllm_runner, monkeypatch, kv_scale_checkpoint):
    """compressed-tensors used to give only `Attention` a KV-cache method, and
    the cache-scale mapper sent MLA scale names to `self_attn.attn`, so every
    MLA layer ran its FP8 KV cache at scale 1.0."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        kv_scale_checkpoint,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=1,
    ) as llm:
        scales = llm.collective_rpc(_mla_kv_scales)[0]
    assert scales and all(s == (KV_SCALE,) * 3 for s in scales), scales


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FlashInfer MLA runs on SM100",
)
@pytest.mark.parametrize("reload_mode", ["layerwise", "direct"])
def test_flashinfer_mla_decode_follows_kv_scale_reload(
    vllm_runner, monkeypatch, kv_scale_checkpoint, reload_mode
):
    """A dummy start captures the decode CUDA graphs with dummy KV scales; after
    reloading the checkpoint, greedy decode must match a cold start on it."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_RELOAD_MODE", reload_mode)
    kwargs = dict(
        kv_cache_dtype="fp8",
        attention_config={"backend": "FLASHINFER_MLA"},
        enable_prefix_caching=False,
        max_model_len=64,
        max_num_seqs=len(PROMPTS),
    )
    with vllm_runner(kv_scale_checkpoint, **kwargs) as llm:
        cold = llm.generate_greedy(PROMPTS, max_tokens=16)

    with vllm_runner(kv_scale_checkpoint, load_format="dummy", **kwargs) as llm:
        llm.collective_rpc(
            "update_config",
            kwargs={"overrides": {"load_config": {"load_format": "auto"}}},
        )
        llm.collective_rpc(
            "reload_weights", kwargs={"weights_path": kv_scale_checkpoint}
        )
        assert all(s == (KV_SCALE,) * 3 for s in llm.collective_rpc(_mla_kv_scales)[0])
        reloaded = llm.generate_greedy(PROMPTS, max_tokens=16)

    assert reloaded == cold
