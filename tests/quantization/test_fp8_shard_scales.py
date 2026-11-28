# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused per-tensor FP8 shards keep their checkpoint weights on CUTLASS.

Run `pytest tests/quantization/test_fp8_shard_scales.py`.
"""

import glob

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from vllm.platforms import current_platform

# Layer 0 q/k/v weight scales differ.
MODEL = "RedHatAI/Llama-3.2-1B-Instruct-FP8"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_qkv_keeps_checkpoint_weights(vllm_runner, monkeypatch):
    # `LLM.apply_model` pickles a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    prefix = "model.layers.0.self_attn"
    tensors = {}
    for f in glob.glob(f"{snapshot_download(MODEL)}/*.safetensors"):
        with safe_open(f, "pt") as st:
            names = st.keys()
            for k in names:
                if k.startswith(prefix):
                    tensors[k] = st.get_tensor(k)
    projs = ("q_proj", "k_proj", "v_proj")
    weights = [tensors[f"{prefix}.{p}.weight"] for p in projs]
    scales = [tensors[f"{prefix}.{p}.weight_scale"].float() for p in projs]
    assert len({s.item() for s in scales}) > 1
    expected_weight = torch.cat(weights).view(torch.uint8)
    expected_scale = torch.cat(
        [s.reshape(1).expand(w.shape[0]) for w, s in zip(weights, scales)]
    )

    def check_qkv(model):
        qkv = model.model.layers[0].self_attn.qkv_proj
        return (
            type(qkv.scheme.fp8_linear).__name__,
            torch.equal(qkv.weight.t().cpu().view(torch.uint8), expected_weight),
            torch.equal(qkv.weight_scale.cpu().flatten(), expected_scale),
        )

    with vllm_runner(MODEL, enforce_eager=True, max_model_len=512) as llm:
        kernel, same_weight, same_scale = llm.apply_model(check_qkv)[0]

    if kernel != "CutlassFP8ScaledMMLinearKernel":
        pytest.skip(f"{kernel} takes one per-tensor weight scale")
    assert same_weight
    assert same_scale
