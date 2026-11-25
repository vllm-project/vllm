# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 NVFP4 experts as one DeepGEMM MegaMoE launch (SM90) vs. Humming,
on the Inkling-2-Mini checkpoint."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_device_capability(90):
    pytest.skip("NVFP4 MegaMoE runs on SM90", allow_module_level=True)

import vllm.models.inkling2.nvidia.moe as moe_module  # noqa: E402
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (  # noqa: E402
    deep_gemm_supports_sm90_nvfp4,
)
from vllm.models.inkling2.nvidia.moe import Inkling2MoE  # noqa: E402
from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts  # noqa: E402

MODEL = "thinkingmachines/Inkling-2-Mini"
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


@pytest.fixture(scope="module")
def checkpoint():
    from huggingface_hub.errors import LocalEntryNotFoundError
    from safetensors import safe_open

    from vllm.transformers_utils.repo_utils import hf_api

    if not deep_gemm_supports_sm90_nvfp4():
        pytest.skip("DeepGEMM has no SM90 NVFP4 MegaMoE")
    try:
        path = hf_api().snapshot_download(
            MODEL, local_files_only=True, allow_patterns=["*.json", "*.safetensors"]
        )
    except LocalEntryNotFoundError:
        pytest.skip(f"{MODEL} is not in the local HF cache")
    with open(os.path.join(path, "config.json")) as f:
        config = SimpleNamespace(**json.load(f)["text_config"])
    with safe_open(
        os.path.join(path, "model-00001-of-00001.safetensors"), "pt", device="cuda"
    ) as tensors:
        yield config, tensors


def _layer(checkpoint, layer: int, backend: str) -> Inkling2MoE:
    config, tensors = checkpoint
    prefix = f"model.llm.layers.{layer}.mlp"
    vllm_config = VllmConfig(kernel_config=KernelConfig(moe_backend=backend))
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        moe = Inkling2MoE(config, "nvfp4", prefix=prefix)
        for name, param in moe.named_parameters():
            param.weight_loader(param, tensors.get_tensor(f"{prefix}.{name}"))
        moe.experts.process_weights_after_loading()
    return moe


def _input(checkpoint, layer: int, tokens: int) -> torch.Tensor:
    """Gaussian tokens times the layer's MLP norm weight."""
    config, tensors = checkpoint
    norm = tensors.get_tensor(f"model.llm.layers.{layer}.mlp_norm.weight").float()
    return (torch.randn(tokens, config.hidden_size, device="cuda") * norm).bfloat16()


def _dequant(checkpoint, layer: int, name: str, e: int) -> torch.Tensor:
    _, tensors = checkpoint
    prefix = f"model.llm.layers.{layer}.mlp.experts.{name}_weight"
    w = tensors.get_tensor(prefix)[e]
    lut = torch.tensor(E2M1, device="cuda")
    v = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], -1).flatten(-2)
    scale = tensors.get_tensor(f"{prefix}_scale")[e].float().repeat_interleave(16, -1)
    amax = tensors.get_tensor(f"{prefix}_amax")[e]
    return v * scale * amax / tensors.get_tensor(f"{prefix}_outer_scale_denominator")


def _reference(checkpoint, layer: int, moe: Inkling2MoE, x: torch.Tensor):
    """FP32 dequantized experts, with gate/up and the SiTU rounded to bf16 as
    both kernels round them."""
    config, _ = checkpoint
    gate = moe.gate
    ids, weights, _ = select_experts(
        x, gate.weight, gate.bias, gate.topk, gate.route_scale, moe.all_experts
    )
    weights = weights.view(ids.shape)
    out = torch.zeros(x.shape, device="cuda")
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        w13 = _dequant(checkpoint, layer, "w13", e)
        gate_up = (x[tok].float() @ w13.T).bfloat16().float()
        g, u = gate_up[:, 0::2], gate_up[:, 1::2]
        g_cap, u_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        act = g_cap * torch.tanh(g / g_cap) * torch.sigmoid(g)
        act = act * u_cap * torch.tanh(u / u_cap) * weights[tok, slot][:, None]
        w2 = _dequant(checkpoint, layer, "w2", e)
        out.index_add_(0, tok, act.bfloat16().float() @ w2.T)
    return out


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("layer", [0, 9, 19])
def test_mega_moe_matches_humming(dist_init, checkpoint, monkeypatch, layer: int):
    """Within 0.1 % of Humming, or no more than 1 % further from the FP32
    reference than Humming is, across the decode and prefill BLOCK_M. The
    launch runs Inkling's softcapped SiTU, not the op's SwiGLU default."""
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    op = deep_gemm.bf16_nvfp4_mega_moe
    launches: list[dict] = []

    def recorded_op(*args, **kwargs):
        launches.append(kwargs)
        return op(*args, **kwargs)

    monkeypatch.setattr(deep_gemm, "bf16_nvfp4_mega_moe", recorded_op)
    config, _ = checkpoint
    humming = _layer(checkpoint, layer, "auto")
    mega = _layer(checkpoint, layer, "deep_gemm_mega_moe")
    assert humming.experts.mega is None and mega.experts.humming is None
    assert mega.experts.w13_weight.numel() == 0
    torch.manual_seed(layer)
    for tokens in (1, 8, 128, 1024):
        x = _input(checkpoint, layer, tokens)
        ours, control = mega(x), humming(x)
        ref = _reference(checkpoint, layer, humming, x)
        vs_humming, vs_ref = _rel_l2(ours, control), _rel_l2(ours, ref)
        humming_vs_ref = _rel_l2(control, ref)
        assert vs_humming <= 1e-3 or vs_ref <= 1.01 * humming_vs_ref, (
            f"{tokens} tokens: {vs_humming=:.2e} {vs_ref=:.2e} {humming_vs_ref=:.2e}"
        )
    assert len(launches) == 4
    for kwargs in launches:
        assert kwargs["activation"] == "situ"
        assert kwargs["activation_alpha"] == config.mlp_gate_softcap == 4.0
        assert kwargs["activation_beta"] == config.mlp_up_softcap == 25.0


def test_mega_moe_graph_replays_padded_batch(dist_init, checkpoint, monkeypatch):
    """A captured launch replays bitwise like eager on new inputs, and CUDA
    graph padding tokens (here NaN) come out as 0."""
    tokens, padded = 5, 8
    mega = _layer(checkpoint, 0, "deep_gemm_mega_moe")
    is_padding = torch.arange(padded, device="cuda") >= tokens
    context = SimpleNamespace(is_padding=is_padding)
    monkeypatch.setattr(moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe_module, "get_forward_context", lambda: context)

    x = _input(checkpoint, 0, padded)
    x[tokens:] = float("nan")
    mega(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mega(x)
    for _ in range(3):
        x.copy_(_input(checkpoint, 0, padded))
        x[tokens:] = float("nan")
        expected = mega(x)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        assert (captured[tokens:] == 0).all()
