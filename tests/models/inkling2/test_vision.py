# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 vision encoder against an eager per-image torch reference."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.models.inkling2.ops.vision import gather_patches
from vllm.models.inkling2.vision import (
    _CLIP_MEAN,
    _CLIP_STD,
    _PROCESSOR_CLIP_STD,
    Inkling2VisionEncoder,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs a Blackwell GPU",
)

CONFIG = {
    "temporal_patch_size": 2,
    "native_patch_size": 20,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 256,
    "projector_bias": False,
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "mlp_ratio": 4.0,
    "posemb_grid_size": 32,
    "layer_norm_eps": 1e-5,
}


def _rope(x, positions):
    dim = x.shape[-1]
    inv_freq = 10000.0 ** (-torch.arange(0, dim, 4, device=x.device) / dim)
    angles = torch.cat((positions[:, 1:] * inv_freq, positions[:, :1] * inv_freq), -1)[
        :, None
    ]
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    rotated = (
        even * angles.cos() - odd * angles.sin(),
        even * angles.sin() + odd * angles.cos(),
    )
    return torch.stack(rotated, -1).flatten(-2).bfloat16()


def _normalize(raw: torch.Tensor) -> torch.Tensor:
    """The former CPU CLIP normalization, then the encoder's PE remap."""
    mean = np.array(_CLIP_MEAN, np.float32)
    clip = (raw.float().cpu().numpy() / np.float32(255) - mean) / np.array(
        _PROCESSOR_CLIP_STD, np.float32
    )
    x = torch.from_numpy(clip).bfloat16().float().to(raw.device)
    x = x * torch.tensor(_CLIP_STD, device=raw.device)
    return (2.0 * (x + torch.tensor(_CLIP_MEAN, device=raw.device)) - 1.0).bfloat16()


def _reference_image(enc, raw, h, w):
    m, p, width = enc.merge_size, enc.patch_size, CONFIG["hidden_size"]
    height, wide = h * m, w * m
    x = _normalize(raw)  # [h * w, 40, 40, 3]
    pixels = x.reshape(h, w, m, p, m, p, 3).permute(0, 2, 1, 4, 3, 5, 6)
    emb = enc.backbone.embeddings
    hidden = emb.patch_embedding(pixels.reshape(height * wide, -1))
    grid = emb.position_embedding.view(1, 32, 32, width).permute(0, 3, 1, 2)
    pos = F.interpolate(grid.float(), (height, wide), mode="bilinear")
    hidden = hidden + pos[0].permute(1, 2, 0).reshape(-1, width).bfloat16()
    cls = (emb.class_embedding + emb.cls_position_embedding)[None]
    hidden = enc.backbone.ln_pre(torch.cat((cls, hidden)))
    y, x_ = torch.meshgrid(
        torch.arange(height, device="cuda"),
        torch.arange(wide, device="cuda"),
        indexing="ij",
    )
    positions = torch.cat(
        (
            torch.zeros(1, 2, device="cuda"),
            torch.stack((y.flatten(), x_.flatten()), -1) + 1,
        )
    )
    for layer in enc.backbone.layers:
        attn = layer.self_attn
        q, k, v = (
            attn.qkv_proj(layer.ln_1(hidden))
            .view(-1, 3, attn.num_heads, attn.head_dim)
            .unbind(1)
        )
        q, k = _rope(q, positions), _rope(k, positions)
        out = F.scaled_dot_product_attention(
            *(t.transpose(0, 1) for t in (q, k, v))
        ).transpose(0, 1)
        hidden = hidden + attn.out_proj(out.reshape(hidden.shape)) * layer.ls_1.gamma
        mlp = layer.mlp(layer.ln_2(hidden))
        hidden = hidden + mlp * layer.ls_2.gamma
    pooled = hidden[1:].reshape(h, m, w, m, width).permute(0, 2, 1, 3, 4)
    pooled = pooled.reshape(h * w, -1)
    return enc.proj_out(F.gelu(enc.proj_in(pooled)))


@torch.inference_mode()
def test_packed_images_match_per_image_reference():
    """Packed multi-image metadata (positions, pixel gather, CLS rows, 2x2
    pooling) must match encoding each image on its own."""
    torch.manual_seed(0)
    enc = Inkling2VisionEncoder(CONFIG, text_hidden_size=192).cuda()
    for param in enc.parameters():
        param.data.normal_(0, 0.05)
    for name, param in enc.named_parameters():
        if name.endswith(("ln_pre.weight", "ln_1.weight", "ln_2.weight")):
            param.data.fill_(1.0)
    grids = [(3, 5), (7, 2), (1, 1)]
    patches = torch.randint(-1, 256, (sum(h * w for h, w in grids), 40, 40, 3))
    patches = patches.cuda().bfloat16()
    grid_thw = torch.tensor([(1, h, w) for h, w in grids])

    out = enc(patches, grid_thw)

    ref = torch.cat(
        [
            _reference_image(enc, image, h, w)
            for image, (h, w) in zip(patches.split([h * w for h, w in grids]), grids)
        ]
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_gather_patches_matches_cpu_normalization():
    """Raw pixels (and -1 padding) normalize bitwise like the CPU processor's
    float32 CLIP step followed by the encoder's bf16 PE remap."""
    raw = torch.arange(-1, 256).repeat(75)[: 4 * 40 * 40 * 3]
    raw = raw.reshape(4, 40, 40, 3).cuda().bfloat16()
    offsets = torch.tensor([0, 60, 20 * 120, 3 * 4800 + 20 * 120 + 60], device="cuda")
    out = gather_patches(
        raw, offsets.int(), 20, _CLIP_MEAN, _PROCESSOR_CLIP_STD, _CLIP_STD
    )
    ref = _normalize(raw)
    expected = [ref[0, :20, :20], ref[0, :20, 20:], ref[0, 20:, :20], ref[3, 20:, 20:]]
    assert torch.equal(out, torch.stack(expected).reshape(4, -1))


@torch.inference_mode()
def test_cuda_graph_replay_matches_eager():
    """A graph captured on padded budget buffers (as EncoderCudaGraphManager
    pads them) replays smaller batches bitwise like the eager encoder."""
    from vllm.models.inkling2.model import Inkling2ForConditionalGeneration as M
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    torch.manual_seed(0)
    enc = Inkling2VisionEncoder(CONFIG, text_hidden_size=192).cuda()
    for param in enc.parameters():
        param.data.normal_(0, 0.05)
    model = SimpleNamespace(
        visual=enc,
        text_config=SimpleNamespace(hidden_size=192),
        _graph_metadata_keys=M._graph_metadata_keys,
    )
    model._graph_values = MethodType(M._graph_values, model)
    config = M.get_encoder_cudagraph_config(model)
    buffers = M.prepare_encoder_cudagraph_capture_inputs(
        model, 64, 4, 0, torch.device("cuda"), torch.bfloat16
    ).values
    graph, out = torch.cuda.CUDAGraph(), None
    M.encoder_cudagraph_forward(model, dict(buffers))
    with torch.cuda.graph(graph):
        out = M.encoder_cudagraph_forward(model, dict(buffers))

    grids = [(3, 5), (2, 7)]
    pixels = torch.randint(-1, 256, (sum(h * w for h, w in grids), 40, 40, 3))
    mm_kwargs = {
        "vision_pixels": pixels.cuda().bfloat16(),
        "vision_grid_thw": torch.tensor([(1, h, w) for h, w in grids]),
    }
    replay = M.prepare_encoder_cudagraph_replay_buffers(model, mm_kwargs, 4, 0)
    for key, buf in buffers.items():
        pad = config.padding_logics.get(
            key, EncoderCudaGraphManager._copy_padded_buffer
        )
        pad(buf, replay.values[key])
    graph.replay()

    eager = enc(mm_kwargs["vision_pixels"], mm_kwargs["vision_grid_thw"])
    assert torch.equal(out[: eager.shape[0]], eager)
