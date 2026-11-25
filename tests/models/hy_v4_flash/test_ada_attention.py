# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.hy_v4_flash.nvidia.ada import apply_attention_sink
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(89),
    reason="HY4 Flash Ada sparse attention requires SM89",
)


# HY4 Flash has 32 sparse and 64 window heads, or 8 and 16 at TP=4.
@pytest.mark.parametrize(
    ("heads", "topk"), [(8, 2048), (16, 512), (32, 2048), (64, 512)]
)
@pytest.mark.parametrize("splits", [1, 4])
def test_ada_attention_sink(heads, topk, splits):
    torch.manual_seed(17)
    device = "cuda"
    rows, block, latent, dim = 5, 64, 512, 576
    scale = 256**-0.5
    q = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
    blocks = (topk + block - 1) // block + 1
    # Real cache blocks have gaps between them when layers are interleaved.
    physical = torch.randn(blocks, block + 3, dim, device=device, dtype=q.dtype)
    slots = torch.arange(blocks * block, device=device)
    slots = slots + (slots // block) * 3
    logical = torch.randperm(slots.numel(), device=device)[:topk]
    indices = slots[logical].to(torch.int32).repeat(rows, 1)
    indices[0, :] = -1
    indices[1, 1:] = -1
    indices[2, 0::7] = -1
    indices[3, topk // 2 :] = -1
    sinks = torch.linspace(-40, 40, heads, device=device)

    def run():
        out, lse = triton_mla_sparse_fwd(
            q,
            physical.flatten(0, 1),
            indices,
            scale,
            return_lse=True,
            num_kv_splits=splits,
        )
        assert lse is not None
        apply_attention_sink(out, lse, sinks)
        return out, torch.logaddexp(lse, sinks)

    def reference():
        # Append the zero-value sink to an independent CPU softmax.
        keys = physical.flatten(0, 1).float().cpu()
        keys = keys[indices.long().cpu().clamp_min(0)]
        logits = q.float().cpu().bmm(keys.transpose(1, 2)) * scale
        logits.masked_fill_(indices[:, None].cpu() < 0, -torch.inf)
        logits = torch.cat(
            [logits, sinks.cpu()[None, :, None].expand(rows, heads, 1)], dim=-1
        )
        output = torch.softmax(logits, dim=-1)[..., :-1].bmm(keys[..., :latent])
        return output, logits.logsumexp(dim=-1)

    def check(out, lse):
        expected, expected_lse = reference()
        assert torch.isfinite(out).all() and not out[0].any()
        torch.testing.assert_close(out.float().cpu(), expected, atol=0.013, rtol=0.018)
        torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-5, rtol=2e-5)

    out, lse = run()
    check(out, lse)
    if splits == 1:
        no_sink, _ = triton_mla_sparse_fwd(
            q, physical.flatten(0, 1), indices, scale, num_kv_splits=splits
        )
        assert (out[-1, -1] - no_sink[-1, -1]).abs().max().item() > 0.03
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = run()
    indices[1, 1:16] = indices[-1, 1:16]
    sinks.add_(2)
    graph.replay()
    check(out, lse)
