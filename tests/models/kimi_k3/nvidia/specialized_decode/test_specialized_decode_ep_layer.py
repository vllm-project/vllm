# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DEP MoE layer schedule (specialized_decode/cutedsl/moe/ep/layer.py EpMoeLayer) vs
the stock op sequence of KimiMoE.forward on one GPU: router GEMM (FP32 logits) +
sigmoid top-16, latent down-projection, routed experts (a deterministic stand-in of
the EP kernel), shared expert (gate_up, SiTU, down), latent RMSNorm + up-projection
accumulated into the shared output. The schedule's split-K GEMMs reduce in another
order than cuBLAS; everything else is the stock op. Also captured in a CUDA graph.

SiTU-and-mul runs both as the stock CUDA kernel (which indexes rows at a 2I pitch, so
it needs contiguous gate_up rows: the packed front's rows are 2I + 3584 apart) and
as the native op; every multi-row batch compares the shared expert row by row."""

import pytest
import torch

from vllm.platforms import current_platform

H, LAT, SHARED, E = 7168, 3584, 6144, 896


SEEN = []


def stand_in(latent, topk_w, topk_ids):
    """Routed-expert stand-in: depends on the latent only (top-k ties may differ);
    the routing it was given is recorded and checked against the stock router."""
    SEEN.append((topk_w, topk_ids))
    return (latent.float() * 0.75 + latent.float().roll(1, dims=1) * 0.25).to(
        latent.dtype
    )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="needs SM100"
)
@pytest.mark.parametrize("impl", ["cuda", "native"])
@pytest.mark.parametrize("tokens", [1, 5, 20, 40, 80, 96])
def test_ep_moe_layer_matches_stock_sequence(tokens, impl):
    from vllm import _custom_ops as ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SituAndMul
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        fused_grouped_topk,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.ep.layer import (
        EpMoeLayer,
    )

    torch.manual_seed(tokens)
    dev = "cuda"

    def bf16(*shape, scale=0.02):
        return (torch.randn(*shape, device=dev) * scale).to(torch.bfloat16)

    gate_up, down = bf16(2 * SHARED, H), bf16(LAT, H)
    w = dict(
        gate=bf16(E, H),
        bias=torch.randn(E, device=dev) * 0.01,
        front=torch.cat([gate_up, down]),
        shared_down=bf16(H, SHARED),
        norm=(torch.rand(LAT, device=dev) + 0.5).to(torch.bfloat16),
        up=bf16(H, LAT),
    )
    x = bf16(tokens, H, scale=1.0)
    with set_current_vllm_config(VllmConfig()):
        situ = SituAndMul(beta=4.0, linear_beta=25.0)
        act = situ.forward_cuda if impl == "cuda" else situ.forward_native
        # 96 > FRONT_MAX: the layer must fall back to the stock order.
        layer = EpMoeLayer(w, act, stand_in, [tokens])
        assert (tokens in layer.batches) == (tokens <= 80)

        logits = torch.mm(x, w["gate"].t(), out_dtype=torch.float32)
        ref_w, ref_ids = fused_grouped_topk(
            x, logits, 16, True, w["bias"], 1, 1, "sigmoid", 1.0
        )
        latent = torch.mm(x, down.t())
        shared = torch.mm(act(torch.mm(x, gate_up.t())), w["shared_down"].t())
        routed = stand_in(latent, ref_w, ref_ids)
        normed = torch.empty_like(routed)
        ops.rms_norm(normed, routed, w["norm"], 1e-5)
        expected = shared.addmm_(normed, w["up"].t())

        SEEN.clear()
        out = layer(x)
        torch.accelerator.synchronize()
        ((got_w, got_ids),) = SEEN
        # Same experts per token (as sets: ties may order differently) and weights.
        order, ref_order = got_ids.sort(dim=1), ref_ids.sort(dim=1)
        assert torch.equal(order.values, ref_order.values)
        torch.testing.assert_close(
            got_w.gather(1, order.indices),
            ref_w.gather(1, ref_order.indices),
            atol=1e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(out, expected, atol=3e-2, rtol=2e-2)
        rel = (out.float() - expected.float()).norm() / expected.float().norm()
        assert rel < 5e-3, rel
        # Row by row (a shared expert row read at the wrong pitch is O(1) off).
        rows = (out.float() - expected.float()).norm(dim=1)
        assert (rows < 1e-2 * expected.float().norm(dim=1)).all(), rows

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = layer(x)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(graph_out, out, atol=0, rtol=0)
