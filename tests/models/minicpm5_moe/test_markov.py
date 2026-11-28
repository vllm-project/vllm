# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch


def dense_reference(base, w1, w2, anchor, prefer_nan):
    previous = anchor
    out = []
    for step in range(base.shape[0]):
        bias = (w2.float() @ w1[previous].float()).half()
        scores = base[step] + bias
        if not prefer_nan:
            scores = scores.masked_fill(scores.isnan(), -torch.inf)
        previous = int(scores.argmax())
        out.append(previous)
    return out


@pytest.mark.skipif(
    os.environ.get("VLLM_TARGET_DEVICE") == "cpu" or not torch.xpu.is_available(),
    reason="Requires XPU Graph replay",
)
@pytest.mark.parametrize("prefer_nan", [False, True])
def test_markov_pruning_graph_against_dense_reference(prefer_nan):
    from vllm.models.minicpm5_moe.xpu.specialized_decode.markov import markov_greedy

    gen = torch.Generator(device="cpu").manual_seed(73)
    vocab, rank, steps = 257, 256, 7
    # Exactly representable products preserve the deliberately tied scores.
    w1 = (torch.randn(vocab, rank, generator=gen) * 8).round().half() / 32
    w2 = (torch.randn(vocab, rank, generator=gen) * 8).round().half() / 32
    w2[19] = w2[17]
    base = torch.randn(steps, vocab, generator=gen).half()
    base[1, 17] = base[1, 19] = 20
    base[2, 33] = base[2, 97] = torch.nan
    base[3].fill_(-torch.inf)
    base[5, 63] = torch.inf

    gpu_w1, gpu_w2 = w1.to("xpu"), w2.to("xpu")
    norms = gpu_w2.float().square().sum(-1).sqrt()
    inputs = base.to("xpu")
    anchor = torch.tensor([7], device="xpu")
    output = torch.empty(steps, dtype=torch.int64, device="xpu")

    def run():
        markov_greedy(
            inputs, gpu_w1, gpu_w2, norms, anchor, output, prefer_nan=prefer_nan
        )

    run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        run()
    for shift, token in ((0, 7), (1, 152), (3, 33), (0, 7)):
        sample = base.roll(shift, 1)
        inputs.copy_(sample)
        anchor.fill_(token)
        graph.replay()
        assert output.tolist() == dense_reference(sample, w1, w2, token, prefer_nan)
