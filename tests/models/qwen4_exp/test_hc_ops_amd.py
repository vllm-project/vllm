# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""qwen4_exp HyperConnection combine_and_mix on ROCm, fused (gfx1100) and unfused."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="AMD HyperConnection ops require ROCm"
)

HC = 4
HIDDEN_SIZE = 2560
HYPER_HIDDEN_SIZE = HC * HIDDEN_SIZE
LORA_RANK = 320
DOWN_N = LORA_RANK + HC + 12  # merged down+inject weight, 16-row padded
EPS = 1e-6


def _inputs(num_tokens: int, shared_norm: bool):
    g = torch.Generator(device="cuda").manual_seed(num_tokens)

    def randn(*shape, scale=1.0):
        x = torch.randn(*shape, generator=g, device="cuda", dtype=torch.float32)
        return (x * scale).to(torch.float16)

    norm_size = HIDDEN_SIZE if shared_norm else HYPER_HIDDEN_SIZE
    return (
        randn(num_tokens, HYPER_HIDDEN_SIZE),
        randn(num_tokens, HIDDEN_SIZE),
        randn(num_tokens, HC),
        randn(norm_size, scale=0.1),
        randn(DOWN_N, HYPER_HIDDEN_SIZE, scale=0.02),
        randn(HYPER_HIDDEN_SIZE, LORA_RANK, scale=0.05),
    )


def _reference(residual, block, inj, norm_w, down_w, up_w):
    """Fp32 math with the fp16 roundings of the unfused module."""
    gain = 2.0 * torch.sigmoid(inj.float() / HC)
    hidden = (
        residual.float().unflatten(-1, (HC, HIDDEN_SIZE))
        + block.float()[:, None, :] * gain[:, :, None]
    ).to(torch.float16)
    streams = hidden.float()
    rrms = torch.rsqrt(streams.square().mean(-1, keepdim=True) + EPS)
    weight = norm_w.float().reshape(-1, HIDDEN_SIZE)
    xn = (streams * rrms * (1.0 + weight)).to(torch.float16)
    down = (xn.flatten(-2).float() @ down_w.float().T).to(torch.float16)
    lora = down[:, :LORA_RANK].float() / HC
    lora = (lora * torch.sigmoid(lora)).to(torch.float16)
    gate = (lora.float() @ up_w.float().T).to(torch.float16)
    gate = gate.float().unflatten(-1, (HC, HIDDEN_SIZE))
    block_input = (torch.sigmoid(gate) * xn.float()).mean(-2).to(torch.float16)
    return hidden.flatten(-2), block_input, down[:, LORA_RANK : LORA_RANK + HC]


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


# 1-2 tokens: fully fused kernel; 3-4: Triton combine+norm then the fused mix;
# 8: unfused chain.
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("shared_norm", [False, True])
def test_hc_combine_and_mix_matches_reference(
    num_tokens: int, shared_norm: bool
) -> None:
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_and_mix

    args = _inputs(num_tokens, shared_norm)
    # The first layer's injection is a split() view of the down projection.
    padded = torch.zeros(num_tokens, 2 * HC, dtype=torch.float16, device="cuda")
    padded[:, :HC] = args[2]
    args = (*args[:2], padded[:, :HC], *args[3:])
    hidden, block_input, injection = hc_combine_and_mix(*args, EPS, HC)
    ref_hidden, ref_block_input, ref_injection = _reference(*args)

    torch.testing.assert_close(hidden, ref_hidden)
    assert _rel_l2(block_input, ref_block_input) < 2e-3
    assert _rel_l2(injection, ref_injection) < 2e-3


@pytest.mark.skipif(
    not hasattr(torch.ops._rocm_C, "qwen4_hc_combine_mix"),
    reason="fused HyperConnection kernel is built for gfx1100 only",
)
def test_hc_fused_is_deterministic() -> None:
    """The fused kernel reduces in a fixed order (no split-K atomics)."""
    from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_and_mix

    args = _inputs(1, shared_norm=False)
    first = hc_combine_and_mix(*args, EPS, HC)[1]
    for _ in range(20):
        torch.testing.assert_close(
            hc_combine_and_mix(*args, EPS, HC)[1], first, atol=0, rtol=0
        )
