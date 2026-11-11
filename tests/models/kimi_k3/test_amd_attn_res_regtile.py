# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD AttnRes requires ROCm",
)

HIDDEN_SIZE = 7168
BLOCK_CAPACITY = 9


def _make_inputs(num_tokens: int, has_delta: bool, seed: int = 0):
    torch.manual_seed(seed)

    def randn(*shape: int, padding: int = 0) -> torch.Tensor:
        storage = torch.randn(
            *shape[:-1], shape[-1] + padding, device="cuda", dtype=torch.bfloat16
        )
        return storage[..., : shape[-1]]

    prefix = randn(num_tokens, HIDDEN_SIZE, padding=7)
    delta = randn(num_tokens, HIDDEN_SIZE, padding=11) if has_delta else None
    blocks = randn(num_tokens, BLOCK_CAPACITY, HIDDEN_SIZE, padding=13)
    norm = 1 + 0.1 * torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    # Large enough logits that the softmax is far from uniform.
    score = 4 * torch.randn_like(norm) / HIDDEN_SIZE**0.5
    out_norm = 1 + 0.1 * torch.randn_like(norm)
    return prefix, delta, blocks, norm, score, out_norm


def _run(regtile: bool, monkeypatch, inputs, **kwargs):
    monkeypatch.setenv("VLLM_K3_ATTN_RES_REGTILE", "1" if regtile else "0")
    prefix, delta, blocks, norm, score, out_norm = (
        None if x is None else x.clone() for x in inputs
    )
    result = attn_res(prefix, delta, blocks, norm, score, out_norm, **kwargs)
    return result, prefix, blocks


def _fp8_ordinal(x: torch.Tensor) -> torch.Tensor:
    # Codes are sign-magnitude; order them by value so a step across zero
    # stays adjacent.
    code = x.view(torch.uint8).to(torch.int16)
    return torch.where(code >= 0x80, 0x80 - code, code)


def _assert_fp8_close(actual, expected) -> None:
    # Rounding can move values at FP8 midpoints; bound each change to the
    # adjacent code and bound how often it occurs.
    output, scale = actual
    ref_output, ref_scale = expected
    torch.testing.assert_close(scale, ref_scale, atol=1e-7, rtol=1e-5)
    code_delta = (_fp8_ordinal(output) - _fp8_ordinal(ref_output)).abs()
    assert code_delta.max().item() <= 1
    assert (code_delta != 0).float().mean().item() < 1e-3


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64, 192, 256, 257, 640])
@pytest.mark.parametrize("num_blocks", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize(
    "has_delta,write_block,apply_output_norm,quant",
    [
        pytest.param(True, True, True, True, id="pre-attn-fp8"),
        pytest.param(True, False, True, False, id="pre-mlp-bf16"),
        pytest.param(False, False, True, True, id="no-delta-fp8"),
        pytest.param(True, False, False, False, id="final-mixture"),
    ],
)
def test_regtile_matches_online_softmax(
    num_tokens,
    num_blocks,
    has_delta,
    write_block,
    apply_output_norm,
    quant,
    monkeypatch,
):
    inputs = _make_inputs(num_tokens, has_delta)
    if not apply_output_norm:
        inputs = inputs[:5] + (None,)
    kwargs = dict(
        num_blocks=num_blocks,
        block_write_idx=num_blocks if write_block else -1,
        eps=1e-5,
        output_norm_eps=2e-5,
        quant_dtype=current_platform.fp8_dtype() if quant else None,
    )
    expected, expected_prefix, expected_blocks = _run(
        False, monkeypatch, inputs, **kwargs
    )
    actual, actual_prefix, actual_blocks = _run(True, monkeypatch, inputs, **kwargs)

    # The residual stream must be bit-identical.
    torch.testing.assert_close(actual_prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(actual_blocks, expected_blocks, atol=0, rtol=0)
    if quant:
        _assert_fp8_close(actual, expected)
    else:
        # Only the softmax summation order differs (one pass vs online).
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1.6e-2)

    # And both match an FP32 reference.
    prefix, delta, blocks, norm, score, out_norm = inputs
    values = torch.cat(
        (blocks[:, :num_blocks].float(), expected_prefix.float().unsqueeze(1)), dim=1
    )
    keys = F.rms_norm(values, (HIDDEN_SIZE,), norm.float(), 1e-5)
    probs = (keys @ score.float()).softmax(dim=-1)
    reference = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
    if out_norm is not None:
        reference = F.rms_norm(reference, (HIDDEN_SIZE,), out_norm.float(), 2e-5)
    if quant:
        output, scale = actual
        dequant = output.float() * scale
        torch.testing.assert_close(dequant, reference, atol=0.15, rtol=0.07)
    else:
        torch.testing.assert_close(actual.float(), reference, atol=5e-2, rtol=2e-2)


def _attn_res_kernel_names(fn, prefix, delta, blocks) -> set[str]:
    def padded_clone(x: torch.Tensor) -> torch.Tensor:
        # Keep the row padding: the compiled graph specialized on the strides.
        return torch.empty_strided(
            x.shape, x.stride(), dtype=x.dtype, device=x.device
        ).copy_(x)

    # The ROCm profiler occasionally drops a kernel record; retry.
    for _ in range(3):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(3):
                fn(padded_clone(prefix), delta, padded_clone(blocks))
            torch.cuda.synchronize()
        names = {e.name for e in prof.events() if "attn_res" in e.name}
        if names:
            return names
    return names


def test_regtile_dispatch_under_torch_compile(monkeypatch):
    """The 8-row split must follow the runtime token count even though vLLM
    traces one graph (at a large size) and drops the shape guards."""
    monkeypatch.setenv("VLLM_K3_ATTN_RES_REGTILE", "1")
    num_blocks = 8
    weights = _make_inputs(1, False)[3:]

    def step(prefix, delta, blocks):
        return attn_res(
            prefix,
            delta,
            blocks,
            *weights,
            num_blocks=num_blocks,
            block_write_idx=-1,
            eps=1e-5,
            output_norm_eps=1e-5,
            quant_dtype=current_platform.fp8_dtype(),
        )

    torch._dynamo.reset()
    # Count only this test's graphs; the counters outlive dynamo.reset().
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        step,
        fullgraph=True,
        dynamic=True,
        options={"guard_filter_fn": lambda guards: [False for _ in guards]},
    )
    # Trace at a size that takes the fallback kernel, as the profile run does.
    for num_tokens in (1024, 4, 192, 256, 257, 640):
        prefix, delta, blocks, *_ = _make_inputs(num_tokens, True, seed=num_tokens)
        eager_prefix, eager_blocks = prefix.clone(), blocks.clone()
        expected = step(eager_prefix, delta, eager_blocks)
        names = _attn_res_kernel_names(compiled, prefix, delta, blocks)
        actual = compiled(prefix, delta, blocks)
        if num_tokens <= 256:
            assert names and all("regtile" in n for n in names), (num_tokens, names)
        else:
            assert names and not any("regtile" in n for n in names), (
                num_tokens,
                names,
            )
        _assert_fp8_close(actual, expected)
        torch.testing.assert_close(prefix, eager_prefix, atol=0, rtol=0)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] <= 1

    # And the compiled split replays under CUDA graph capture.
    prefix, delta, blocks, *_ = _make_inputs(4, True, seed=4)
    initial_prefix = prefix.clone()
    expected = step(prefix.clone(), delta, blocks.clone())
    compiled(prefix, delta, blocks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = compiled(prefix, delta, blocks)
    prefix.copy_(initial_prefix)
    graph.replay()
    torch.cuda.synchronize()
    _assert_fp8_close(graph_out, expected)
