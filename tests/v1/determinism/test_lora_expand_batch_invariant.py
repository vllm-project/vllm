# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant LoRA expand: the delta is added to the base with one rounding."""

import pytest
import torch
from utils import skip_if_not_cuda_alike

import vllm.lora.ops.triton_ops.lora_expand_op as expand_op
import vllm.lora.ops.triton_ops.utils as lora_utils
from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand
from vllm.lora.ops.triton_ops.utils import _LORA_B_PTR_DICT
from vllm.utils.torch_utils import set_random_seed

DEVICE = "cuda"


@pytest.fixture
def set_batch_invariant(monkeypatch):
    def _set(enabled: bool):
        monkeypatch.setattr(lora_utils, "is_batch_invariant", enabled)
        monkeypatch.setattr(expand_op, "is_batch_invariant", enabled)
        lora_utils.get_lora_op_configs.cache_clear()

    yield _set
    lora_utils.get_lora_op_configs.cache_clear()


def _expand(inputs, weights, base, mapping):
    """Return base + inputs[s] @ weights[s][lora].T per slice s (add_inputs=True)."""
    num_tokens = base.size(0)
    meta = LoRAKernelMeta.make(weights[0].size(0), num_tokens, device=DEVICE)
    meta.prepare_tensors(mapping)
    out = base.clone()
    _LORA_B_PTR_DICT.clear()
    args = meta.meta_args(token_nums=num_tokens, specialize_active_lora=False)
    lora_expand(inputs, weights, out, *args, offset_start=0, add_inputs=True)
    return out


def _exact_problem(num_tokens, hidden_size, rank, nslices, num_loras):
    # Multiples of 1/64 with small magnitudes: every product and sum below is
    # exact in fp32, so the fp32 delta equals the exact delta and the only
    # rounding is the final one (or two, on the default path).
    def grid(shape, lo, hi):
        return torch.randint(lo, hi, shape, device=DEVICE).float() / 64

    inputs = grid((nslices, num_tokens, rank), -64, 65)
    weights = [
        grid((num_loras, hidden_size, rank), -64, 65).to(torch.bfloat16)
        for _ in range(nslices)
    ]
    base = grid((num_tokens, nslices * hidden_size), -256, 257).to(torch.bfloat16)
    mapping = torch.randint(
        -1, num_loras, (num_tokens,), dtype=torch.int32, device=DEVICE
    )
    delta = torch.zeros(num_tokens, nslices * hidden_size, device=DEVICE)
    for s, weight in enumerate(weights):
        for lora_id in range(num_loras):
            rows = mapping == lora_id
            delta[rows, s * hidden_size : (s + 1) * hidden_size] = (
                inputs[s, rows].double() @ weight[lora_id].double().T
            ).float()
    return inputs, weights, base, mapping, delta


@skip_if_not_cuda_alike
@pytest.mark.parametrize(
    "hidden_size,rank,nslices",
    [(2048, 16, 1), (1024, 64, 3), (4097, 8, 2)],  # last: N tail block
)
def test_lora_expand_single_rounding(set_batch_invariant, hidden_size, rank, nslices):
    set_random_seed(0)
    inputs, weights, base, mapping, delta = _exact_problem(
        65, hidden_size, rank, nslices, num_loras=3
    )
    no_lora = mapping == -1
    single = (base.float() + delta).to(torch.bfloat16)
    double = (base.float() + delta.to(torch.bfloat16).float()).to(torch.bfloat16)
    # The two roundings must disagree somewhere, or the test proves nothing.
    assert (single != double)[~no_lora].sum() > 100

    set_batch_invariant(True)
    out = _expand(inputs, weights, base, mapping)
    assert torch.equal(out, single)
    assert torch.equal(out[no_lora], base[no_lora])

    # The default path is unchanged: the delta is rounded before the add.
    set_batch_invariant(False)
    out = _expand(inputs, weights, base, mapping)
    assert torch.equal(out, double)


@skip_if_not_cuda_alike
@pytest.mark.parametrize("nslices", [1, 3])
def test_lora_expand_invariant_across_batch_sizes(set_batch_invariant, nslices):
    set_batch_invariant(True)
    set_random_seed(0)
    hidden_size, rank = 2048, 16
    weights = [
        torch.randn((3, hidden_size, rank), dtype=torch.bfloat16, device=DEVICE)
        for _ in range(nslices)
    ]
    x = torch.randn((nslices, 1, rank), device=DEVICE)
    y = torch.randn((1, nslices * hidden_size), dtype=torch.bfloat16, device=DEVICE)
    target_id = torch.ones(1, dtype=torch.int32, device=DEVICE)
    expected = _expand(x, weights, y, target_id)[0]
    for num_tokens in (2, 31, 64, 129, 256):
        inputs = torch.randn((nslices, num_tokens, rank), device=DEVICE)
        base = torch.randn(
            (num_tokens, nslices * hidden_size), dtype=torch.bfloat16, device=DEVICE
        )
        mapping = torch.randint(-1, 3, (num_tokens,), dtype=torch.int32, device=DEVICE)
        for pos in (0, num_tokens // 2, num_tokens - 1):
            inputs[:, pos], base[pos], mapping[pos] = x[:, 0], y[0], 1
            out = _expand(inputs, weights, base, mapping)
            assert torch.equal(out[pos], expected), (num_tokens, pos)
