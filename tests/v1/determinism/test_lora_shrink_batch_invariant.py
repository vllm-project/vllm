# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant LoRA shrink: fixed split-K partials, fixed-order reduction."""

import pytest
import torch
from utils import skip_if_not_cuda_alike

import vllm.lora.ops.triton_ops.lora_shrink_op as shrink_op
import vllm.lora.ops.triton_ops.utils as lora_utils
from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_shrink
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
from vllm.utils.torch_utils import set_random_seed

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def batch_invariant(monkeypatch):
    monkeypatch.setattr(lora_utils, "is_batch_invariant", True)
    monkeypatch.setattr(shrink_op, "is_batch_invariant", True)
    lora_utils.get_lora_op_configs.cache_clear()
    yield
    lora_utils.get_lora_op_configs.cache_clear()


def _shrink(inputs, weights, mapping, scaling):
    num_tokens = inputs.size(0)
    meta = LoRAKernelMeta.make(weights[0].size(0), num_tokens, device=DEVICE)
    meta.prepare_tensors(mapping)
    out = torch.zeros(
        (len(weights), num_tokens, weights[0].size(1)),
        dtype=torch.float32,
        device=DEVICE,
    )
    _LORA_A_PTR_DICT.clear()
    args = meta.meta_args(token_nums=num_tokens, specialize_active_lora=False)
    lora_shrink(inputs, weights, out, *args, scaling)
    return out


@skip_if_not_cuda_alike
@pytest.mark.parametrize(
    "hidden_size,rank,nslices,dtype",
    [
        (2560, 8, 3, torch.bfloat16),
        (9728, 16, 1, torch.float16),
        (4097, 64, 1, torch.bfloat16),  # K tail block
        (1024, 8, 1, torch.float16),  # K < 8 * BLOCK_K: some splits are empty
    ],
)
def test_lora_shrink_matches_reference(hidden_size, rank, nslices, dtype):
    set_random_seed(0)
    inputs = torch.randn((33, hidden_size), dtype=dtype, device=DEVICE)
    weights = [
        torch.randn((2, rank, hidden_size), dtype=dtype, device=DEVICE)
        for _ in range(nslices)
    ]
    mapping = torch.randint(-1, 2, (33,), dtype=torch.int32, device=DEVICE)
    out = _shrink(inputs, weights, mapping, 0.5).double().cpu()

    ids, x = mapping.cpu(), inputs.double().cpu()
    ref = torch.zeros_like(out)
    for s, weight in enumerate(weights):
        for lora_id in (0, 1):
            rows = ids == lora_id
            ref[s, rows] = 0.5 * x[rows] @ weight[lora_id].double().cpu().T
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)


@skip_if_not_cuda_alike
@pytest.mark.parametrize("nslices", [1, 3])
def test_lora_shrink_invariant_across_batch_sizes(monkeypatch, nslices):
    reduce_kernel, reduce_grids = shrink_op._lora_shrink_reduce_kernel, []

    class RecordReduce:
        def __getitem__(self, grid):
            reduce_grids.append(grid)
            return reduce_kernel[grid]

    monkeypatch.setattr(shrink_op, "_lora_shrink_reduce_kernel", RecordReduce())
    set_random_seed(0)
    # K = 2560 is ten BLOCK_K=256 blocks, so all 8 splits contribute.
    dtype, hidden_size = torch.bfloat16, 2560
    weights = [
        torch.randn((3, 16, hidden_size), dtype=dtype, device=DEVICE)
        for _ in range(nslices)
    ]
    target = torch.randn((1, hidden_size), dtype=dtype, device=DEVICE)
    target_id = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    expected = _shrink(target, weights, target_id, 0.7)[:, 0]
    for num_tokens in (31, 32, 33, 127, 128, 129):
        inputs = torch.randn((num_tokens, hidden_size), dtype=dtype, device=DEVICE)
        mapping = torch.randint(-1, 3, (num_tokens,), dtype=torch.int32, device=DEVICE)
        pos = num_tokens // 2
        inputs[pos], mapping[pos] = target[0], 0
        out = _shrink(inputs, weights, mapping, 0.7)
        assert torch.equal(out[:, pos], expected), num_tokens
    assert len(reduce_grids) == 7
