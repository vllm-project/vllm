# SPDX-License-Identifier: Apache-2.0
import math

import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_tpu

if current_platform.is_tpu():
    import torch_xla.core.xla_model as xm

DEVICE = xm.xla_device() if current_platform.is_tpu() else torch.device("cuda")

BATCH_SIZE = 1024
VOCAB_SIZE = 128 * 1024


def test_topk_and_no_op_topp():
    with torch.device(DEVICE):
        if current_platform.is_tpu():
            xm.set_rng_state(seed=33)
        else:
            torch.manual_seed(33)

        logits = torch.rand((BATCH_SIZE, VOCAB_SIZE))

        # Random top-k values between 1 and 9.
        k = torch.randint(1, 10, (BATCH_SIZE, ))

        # Set k=vocab_size for ~50% of requests in the batch (top-k disabled).
        k.masked_fill_(torch.randint(0, 2, (BATCH_SIZE, ), dtype=bool),
                       VOCAB_SIZE)

        # Top-k only implementation
        result1 = apply_top_k_top_p_tpu(logits=logits.clone(), k=k, p=None)

        # Top-p + top-k
        no_op_top_p = torch.tensor([1.0])
        result2 = apply_top_k_top_p_tpu(logits=logits.clone(),
                                        k=k,
                                        p=no_op_top_p)

        assert torch.allclose(result1, result2)


def test_topp_basic():
    with torch.device(DEVICE):
        logits = torch.tensor([[math.log(0.2),
                                math.log(0.3),
                                math.log(0.5)],
                               [math.log(0.5),
                                math.log(0.1),
                                math.log(0.4)]])

        result = apply_top_k_top_p_tpu(logits=logits.clone(),
                                       k=torch.tensor([3, 3]),
                                       p=torch.tensor([0.79, 0.79]))

        # Expect the smallest elements to be dropped.
        expected_result = logits.clone()
        expected_result[0, 0] = float("-inf")
        expected_result[1, 1] = float("-inf")
        assert torch.allclose(expected_result, result)


def test_topp_select_all():
    with torch.device(DEVICE):
        logits = torch.tensor([[math.log(0.2),
                                math.log(0.3),
                                math.log(0.5)],
                               [math.log(0.5),
                                math.log(0.1),
                                math.log(0.4)]])

        result = apply_top_k_top_p_tpu(logits=logits.clone(),
                                       k=torch.tensor([3, 3]),
                                       p=torch.tensor([1.0, 1.0]))

        assert torch.allclose(logits, result)


def test_topp_with_ties():
    with torch.device(DEVICE):
        # Input has multiple math.log(0.3).
        logits = torch.tensor(
            [[math.log(0.3),
              math.log(0.3),
              math.log(0.3),
              math.log(0.1)]])

        result = apply_top_k_top_p_tpu(logits=logits.clone(),
                                       k=torch.tensor([4]),
                                       p=torch.tensor([0.2]))

        # Expect math.log(0.3) to be the only selected element.
        expected_result = torch.tensor([math.log(0.3)])
        assert torch.allclose(expected_result, result[result.isfinite()])


def test_both_topk_topp():
    with torch.device(DEVICE):
        logits = torch.tensor([[math.log(0.2),
                                math.log(0.3),
                                math.log(0.5)],
                               [math.log(0.5),
                                math.log(0.1),
                                math.log(0.4)]])

        # Set k=1 for the first batch.
        result = apply_top_k_top_p_tpu(logits=logits.clone(),
                                       k=torch.tensor([1, 3]),
                                       p=torch.tensor([0.79, 0.79]))

        # Since for the first batch k=1, expect only the largest element gets
        # selected.
        expected_result = logits.clone()
        expected_result[0, 0] = float("-inf")
        expected_result[0, 1] = float("-inf")
        expected_result[1, 1] = float("-inf")
        assert torch.allclose(expected_result, result)
