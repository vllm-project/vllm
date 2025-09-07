# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

# isort: off
from vllm.v1.sample.tpu.sampler import (apply_top_k_top_p as
                                        apply_top_k_top_p_tpu)
# isort: on

if not current_platform.is_tpu():
    pytest.skip("This test needs a TPU.", allow_module_level=True)
import torch_xla.core.xla_model as xm

BATCH_SIZE = 1024
VOCAB_SIZE = 128 * 1024
TOLERANCE = 1e-6


def test_topk_equivalence_to_native_impl():
    with torch.device(xm.xla_device()):
        xm.set_rng_state(seed=33)

        logits = torch.rand((BATCH_SIZE, VOCAB_SIZE))

        # Random top-k values between 1 and 10.
        k = torch.randint(1, 10, (BATCH_SIZE, ))

        # Set k=vocab_size for ~50% of requests in the batch (top-k disabled).
        k.masked_fill_(torch.randint(0, 2, (BATCH_SIZE, ), dtype=bool),
                       VOCAB_SIZE)

        result_tpu = apply_top_k_top_p_tpu(logits=logits.clone(), k=k, p=None)

        result_native = apply_top_k_top_p(logits=logits.clone(), k=k, p=None)
        assert torch.allclose(result_native, result_tpu)


def test_topp_result_sums_past_p():
    with torch.device(xm.xla_device()):
        xm.set_rng_state(seed=33)

        logits = torch.rand((BATCH_SIZE, VOCAB_SIZE))
        probs = logits.softmax(dim=-1)

        # Random top-p values between 0 and 1.
        p = torch.rand((BATCH_SIZE, ))

        # Set p=1 for ~50% of requests in the batch (top-p disabled).
        p.masked_fill_(torch.randint(0, 2, (BATCH_SIZE, ), dtype=bool), 1)

        no_op_k = torch.tensor([VOCAB_SIZE])
        logits_masked = apply_top_k_top_p_tpu(logits=logits.clone(),
                                              k=no_op_k,
                                              p=p)

        # Verify that the masked logit's probability sums to at least p.
        probs.masked_fill_(logits_masked.isinf(), 0)
        masked_prob_sum = probs.sum(dim=-1)

        xm.mark_step()

    # Perform assertion on CPU.
    assert torch.all(torch.ge(masked_prob_sum.cpu() + TOLERANCE, p.cpu()))


def test_topp_basic():
    with torch.device(xm.xla_device()):
        logits = torch.tensor([[math.log(0.2),
                                math.log(0.3),
                                math.log(0.5)],
                               [math.log(0.5),
                                math.log(0.1),
                                math.log(0.4)]])

        result = apply_top_k_top_p_tpu(logits=logits.clone(),
                                       k=torch.tensor([3, 3]),
                                       p=torch.tensor([0.79, 0.79]))

        xm.mark_step()

    # Expect the smallest elements to be dropped.
    expected_result = logits.clone().cpu()
    expected_result[0, 0] = float("-inf")
    expected_result[1, 1] = float("-inf")
    assert torch.allclose(expected_result, result.cpu())


def test_topp_select_all():
    with torch.device(xm.xla_device()):
        logits = torch.tensor([[math.log(0.2),
                                math.log(0.3),
                                math.log(0.5)],
                               [math.log(0.5),
                                math.log(0.1),
                                math.log(0.4)]])

        result = apply_top_k_top_p_tpu(logits=logits.clone(),
                                       k=torch.tensor([3, 3]),
                                       p=torch.tensor([1.0, 1.0]))

        xm.mark_step()

    assert torch.allclose(logits.cpu(), result.cpu())


def test_topp_with_ties():
    with torch.device(xm.xla_device()):
        # Input has multiple math.log(0.3).
        logits = torch.tensor(
            [[math.log(0.3),
              math.log(0.3),
              math.log(0.3),
              math.log(0.1)]])

        result = apply_top_k_top_p_tpu(logits=logits.clone(),
                                       k=torch.tensor([4]),
                                       p=torch.tensor([0.2]))

        xm.mark_step()

    # All tie values are included in the top-p set. Tie breaking is left
    # to be done during final sampling (all tie tokens have equal
    # probability of being chosen).
    expected_result = logits.clone().cpu()
    expected_result[0, 3] = float("-inf")
    assert torch.allclose(expected_result, result.cpu())


def test_both_topk_topp():
    with torch.device(xm.xla_device()):
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

        xm.mark_step()

    # Since for the first batch k=1, expect only the largest element gets
    # selected.
    expected_result = logits.clone().cpu()
    expected_result[0, 0] = float("-inf")
    expected_result[0, 1] = float("-inf")
    expected_result[1, 1] = float("-inf")
    assert torch.allclose(expected_result, result.cpu())
