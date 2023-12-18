from typing import List
import pytest
import torch
from vllm import topk

batch_size = 100
vocab_size = 50000
TOPK_TEST = [
    [32 for _ in range(batch_size)],
    [64 for _ in range(batch_size)]]
TOPS_TEST = [
    [0.3 for _ in range(batch_size)],
    [0.7 for _ in range(batch_size)],
    [0.9 for _ in range(batch_size)]]
INPUTS_TEST = [torch.randn(batch_size, vocab_size, device="cuda:0")]
DTYPE_TEST = [torch.float32, torch.float16, torch.bfloat16]

def _apply_top_p_top_k_with_new_kernel(
    logits: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:
    do_top_p = True
    do_top_k = True
    softmax_res = logits.softmax(dim=-1, dtype=logits.dtype)
    logit_dst = torch.full(logits.shape, -float("inf"), device=logits.device, dtype=logits.dtype)
    max_top_k = 0
    if top_ps:
        p = torch.tensor(top_ps, device=logits.device)
    else:
        p = torch.Tensor()
        do_top_p = False

    if top_ks:
        max_top_k = max(top_ks)
        k = torch.tensor(top_ks, dtype=torch.int32, device=logits.device)
    else:
        k = torch.Tensor()
        do_top_k = False
    topk.top_k(logits, softmax_res, logit_dst, do_top_k, max_top_k, k,
               do_top_p, p)
    return logit_dst


def _apply_top_p_top_k(
    logits: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:

    p = torch.tensor(top_ps, dtype=logits.dtype, device=logits.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=logits.device)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)
    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    logits_sort[top_p_mask] = -float("inf")

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    logits_sort[top_k_mask] = -float("inf")

    # Re-sort the probabilities.
    logits = torch.gather(logits_sort,
                          dim=-1,
                          index=torch.argsort(logits_idx, dim=-1))
    return logits


@pytest.mark.parametrize("inputs", INPUTS_TEST)
@pytest.mark.parametrize("topps", TOPS_TEST)
@pytest.mark.parametrize("topks", TOPK_TEST)
@pytest.mark.parameterize("data_type", DTYPE_TEST)
def test_topk_kernel(inputs, topps, topks, data_type):
    INPUT = inputs.type(data_type)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    pre = torch.cuda.max_memory_allocated(device="cuda:0")
    logit_with_new_kernel = _apply_top_p_top_k_with_new_kernel(INPUT, TOPS_TEST[0],
                                       TOPK_TEST[0])
    aft = torch.cuda.max_memory_allocated(device="cuda:0")
    end.record()
    torch.cuda.synchronize()
    print(f"time cost of new kernel is {start.elapsed_time(end)/1000}s")
    print(f"memory cost of new kernel cost is {aft-pre}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    pre = torch.cuda.max_memory_allocated(device="cuda:0")
    logit_with_old_kernel = _apply_top_p_top_k(INPUT, TOPS_TEST[0], TOPK_TEST[0])
    aft = torch.cuda.max_memory_allocated(device="cuda:0")
    end.record()
    torch.cuda.synchronize()
    print(f"time cost of old kernel is {start.elapsed_time(end)/1000}s")
    print(f"memory cost of old kernel cost is {aft-pre}")

    lnew_sorted = logit_with_new_kernel.sort(dim=-1, descending=True)[0]
    lold_sorted = logit_with_old_kernel.sort(dim=-1, descending=True)[0]

    assert torch.allclose(lnew_sorted, lold_sorted)

if __name__ == "__main__":
    test_topk_kernel(INPUTS_TEST[0], TOPS_TEST[0], TOPK_TEST[0], torch.float32)
    test_topk_kernel(INPUTS_TEST[0], TOPS_TEST[0], TOPK_TEST[0], torch.float16)
    test_topk_kernel(INPUTS_TEST[0], TOPS_TEST[0], TOPK_TEST[0], torch.bfloat16)
