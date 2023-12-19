import pytest
import torch

from vllm import topk

iterations = 100
batch_size = 100
vocab_size = 50000
TOPK_TEST = [[10 for i in range(batch_size)]]
TOPS_TEST = [[0.9 for _ in range(batch_size)]]
INPUTS_TEST = [
    3 * (-0.5 + torch.randn(batch_size, vocab_size, device="cuda:0"))
]
DTYPE_TEST = [torch.float32, torch.float16, torch.bfloat16]


def _apply_top_p_top_k_new_kernel(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    do_top_p = True
    do_top_k = True
    softmax_res = logits.softmax(dim=-1, dtype=logits.dtype)
    logit_dst = torch.full(logits.shape,
                           -float("inf"),
                           device=logits.device,
                           dtype=logits.dtype)
    k = k.clamp(1, 1024)
    p = p.type(torch.float32).clamp(0, 1)
    max_top_k = k.max().item()
    topk.top_k(logits, softmax_res, logit_dst, do_top_k, max_top_k, k,
               do_top_p, p)
    return logit_dst


def _apply_top_p_top_k(
    logits: torch.Tensor,
    p: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = probs_sort.cumsum(dim=-1).sub_(probs_sort)
    top_p_mask = probs_sum > p.unsqueeze_(dim=1)

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(logits_idx.shape[-1], device=logits_idx.device)
    top_k_mask = top_k_mask.expand(logits_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze_(dim=1)

    # Final mask.
    mask = (top_p_mask | top_k_mask)
    logits_sort.masked_fill_(mask, -float("inf"))

    # Re-sort the probabilities.
    src = torch.arange(logits_idx.shape[-1],
                       device=logits_idx.device).expand_as(logits_idx)
    logits_idx_inv = torch.empty_like(logits_idx).scatter_(dim=-1,
                                                           index=logits_idx,
                                                           src=src)
    logits = torch.gather(logits_sort, dim=-1, index=logits_idx_inv)
    return logits


@pytest.mark.parametrize("inputs", INPUTS_TEST)
@pytest.mark.parametrize("topps", TOPS_TEST)
@pytest.mark.parametrize("topks", TOPK_TEST)
@pytest.mark.parameterize("data_type", DTYPE_TEST)
def test_topk_speed(inputs, topps, topks, data_type):
    logits = inputs.type(data_type)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        p = torch.FloatTensor(topps).to(logits.device)
        k = torch.IntTensor(topks).to(logits.device)
        k = k.clamp(1, 1024)
        p = p.type(torch.float32).clamp(0, 1)
        _apply_top_p_top_k_new_kernel(logits, p, k)
    end.record()
    torch.cuda.synchronize()
    print(f"time cost of new kernel is {start.elapsed_time(end)/1000:0.4f}s")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        p = torch.FloatTensor(topps).to(logits.device)
        k = torch.IntTensor(topks).to(logits.device)
        _apply_top_p_top_k(logits, p, k)

    end.record()
    torch.cuda.synchronize()
    print(f"time cost of old kernel is {start.elapsed_time(end)/1000:0.4f}s")


@pytest.mark.parametrize("inputs", INPUTS_TEST)
@pytest.mark.parametrize("topps", TOPS_TEST)
@pytest.mark.parametrize("topks", TOPK_TEST)
@pytest.mark.parameterize("data_type", DTYPE_TEST)
def test_topk_acc(inputs, topps, topks, data_type):
    logits = inputs.type(data_type)
    p = torch.FloatTensor(topps).to(logits.device)
    k = torch.IntTensor(topks).to(logits.device)
    k = k.clamp(1, 1024)
    p = p.type(torch.float32).clamp(0, 1)

    logit_new_kernel = _apply_top_p_top_k_new_kernel(logits, p, k)
    logit_old_kernel = _apply_top_p_top_k(logits, p, k)
    logit_old_kernel = logit_old_kernel.to(logits.dtype)
    lnew_sorted = logit_new_kernel.sort(dim=-1, descending=True)[0]
    lold_sorted = logit_old_kernel.sort(dim=-1, descending=True)[0]
    print(lnew_sorted[:, :30])
    print(lold_sorted[:, :30])
    assert torch.allclose(lnew_sorted, lold_sorted)


if __name__ == "__main__":
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        test_topk_acc(INPUTS_TEST[0], TOPS_TEST[0], TOPK_TEST[0], dtype)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        test_topk_speed(INPUTS_TEST[0], TOPS_TEST[0], TOPK_TEST[0], dtype)
