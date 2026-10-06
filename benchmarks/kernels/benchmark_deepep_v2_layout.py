# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Separate DeepEP layout and CPU-count synchronization costs.

Run each case in a fresh process with torchrun --nproc-per-node=2. This measures
fresh dispatch, GPU metadata, a synthetic expert operation, and combine. It
defaults to a synthetic scalar expert. Select --expert-kernel grouped-fp8 to
compare the existing input permutation with direct grouped FP8 GEMM inputs.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

CASES = {
    "nonexpanded-nosync": (False, False),
    "nonexpanded-sync": (False, True),
    "expanded-nosync": (True, False),
    "expanded-sync": (True, True),
}


def expanded_metadata(
    prefix: torch.Tensor, capacity: int, alignment: int, expert_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode GPU expert prefixes without reading counts on the host.

    prefix[e] is the aligned start of expert e plus its real token count.
    Return one global expert ID per allocated row (-1 for alignment/tail
    padding), and the real per-expert counts. This is a correctness prototype;
    a fused metadata kernel should be evaluated before serving integration.
    """
    if prefix.ndim != 1 or prefix.numel() == 0:
        raise ValueError("Expected a nonempty vector of expert prefixes")
    if alignment < 1 or capacity < 0:
        raise ValueError("Alignment must be positive and capacity nonnegative")
    aligned_ends = ((prefix + alignment - 1) // alignment) * alignment
    starts = torch.cat((torch.zeros_like(prefix[:1]), aligned_ends[:-1]))
    counts = prefix - starts
    rows = torch.arange(capacity, device=prefix.device, dtype=prefix.dtype)
    experts = torch.searchsorted(aligned_ends, rows, right=True)
    safe_experts = experts.clamp_max(prefix.numel() - 1)
    valid = (experts < prefix.numel()) & (rows < prefix[safe_experts])
    ids = torch.where(valid, experts + expert_offset, -1).to(torch.int64)
    return ids, counts


def synthetic_experts(recv_x, recv_ids, recv_weights, handle, expanded, offset):
    """Apply an expert-dependent scalar so dispatch/combine has a reference."""
    if expanded:
        ids, counts = expanded_metadata(
            handle.psum_num_recv_tokens_per_expert,
            recv_x.shape[0],
            handle.expert_alignment,
            offset,
        )
        weights = recv_weights
    else:
        rows = torch.arange(recv_x.shape[0], device=recv_x.device)
        valid_rows = rows < handle.psum_num_recv_tokens_per_scaleup_rank[-1]
        valid = valid_rows[:, None] & (recv_ids >= 0)
        ids = torch.where(valid, recv_ids + offset, -1)
        weights = recv_weights
        counts = None
    # Mask weights before arithmetic: unused receive rows are uninitialized.
    weights = torch.where(ids >= 0, weights, 0)
    factors = ((ids.clamp_min(0) % 4).float() + 1) / 4
    factors = factors * weights
    if not expanded:
        factors = factors.sum(dim=1)
    safe_x = (
        torch.where((ids >= 0).any(dim=1)[:, None], recv_x, 0)
        if (not expanded)
        else torch.where((ids >= 0)[:, None], recv_x, 0)
    )
    return (safe_x.float() * factors[:, None]).to(recv_x.dtype), counts


class GroupedFp8Experts:
    """Compare input permutation with direct DeepEP grouped GEMM input."""

    def __init__(self, local_experts: int, hidden: int, offset: int):
        from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout

        self.alignment = get_mk_alignment_for_contiguous_layout()[0]
        self.offset = offset
        self.local_experts = local_experts
        factors = ((torch.arange(local_experts, device="cuda") + offset) % 4 + 1) / 4
        self.weight = (
            torch.eye(hidden, device="cuda")[None] * factors[:, None, None]
        ).to(torch.float8_e4m3fn)
        self.weight_scale = torch.ones(
            local_experts, hidden // 128, hidden // 128, device="cuda"
        )

    def __call__(self, recv_x, recv_ids, recv_weights, handle, expanded):
        from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
            deepgemm_moe_permute,
            deepgemm_unpermute_and_reduce,
        )
        from vllm.utils.deep_gemm import (
            m_grouped_fp8_gemm_nt_contiguous,
            mk_alignment_scope,
        )

        aq, scales = recv_x
        rows, hidden = aq.shape
        counts = None
        if expanded:
            if handle.expert_alignment != self.alignment:
                raise ValueError("DeepEP and DeepGEMM expert alignment must match")
            ids, counts = expanded_metadata(
                handle.psum_num_recv_tokens_per_expert,
                rows,
                self.alignment,
                self.offset,
            )
            m_indices = torch.where(ids >= 0, ids - self.offset, -1).int()
            # DeepEP has already duplicated and grouped the activation rows.
            gemm_input, gemm_scales = aq, scales
            alignment = self.alignment
        else:
            valid_rows = (
                torch.arange(rows, device=aq.device)
                < handle.psum_num_recv_tokens_per_scaleup_rank[-1]
            )
            ids = torch.where(valid_rows[:, None] & (recv_ids >= 0), recv_ids, -1)
            gemm_input, gemm_scales, m_indices, inverse, alignment = (
                deepgemm_moe_permute(
                    aq=aq,
                    aq_scale=scales,
                    topk_ids=ids,
                    local_num_experts=self.local_experts,
                    expert_map=None,
                    expert_tokens_meta=None,
                )
            )
        mm = torch.zeros(
            gemm_input.shape[0], hidden, dtype=torch.bfloat16, device=aq.device
        )
        with mk_alignment_scope(alignment):
            m_grouped_fp8_gemm_nt_contiguous(
                (gemm_input, gemm_scales),
                (self.weight, self.weight_scale),
                mm,
                m_indices,
            )
        weights = torch.where(ids >= 0, recv_weights, 0)
        if expanded:
            safe_mm = torch.where((ids >= 0)[:, None], mm, 0)
            return (safe_mm.float() * weights[:, None]).bfloat16(), counts
        out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=aq.device)
        deepgemm_unpermute_and_reduce(
            a=mm,
            topk_ids=ids,
            topk_weights=weights,
            inv_perm=inverse,
            expert_map=None,
            output=out,
        )
        return out, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--tokens-per-rank", default="8192,8192")
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument(
        "--expert-kernel", choices=["scalar", "grouped-fp8"], default="scalar"
    )
    parser.add_argument("--local-experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--expert-alignment", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("Warmup and measured iteration counts must be positive")
    if min(args.local_experts, args.hidden_size, args.topk, args.expert_alignment) < 1:
        parser.error("Dimensions and expert alignment must be positive")
    if args.hidden_size % 256:
        parser.error("BF16 hidden size must be a multiple of 256")

    import deep_ep

    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    token_counts = [int(n) for n in args.tokens_per_rank.split(",")]
    if len(token_counts) != world or min(token_counts) < 0:
        raise ValueError("Provide one nonnegative token count per rank")
    num_experts = args.local_experts * world
    if args.topk > num_experts:
        raise ValueError("topk cannot exceed the global expert count")
    # Force NCCL initialization before checking the GIN prerequisite.
    probe = torch.zeros(1, device="cuda")
    dist.all_reduce(probe)
    gin_disabled = os.environ.get("EP_DISABLE_GIN") == "1"
    if not gin_disabled:
        from vllm.utils.nccl import query_nccl_gin_type

        gin = query_nccl_gin_type(dist.group.WORLD)
        if gin is None or gin == 0:
            raise RuntimeError("DeepEP v2 requires GIN unless EP_DISABLE_GIN=1")
    # Verify identical experimental policy before issuing any DeepEP calls.
    signatures = [None] * world
    dist.all_gather_object(signatures, vars(args))
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError("All EP ranks must use identical benchmark arguments")

    torch.manual_seed(42 + rank)
    tokens = token_counts[rank]
    # Values are exactly representable in BF16 and FP8 for the GEMM reference.
    x = (torch.randint(-8, 9, (tokens, args.hidden_size), device="cuda") / 8).bfloat16()
    ids = torch.rand(tokens, num_experts, device="cuda").argsort(dim=1)
    ids = ids[:, : args.topk].contiguous().to(torch.int64)
    weights = torch.full((tokens, args.topk), 1 / args.topk, device="cuda")
    factor = (((ids % 4).float() + 1) / 4 * weights).sum(dim=1)
    reference = (x.float() * factor[:, None]).bfloat16()
    capacity = 1 << max(max(token_counts) - 1, 0).bit_length()
    expanded, cpu_sync = CASES[args.case]
    grouped = (
        GroupedFp8Experts(
            args.local_experts, args.hidden_size, rank * args.local_experts
        )
        if args.expert_kernel == "grouped-fp8"
        else None
    )
    if grouped is not None:
        args.expert_alignment = grouped.alignment
        dispatch_input = (
            x.to(torch.float8_e4m3fn),
            torch.ones(tokens, args.hidden_size // 128, device="cuda"),
        )
    else:
        dispatch_input = x
    buffer = deep_ep.ElasticBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=capacity,
        hidden=args.hidden_size,
        num_topk=args.topk,
        use_fp8_dispatch=grouped is not None,
        allow_hybrid_mode=False,
        explicitly_destroy=True,
    )

    physical_domain = tuple(buffer.get_physical_domain_size())
    if gin_disabled and physical_domain != (1, world):
        buffer.destroy()
        dist.destroy_process_group()
        raise RuntimeError("GIN-disabled experiment requires one NVLink domain")

    def step():
        recv_x, recv_ids, recv_weights, handle, _ = buffer.dispatch(
            x=dispatch_input,
            topk_idx=ids,
            topk_weights=weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=capacity,
            expert_alignment=args.expert_alignment,
            do_expand=expanded,
            do_cpu_sync=cpu_sync,
            async_with_compute_stream=False,
        )
        if grouped is not None:
            y, counts = grouped(recv_x, recv_ids, recv_weights, handle, expanded)
            received_rows = recv_x[0].shape[0]
        else:
            y, counts = synthetic_experts(
                recv_x,
                recv_ids,
                recv_weights,
                handle,
                expanded,
                rank * args.local_experts,
            )
            received_rows = recv_x.shape[0]
        # Expert outputs are already weighted; combine only reverses routing.
        out, _, _ = buffer.combine(x=y, handle=handle, async_with_compute_stream=False)
        return out, received_rows, counts, handle

    try:
        for _ in range(args.warmup):
            out, received_capacity, counts, handle = step()
        torch.testing.assert_close(out, reference, atol=0.01, rtol=0.03)
        if counts is not None:
            assert torch.all(counts >= 0).item()
            if cpu_sync:
                torch.testing.assert_close(
                    (
                        (counts.cpu() + args.expert_alignment - 1)
                        // args.expert_alignment
                        * args.expert_alignment
                    ),
                    torch.tensor(handle.num_recv_tokens_per_expert_list),
                    check_dtype=False,
                )
        del out, counts, handle
        dist.barrier()
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            out, received_capacity, counts, handle = step()
            # Deliberately includes host waits and per-step completion overhead.
            torch.accelerator.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            del out, counts, handle
        report = {
            "rank": rank,
            "gpu": torch.cuda.get_device_name(),
            "case": args.case,
            "do_expand": expanded,
            "do_cpu_sync": cpu_sync,
            "gemm_input_permutation": ("skipped" if expanded else "performed")
            if grouped is not None
            else "not_applicable",
            "input_tokens": tokens,
            "receive_capacity_rows": received_capacity,
            "wall_ms_per_step": statistics.mean(samples),
            "samples_ms": samples,
            "peak_allocated_gib": torch.accelerator.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.accelerator.max_memory_reserved() / 2**30,
            "correctness": "passed",
        }
        reports = [None] * world
        dist.all_gather_object(reports, report)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "config": {**vars(args), "output": str(args.output)},
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "deep_ep": deep_ep.__file__,
                "physical_domain": physical_domain,
                "gin_disabled": gin_disabled,
                "metric": "median per-step max-rank wall time, including CUDA sync",
                "wall_ms_per_step": statistics.median(
                    max(step) for step in zip(*(r["samples_ms"] for r in reports))
                ),
                "ranks": reports,
            }
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
