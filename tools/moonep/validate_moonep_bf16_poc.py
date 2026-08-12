# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Distributed BF16 correctness validation for the MoonEP PoC integration.

Runs MoonEPPrepareAndFinalize + the reference segment runner against a dense
per-token reference MoE with replicated global expert weights.

Requires an NVLink symmetric-memory capable node and the `moonep` package:

    torchrun --nproc_per_node=4 tools/moonep/validate_moonep_bf16_poc.py
"""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


def reference_moe(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """Dense per-token reference with replicated global weights."""
    intermediate = w13.shape[1] // 2
    out = torch.zeros_like(hidden)
    for t in range(hidden.shape[0]):
        x = hidden[t]
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            gate = F.linear(x, w13[e, :intermediate])
            up = F.linear(x, w13[e, intermediate:])
            y = F.linear(F.silu(gate) * up, w2[e])
            out[t] += topk_weights[t, k].to(y.dtype) * y
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=128, help="S")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--num-prefetch-slots", type=int, default=4)
    parser.add_argument("--token-padding", type=int, default=128)
    parser.add_argument("--num-sms", type=int, default=32)
    parser.add_argument("--skew", type=float, default=2.0,
                        help="router logit scale; higher = more imbalance")
    parser.add_argument("--tolerance", type=float, default=2e-2)
    args = parser.parse_args()

    from moonep import Buffer  # noqa: PLC0415

    from vllm.model_executor.layers.fused_moe.prepare_finalize.moonep import (  # noqa: PLC0415, E501
        MoonEPPrepareAndFinalize,
        make_moonep_weight_layout,
        run_moonep_bf16_reference_experts,
    )

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    assert args.num_experts % world == 0, "num_experts must divide EP size"

    # Identical replicated global expert weights on every rank.
    torch.manual_seed(1234)
    w13 = torch.randn(
        args.num_experts, 2 * args.intermediate_size, args.hidden_size,
        dtype=torch.bfloat16, device=device,
    ) / args.hidden_size**0.5
    w2 = torch.randn(
        args.num_experts, args.hidden_size, args.intermediate_size,
        dtype=torch.bfloat16, device=device,
    ) / args.intermediate_size**0.5
    weight_layout = make_moonep_weight_layout(
        w13, w2, num_prefetch_slots=args.num_prefetch_slots
    )

    # Per-rank tokens and (skewed) routing.
    torch.manual_seed(5678 + rank)
    hidden = torch.randn(
        args.num_tokens, args.hidden_size, dtype=torch.bfloat16, device=device
    )
    logits = args.skew * torch.randn(
        args.num_tokens, args.num_experts, dtype=torch.float32, device=device
    )
    topk_weights, topk_ids = torch.topk(logits, args.topk, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    buffer = Buffer(
        S=args.capacity,
        H=args.hidden_size,
        K=args.topk,
        E=args.num_experts,
        num_ep_ranks=world,
        num_sms=args.num_sms,
        token_padding=args.token_padding,
        B=args.num_prefetch_slots,
        group=dist.group.WORLD,
    )

    pf = MoonEPPrepareAndFinalize(
        buffer=buffer,
        max_tokens_per_rank=args.capacity,
        num_dispatchers=world,
        num_global_experts=args.num_experts,
        weight_layout=weight_layout,
    )

    no_quant = SimpleNamespace(quant_dtype=None)
    hidden_nvsh, _, expert_tokens_meta, _, route_weights_nvs = pf.prepare(
        hidden,
        topk_weights,
        topk_ids,
        num_experts=args.num_experts,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=no_quant,
    )
    expert_output_nvsh = run_moonep_bf16_reference_experts(
        hidden_nvsh,
        route_weights_nvs,
        pf.cu_seqlens,
        weight_layout,
    )
    output = torch.empty_like(hidden)
    pf.finalize(
        output,
        expert_output_nvsh,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input=False,
        weight_and_reduce_impl=None,
    )

    expected = reference_moe(hidden, topk_ids, topk_weights, w13, w2)
    max_abs_err = (output.float() - expected.float()).abs().max()
    denom = expected.float().abs().max().clamp_min(1e-6)
    rel_err = max_abs_err / denom
    ok = torch.tensor(
        [float(rel_err.item() <= args.tolerance)], device=device
    )
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    global_ok = bool(ok.item() > 0)

    print(
        f"[rank {rank}] max_abs_err={max_abs_err.item():.3e} "
        f"rel_err={rel_err.item():.3e} global_ok={global_ok}"
    )

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and not global_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
