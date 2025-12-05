# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.cuda.profiler as profiler

from vllm.model_executor.layers.fused_moe.gpt_oss_fused_router import (
    gpt_oss_custom_routing_function,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def profile_run():
    torch.manual_seed(0)
    device = "cuda"

    test_cases = [
        {
            "name": "GPTOSS20B",
            "desc": "gpt oss 20b prefill",
            "M": 4096,
            "N": 32,
            "topk": 4,
        },
    ]

    def run_origin(hidden_states, router_logits, topk):
        _ = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=topk,
            use_grouped_topk=False,
            renormalize=True,
            custom_routing_function=None,
        )

    def run_triton(hidden_states, router_logits, topk):
        _ = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=topk,
            use_grouped_topk=False,
            renormalize=True,
            custom_routing_function=gpt_oss_custom_routing_function,
        )

    for case in test_cases:
        M, N, topk = case["M"], case["N"], case["topk"]
        hidden_states = torch.randn(M, 4096, device=device, dtype=torch.float16)
        router_logits = torch.randn(M, N, device=device, dtype=torch.float16)

        for i in range(20):
            print(f"Starting Global Warmups, Iter {i}")
            run_origin(hidden_states, router_logits, topk)
            run_triton(hidden_states, router_logits, topk)

    torch.cuda.synchronize()
    print("Warmup Completed. All kernels are compiled.")

    profiler.start()

    for case in test_cases:
        M, N, topk = case["M"], case["N"], case["topk"]
        hidden_states = torch.randn(M, 4096, device=device, dtype=torch.float16)
        router_logits = torch.randn(M, N, device=device, dtype=torch.float16)
        run_origin(hidden_states, router_logits, topk)
        run_triton(hidden_states, router_logits, topk)
        torch.cuda.synchronize()

    profiler.stop()
    print("Benchmark finished.")


if __name__ == "__main__":
    profile_run()
