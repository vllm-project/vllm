"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.models.mixtral import MixtralMoE
import os

def torch_moe(a, w1, w2, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)


@pytest.mark.parametrize("m", [512, 222, 33, 1])
@pytest.mark.parametrize("n", [2048, 256, 1024])
@pytest.mark.parametrize("k", [128, 511, 1024])
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
    torch_output = torch_moe(a, w1, w2, score, topk)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32] # , torch.float16, torch.bfloat16]
)
@torch.inference_mode()
def test_mixtral_moe(dtype: torch.dtype):
    """Make sure our Mixtral MoE implementation agrees with the one from
    huggingface."""

    # Instantiate our and huggingface's MoE blocks
    config = MixtralConfig()
    hf_moe = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    vllm_moe = MixtralMoE(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        params_dtype=dtype,
        tp_size=1,
    ).cuda()

    # Load the weights
    vllm_moe.gate.weight.data[:] = hf_moe.gate.weight.data
    for i in range(config.num_local_experts):
        weights = (hf_moe.experts[i].w1.weight.data,
                   hf_moe.experts[i].w3.weight.data)
        vllm_moe.w13_weight[i][:] = torch.cat(weights, dim=0)
        vllm_moe.w2_weight[i][:] = hf_moe.experts[i].w2.weight.data

    # Generate input batch of dimensions [batch_size, seq_len, hidden_dim]
    hf_inputs = torch.randn((1, 64, config.hidden_size)).to(dtype).to("cuda")
    # vLLM uses 1D query [num_tokens, hidden_dim]
    vllm_inputs = hf_inputs.flatten(0, 1)

    # Run forward passes for both MoE blocks
    hf_states, _ = hf_moe.forward(hf_inputs)
    vllm_states = vllm_moe.forward(vllm_inputs)

    mixtral_moe_tol = {
        torch.float32: 1e-3,
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
    }

    assert torch.allclose(hf_states.flatten(0, 1),
                          vllm_states,
                          rtol=mixtral_moe_tol[dtype],
                          atol=mixtral_moe_tol[dtype])


import triton


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["bsz"],
            x_vals=[i for i in range(4, 8, 2)],
            line_arg="provider",
            line_vals=["vllm", "hf"],
            line_names=["vLLM", "HF"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="time (ms)",
            plot_name=f"moe-fp32-speed-benchmark",
            args={"seq_len": 4096, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.float32},
        ),
        triton.testing.Benchmark(
            x_names=["bsz"],
            x_vals=[i for i in range(4, 8, 2)],
            line_arg="provider",
            line_vals=["vllm", "hf"],
            line_names=["vLLM", "HF"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="time (ms)",
            plot_name=f"moe-bf16-speed-benchmark",
            args={"seq_len": 4096, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.bfloat16},
        ),
        triton.testing.Benchmark(
            x_names=["bsz"],
            x_vals=[i for i in range(4, 8, 2)],
            line_arg="provider",
            line_vals=["vllm", "hf"],
            line_names=["vLLM", "HF"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="time (ms)",
            plot_name=f"moe-fp16-speed-benchmark",
            args={"seq_len": 4096, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.float16},
        ),
    ]
)
def bench_speed_moe(bsz, seq_len, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, provider, dtype):
    print(f"Running: bsz={bsz}, seq_len={seq_len}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_local_experts={num_local_experts}, num_experts_per_tok={num_experts_per_tok}, provider={provider}, dtype={dtype}")

    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    if provider == "vllm":
        moe_block = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=dtype,
            tp_size=1,
        ).to(dtype).to("cuda")
    elif provider == "hf":
        moe_block = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    else:
        raise ValueError(f"Invalid provider: {provider} for MoE block")

    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)
    dy = torch.randn_like(x)

    quantiles = [0.5, 0.2, 0.8]

    def fwd():
        if provider == "vllm":
            moe_block(x.view(bsz*seq_len, -1))
        elif provider == "hf":
            moe_block(x)
        else:
            raise ValueError(f"Invalid provider: {provider} for MoE block")

    ms, min_ms, max_ms = triton.testing.do_bench(
        fwd, quantiles=quantiles, grad_to_none=[x], warmup=2, rep=4
    )

    return ms, max_ms, min_ms


@pytest.mark.speed
def test_bench_speed_moe_wrapper():
    output_dir = "./moe_speed"
    os.makedirs(output_dir, exist_ok=True)
    bench_speed_moe.run(save_path=output_dir, print_data=True)

def _test_memory_once(func):
    torch.cuda.memory.reset_peak_memory_stats()

    func()

    mem = torch.cuda.max_memory_allocated()

    torch.cuda.memory._record_memory_history(enabled=None)
    return mem


def _test_memory(func, _iter):
    total_mem = []

    for _ in range(_iter):
        mem = _test_memory_once(func)
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bsz"],
        x_vals=[i for i in range(4, 8, 2)],
        line_arg="provider",
        line_vals=["vllm", "hf"],
        line_names=["vLLM", "HF"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Memory (MB)",
        plot_name=f"moe-memory-benchmark",
        args={"seq_len": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.bfloat16},
    ),
)
def bench_memory_moe(bsz, seq_len, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, provider, dtype):
    print(f"Running: bsz={bsz}, seq_len={seq_len}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_local_experts={num_local_experts}, num_experts_per_tok={num_experts_per_tok}, provider={provider}, dtype={dtype}")

    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    if provider == "vllm":
        moe_block = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            params_dtype=dtype,
            tp_size=1,
        ).to(dtype).to("cuda")
    elif provider == "hf":
        moe_block = MixtralSparseMoeBlock(config).to(dtype).to("cuda")
    else:
        raise ValueError(f"Invalid provider: {provider} for MoE block")


    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)
    dx = torch.randn_like(x)

    def fwd():
        if provider == "vllm":
            moe_block(x.view(bsz*seq_len, -1))
        elif provider == "hf":
            moe_block(x)
        else:
            raise ValueError(f"Invalid provider: {provider} for MoE block")

    mem = _test_memory(fwd, _iter=2)
    return mem / 2**20


@pytest.mark.memory
def test_bench_memory_moe_wrapper():
    output_dir = "./moe_memory"
    os.makedirs(output_dir, exist_ok=True)
    bench_memory_moe.run(save_path=output_dir, print_data=True)