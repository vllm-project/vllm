# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 DSpark draft components against torch ports of the reference
implementation."""

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not (
    current_platform.is_device_capability_family(100)
    or current_platform.is_device_capability(90)
    or current_platform.is_device_capability_family(120)
):
    pytest.skip("Inkling2 requires Hopper or Blackwell", allow_module_level=True)

from vllm.models.inkling2.nvidia.dspark import (  # noqa: E402
    Inkling2MarkovHead,
)
from vllm.models.inkling2.nvidia.ops.attention import block_decode_attention  # noqa: E402
from vllm.models.inkling2.nvidia.ops.dspark import (  # noqa: E402
    add_softcap,
    markov_gate,
    mix_rmsnorm,
)

DEVICE = "cuda"


@pytest.mark.parametrize("rows,taps,width", [(1, 3, 128), (17, 8, 4608)])
@pytest.mark.parametrize("strided", [False, True])
def test_mix_rmsnorm_rounds_once(rows, taps, width, strided):
    g = torch.Generator().manual_seed(851)
    features = torch.randn(rows, taps, width * 2, generator=g).bfloat16().to(DEVICE)
    features = features[..., ::2] if strided else features[..., :width]
    alpha = torch.randn(taps, generator=g).softmax(-1).bfloat16().to(DEVICE)
    weight = (0.5 + torch.rand(width, generator=g)).bfloat16().to(DEVICE)
    mixed = (features.float() * alpha.float()[:, None]).sum(1)
    ref = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
    ref = (ref * weight.float()).bfloat16()
    out = mix_rmsnorm(features, alpha, weight, 1e-6)
    torch.testing.assert_close(out, ref, atol=0, rtol=1e-2)


def test_markov_head_fold_and_softcapped_logits(dist_init, default_vllm_config):
    """The rank-r latent goes through `hidden_out` and the LM head folded
    once in fp64; step logits are softcap(bf16(base + bias)); the recurrent
    gating rounds as the reference's torch ops."""
    g = torch.Generator().manual_seed(852)
    width, rank, vocab, cap = 128, 32, 256, 30.0

    def normal(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).bfloat16().to(DEVICE)

    config = SimpleNamespace(
        hidden_size=width, markov_rank=rank, padded_vocab_size=vocab, rms_norm_eps=1e-6
    )
    with torch.device(DEVICE):
        head = Inkling2MarkovHead(config).bfloat16()
    head.hidden_out.weight.data = normal(width, rank, scale=0.03)
    head.hidden_out.bias.data = normal(width, scale=0.03)
    lm_head = normal(vocab, width, scale=0.03)
    head.fold(lm_head)
    projection = lm_head.double() @ head.hidden_out.weight.double()
    assert torch.equal(head.fold_weight, projection.bfloat16())
    assert torch.equal(
        head.fold_bias, (lm_head.double() @ head.hidden_out.bias.double()).bfloat16()
    )
    # A refold after a weight update refills the same tensors (CUDA graphs).
    pointers = head.fold_weight.data_ptr(), head.fold_bias.data_ptr()
    lm_head.copy_(normal(vocab, width, scale=0.03))
    head.fold(lm_head)
    assert (head.fold_weight.data_ptr(), head.fold_bias.data_ptr()) == pointers
    projection = lm_head.double() @ head.hidden_out.weight.double()
    assert torch.equal(head.fold_weight, projection.bfloat16())

    base, bias = normal(3, vocab, scale=20.0), normal(3, vocab)
    ref = ((base + bias).float() / cap).tanh().mul(cap).bfloat16()
    torch.testing.assert_close(add_softcap(base, bias, cap), ref, atol=0, rtol=1e-2)

    # The fused gating matches the reference's bf16 torch ops bitwise.
    raw = torch.randn(512, 3 * rank, generator=g).mul(4).to(DEVICE)
    state = normal(512, rank)
    gate_raw, cand_raw, out_raw = raw.bfloat16().chunk(3, dim=-1)
    gate = torch.sigmoid(gate_raw)
    new_state, latent = markov_gate(raw, state)
    assert torch.equal(new_state, gate * state + (1.0 - gate) * torch.tanh(cand_raw))
    assert torch.equal(latent, torch.tanh(out_raw))


@pytest.mark.parametrize("is_local", [True, False])
@pytest.mark.parametrize("num_splits", [1, 3])
def test_block_attention_anchored_window(is_local, num_splits):
    """Every query of a block sees the whole block and, for local layers, the
    `window` positions before the block; a padding request (no keys) comes
    out zero."""
    g = torch.Generator().manual_seed(854)
    block, window, num_heads, num_kv_heads, head_dim, page = 8, 64, 6, 2, 128, 16
    context_lens, num_pages = [5, 70, 150, 300], 20

    def rand(*shape):
        return torch.randn(*shape, generator=g).bfloat16().to(DEVICE)

    num_reqs = len(context_lens) + 1
    # The model's [blocks, block_size, kv_heads, (K | V)] cache layout.
    kv = rand(num_reqs * num_pages, page, num_kv_heads, 2 * head_dim)
    key_cache, value_cache = kv.split(head_dim, dim=-1)
    block_table = torch.arange(num_reqs * num_pages, dtype=torch.int32, device=DEVICE)
    seq_lens = [n + block for n in context_lens] + [0]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    q = rand(block * num_reqs, num_heads, head_dim)
    out = torch.full((q.shape[0], num_heads * head_dim), float("nan"), device=DEVICE)
    out = out.bfloat16()
    block_decode_attention(
        q,
        key_cache,
        value_cache,
        out,
        block_table=block_table.view(num_reqs, num_pages),
        cache_seqlens=seq_lens,
        cu_seqlens_q=torch.arange(num_reqs + 1, device=DEVICE).int() * block,
        block=block,
        window=window + block if is_local else 0,
        num_splits=num_splits,
    )
    out = out.view(-1, num_heads, head_dim)

    group = num_heads // num_kv_heads
    for i, n in enumerate(context_lens):
        start = max(0, n - window) if is_local else 0
        flat = slice(i * num_pages * page + start, i * num_pages * page + n + block)
        keys = key_cache.flatten(0, 1)[flat].float().repeat_interleave(group, 1)
        vals = value_cache.flatten(0, 1)[flat].float().repeat_interleave(group, 1)
        rows = slice(i * block, (i + 1) * block)
        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), keys) * head_dim**-0.5
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals)
        torch.testing.assert_close(out[rows].float(), ref, atol=2e-2, rtol=2e-2)
    assert (out[len(context_lens) * block :] == 0).all()


def _all_reduce_norm_worker(rank: int, world: int, port: int, queue) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import (
        get_tp_group,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )
    from vllm.models.inkling2.nvidia.dspark import _RMSNorm, all_reduce_add_rmsnorm

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(world, 1, rank, str(port), rank)
    comm = get_tp_group().device_communicator
    assert isinstance(comm, CudaCommunicator) and comm.fi_ar_comm is not None
    dim = 4608
    norm = _RMSNorm(dim, 1e-6).to(DEVICE, torch.bfloat16)
    failures, stats = [], {}
    # One shot, two shots, and just above the workspace (unfused). The
    # workspace size depends on the GPU and the world size.
    max_rows = comm.fi_ar_comm.max_workspace_size // (dim * 2)
    for rows in (8, 64, 256, max_rows + 1):
        g = torch.Generator().manual_seed(rows)  # shared by all ranks
        residual = torch.randn(rows, dim, generator=g).bfloat16().to(DEVICE)
        norm.weight.copy_(0.5 + torch.rand(dim, generator=g))
        g.manual_seed(rows + 1 + rank)
        x = torch.randn(rows, dim, generator=g).bfloat16().to(DEVICE)
        fused_res, ref_res = residual.clone(), residual.clone()
        out = all_reduce_add_rmsnorm(x.clone(), fused_res, norm)
        reduced = tensor_model_parallel_all_reduce(x)
        exact = reduced.double() + residual.double()
        ref = norm(reduced, ref_res)
        exact = exact * torch.rsqrt(exact.square().mean(-1, keepdim=True) + 1e-6)
        exact = exact * norm.weight.double()
        ulp = torch.finfo(torch.bfloat16).eps * exact.abs().clamp_min(1e-30)
        stats[rows] = [
            ((o.double() - exact).abs() / ulp).mean().item() for o in (out, ref)
        ]
        gathered = tensor_model_parallel_all_gather(out[None], 0)
        checks = {
            "fused iff it fits": comm.fi_ar_comm.should_use_fi_ar(x)
            == (rows <= max_rows),
            "same on all ranks": all(torch.equal(o, out) for o in gathered),
            "residual": torch.equal(fused_res, ref_res),
            # The norm reads the bf16-rounded residual.
            "norm": torch.allclose(out.float(), ref.float(), atol=0, rtol=2**-7),
        }
        failures += [f"rows={rows}: {k}" for k, ok in checks.items() if not ok]
    queue.put((rank, failures, stats))


@pytest.mark.parametrize("world", [2, 4])
def test_all_reduce_add_rmsnorm_matches_unfused(world):
    """The fused all-reduce + residual add + RMSNorm keeps the unfused bf16
    residual bits and gives the same norm output on every rank."""
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    queue = mp.get_context("spawn").Queue()
    mp.spawn(_all_reduce_norm_worker, args=(world, 29620 + world, queue), nprocs=world)
    results = [queue.get() for _ in range(world)]
    assert all(not failures for _, failures, _ in results), results
