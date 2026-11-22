# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Does vLLM fuse the release DSpark draft's all-reduce + RMSNorm pairs (the preview's
`DAR` route) on its own? Two ranks compile the release draft's decoder layer
(`DFlashQwen3DecoderLayer.forward`: input norm, o_proj all-reduce, post-attention
norm, MLP all-reduce; attention left out, it is a graph split point) and the final
norm with vLLM's post-grad pass manager configured as the recipe's server is (ROCm,
AITER on, VLLM_COMPILE at the default -O2, TP > 1, hidden 7168), for the stock
`Qwen2MLP` and for the amd_kernels MLP (`akr_draft.AkrDenseMlp`). Every pair must
become AITER's fused all-reduce + RMSNorm and match the eager output.

Run (spawns 2 ranks, needs 2 GPUs):
    python tests/models/kimi_k3/akr/draft_ar_fusion_ranks2.py
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp
from torch import nn

HIDDEN, INTER, ATTN_SHARD, LAYERS, ROWS = 7168, 3584, 768, 2, 32


class _OProj(nn.Module):
    """The attention block reduced to its o_proj (RowParallelLinear, all-reduce
    inside), on this rank's first `ATTN_SHARD` input columns."""

    def __init__(self, world: int):
        super().__init__()
        from vllm.model_executor.layers.linear import RowParallelLinear

        self.o_proj = RowParallelLinear(ATTN_SHARD * world, HIDDEN, bias=False)

    def forward(self, positions, hidden_states):
        out, _ = self.o_proj(hidden_states[:, :ATTN_SHARD].contiguous())
        return out


class _Draft(nn.Module):
    def __init__(self, world: int):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.models.qwen2 import Qwen2MLP

        self.layers = nn.ModuleList()
        for _ in range(LAYERS):
            layer = nn.Module()
            layer.input_layernorm = RMSNorm(HIDDEN, eps=1e-5)
            layer.post_attention_layernorm = RMSNorm(HIDDEN, eps=1e-5)
            layer.self_attn = _OProj(world)
            layer.mlp = Qwen2MLP(HIDDEN, INTER, "silu")
            self.layers.append(layer)
        self.norm = RMSNorm(HIDDEN, eps=1e-5)

    def forward(self, positions, hidden_states):
        from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer

        residual = None
        for layer in self.layers:
            hidden_states, residual = DFlashQwen3DecoderLayer.forward(
                layer, positions, hidden_states, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _vllm_config(world: int, rms_custom_op: bool):
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        VllmConfig,
    )

    custom_ops = ["+fused_rms_norm_gated"] + (["+rms_norm"] if rms_custom_op else [])
    return VllmConfig(
        model_config=ModelConfig(
            model="facebook/opt-125m",
            dtype=torch.bfloat16,
            hf_overrides={"hidden_size": HIDDEN},
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=world),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=custom_ops
        ),
        device_config=DeviceConfig(device=torch.device("cuda")),
    )


def _check(rank, world, rms_custom_op: bool, akr_mlp: bool) -> str:
    from tests.compile.backend import TestBackend
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.compilation.passes.pass_manager import PostGradPassManager
    from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_model_parallel,
        initialize_model_parallel,
    )

    cfg = _vllm_config(world, rms_custom_op)
    assert cfg.compilation_config.pass_config.fuse_allreduce_rms, (
        "the -O2 default leaves the all-reduce + RMSNorm fusion off"
    )
    with set_current_vllm_config(cfg):
        initialize_model_parallel(tensor_model_parallel_size=world)
        assert rocm_aiter_ops.get_aiter_allreduce() is not None, "no AITER custom AR"
        torch.manual_seed(5)  # replicated parts equal on every rank
        model = _Draft(world).to(torch.bfloat16)
        for p in model.parameters():
            p.data.normal_(0, 0.02)
        for layer in model.layers:
            for n in (layer.input_layernorm, layer.post_attention_layernorm):
                n.weight.data.uniform_(0.5, 1.5)
        if akr_mlp:
            from vllm.models.kimi_k3.amd import akr_draft

            spec = SimpleNamespace(
                device=torch.device("cuda"), max_num_reqs=8, num_speculative_steps=4
            )
            dm = SimpleNamespace(model=model)
            assert akr_draft._mlp_reason(spec, dm) is None
            assert akr_draft._install_mlp(spec, dm) == LAYERS
            assert all(lyr.mlp.all_reduce for lyr in model.layers)
        pm = PostGradPassManager()
        pm.configure(cfg)

        def post_grad(graph):
            pm(graph)
            VllmInductorPass.dump_prefix = 0  # the manager ends it; TestBackend counts

        backend = TestBackend(post_grad)
        positions = torch.arange(ROWS)
        x = torch.randn(ROWS, HIDDEN).to(torch.bfloat16)
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        out = compiled(positions, x)
        ref = model(positions, x)
        fused = backend.op_count(rocm_aiter_ops.get_fused_allreduce_rmsnorm_op())
        left = backend.op_count(torch.ops.vllm.all_reduce.default)
        rel = float((out.float() - ref.float()).norm() / ref.float().norm())
        destroy_model_parallel()
    want = 2 * LAYERS  # o_proj + MLP all-reduce of every layer
    assert fused == want and left == 0, (rank, fused, left)
    assert rel < 2e-2, (rank, rel)
    return (
        f"rms_norm custom op {rms_custom_op}, akr MLP {akr_mlp}: "
        f"{fused} fused AR + RMSNorm, {left} plain all-reduces, rel {rel:.2e}"
    )


def worker(rank, world, port, cases):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        VLLM_ROCM_USE_AITER="1",
        VLLM_ROCM_USE_AITER_CUSTOM_AR="1",
    )
    torch.accelerator.set_device_index(rank)
    torch.set_default_device(f"cuda:{rank}")
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment

    rocm_aiter_ops.refresh_env_variables()
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment()
    for rms_custom_op, akr_mlp in cases:
        msg = _check(rank, world, rms_custom_op, akr_mlp)
        if rank == 0:
            print(msg, flush=True)


if __name__ == "__main__":
    want_akr = "--no-akr" not in sys.argv
    cases = [
        (c, a) for c in (False, True) for a in ((False, True) if want_akr else (False,))
    ]
    port = 29600 + os.getpid() % 300
    mp.spawn(worker, args=(2, port, cases), nprocs=2)
    print("PASS", cases)
