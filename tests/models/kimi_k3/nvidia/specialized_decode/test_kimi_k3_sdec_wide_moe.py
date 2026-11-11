# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3.1 specialized decode, TP8: the wide MoE layer (`moe.wide.WideMoE`: the
MoE layer of verify steps past the 64-row full step) against the stock
`KimiMoE.forward` of the same module, on 8 real ranks.

The module is the production one, built from a K3.1 checkpoint's config
(compressed-tensors MXFP4 experts, `process_weights_after_loading`: the
trtllm-gen layout) on random weights. Both paths route exactly on FP32 router logits
and differ only in accumulation order: the wide engine's sharded front, experts and
`Finalize` (80, 320 rows) or the stock TRTLLM-Gen experts (352 rows, `ENGINES`),
and the column-parallel tail. The sharded front computes its own FP32 router logits,
so a row may route differently at a near tie (asserted).

Launch: 8 ranks across two GB300 nodes with `srun -n8` (one pytest per rank),
`KIMI_K3_SD_TP8_INIT` a shared rendezvous file and `KIMI_K31_CONFIG` a K3.1
checkpoint directory (its config is read; no weights are loaded).
"""

import os

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)

INIT = os.environ.get("KIMI_K3_SD_TP8_INIT")
CONFIG = os.environ.get("KIMI_K31_CONFIG")
TP, HIDDEN = 8, 7168
# 80 and 320 rows: the wide engine (sharded front, listed experts) + Finalize; 352:
# the stock TRTLLM-Gen experts (``ENGINES``).
ROWS = (80, 320, 352)
# Listed vs grouped spec engine (bit-exact), over consecutive calls (list parities).
LISTED_ROWS = (80, 200)


def make_config(model_dir, max_tokens):
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        CUDAGraphMode,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    model = ModelConfig(
        model=model_dir,
        tokenizer=model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        seed=0,
        max_model_len=1 << 16,
    )
    config = VllmConfig(
        model_config=model,
        cache_config=CacheConfig(block_size=64, cache_dtype="fp8"),
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(
            max_num_seqs=128,
            max_num_batched_tokens=max(max_tokens, 8192),
            max_model_len=1 << 16,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig(),
        load_config=LoadConfig(load_format="dummy"),
        compilation_config=CompilationConfig(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
        ),
    )
    # 8 ranks over 2 nodes: set after validation (one node has 4 GPUs).
    parallel = config.parallel_config
    parallel.tensor_parallel_size = parallel.world_size = TP
    return config


def init_weights(module, seed):
    """BF16 fan-in scaled normals, packed FP4 uniform bytes, E8M0 scales 2^-5..2^-3,
    norms 1, a small correction bias."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    for name, p in module.named_parameters():
        if p.dtype in (torch.bfloat16, torch.float32):
            if "norm" in name:
                p.data.fill_(1.0)
            elif "bias" in name:
                p.data.copy_(torch.randn(p.shape, generator=g, device="cuda") * 1e-3)
            else:
                v = torch.randn(p.shape, generator=g, device="cuda")
                p.data.copy_(v * p.shape[-1] ** -0.5)
        elif p.dtype == torch.uint8 and "scale" in name:
            p.data.copy_(torch.randint(122, 125, p.shape, generator=g, device="cuda"))
        elif p.dtype == torch.uint8:
            p.data.copy_(torch.randint(0, 256, p.shape, generator=g, device="cuda"))
        else:
            raise TypeError(f"{name}: {p.dtype}")


def same_routing(module, x, ids, kind):
    """Rows whose top16 set equals stock's. The stock-engine path routes inside the
    stock experts (every row); the spec engine's front computes its own FP32 router
    logits (a different GEMM: accumulation order only), so a row may pick another
    expert only at a near tie of its 16th and 17th scores, which is asserted."""
    rows = x.shape[0]
    if kind == "trtllm":
        return torch.ones(rows, dtype=torch.bool, device=x.device)
    logits = module.gate(x)[0].float()
    scores = logits.sigmoid() + module.gate.e_score_correction_bias.float()
    top = scores.topk(17, dim=-1)
    ref = top.indices[:, :16].sort(-1).values
    same = (ids.long().sort(-1).values == ref).all(-1)
    gap = (top.values[:, 15] - top.values[:, 16]) / top.values[:, 15].abs()
    assert (gap[~same] < 1e-4).all(), gap[~same]
    assert int((~same).sum()) <= max(1, rows // 50), int((~same).sum())
    return same


def check(config, device):
    from vllm.distributed.parallel_state import graph_capture
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.kimi_k3.nvidia.model import KimiMoE
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import wide
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts_runtime import (  # noqa: E501
        PersistentExperts,
    )

    text = config.model_config.hf_text_config
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # as the model loader
    try:
        with torch.device(device):
            module = KimiMoE(
                text,
                config,
                quant_config=config.quant_config,
                prefix="model.layers.1.mlp",
                layer_idx=1,
            )
    finally:
        torch.set_default_dtype(dtype)
    init_weights(module, 101)
    process_weights_after_loading(module, config.model_config, device)
    beta, linear_beta = text.activation_situ_beta, text.activation_situ_linear_beta
    routed = module.experts.routed_experts
    w13, w2 = (w.view(torch.uint8) for w in (routed.w13_weight, routed.w2_weight))
    weights = w13, routed.w13_weight_scale, w2, routed.w2_weight_scale
    experts = PersistentExperts(*weights, beta=beta, linear_beta=linear_beta)
    batches = wide.wide_batches(ROWS)
    assert batches == ROWS
    state = wide.WideScratch(TP, device, batches)
    layer = wide.WideMoE(module, experts, state, beta, linear_beta)
    assert all(layer.serves(b) for b in ROWS) and not layer.serves(64)
    for rows in ROWS:
        g = torch.Generator(device="cuda").manual_seed(1000 + rows)
        x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
        with set_forward_context(None, config):
            stock = module(x).float().clone()
        # The wide providers run in the decode step's CUDA graph.
        graph = torch.cuda.CUDAGraph()
        with graph_capture(device) as ctx:
            with torch.cuda.stream(ctx.stream):
                layer.run(x)  # warm up (compiles, plans)
            torch.accelerator.synchronize()
            holder = {}
            with torch.cuda.graph(graph, stream=ctx.stream):
                holder["y"] = layer.run(x)
        for _ in range(2):  # replays are the step's
            graph.replay()
        torch.accelerator.synchronize()
        ours = holder["y"].float()
        assert torch.isfinite(ours).all()
        err = float((ours - stock).norm() / stock.norm())
        if torch.distributed.get_rank() == 0:
            print(f"WideMoE rows={rows} rel err vs KimiMoE.forward {err:.2e}")
        # BF16 rounding of the paths' different accumulation orders (expert rows,
        # top16 sum, all-reduce; the sharded front's FP32 router GEMM) only.
        assert err < 5e-3, (rows, err)
        # Rows routed as stock; a few elements may move further: the sharded front's
        # latent (FP32-out GEMM, one BF16 rounding) can differ from stock's BF16 GEMM
        # by an ulp (accumulation order), which flips some MXFP8 activation codes.
        same = same_routing(module, x, layer.state.ids[:rows], layer.plans[rows][0])
        a, b = ours[same], stock[same]
        far = ((a - b).abs() > 3e-2 + 3e-2 * b.abs()).float().mean().item()
        assert far < 1e-4, (rows, far)
        del graph
    check_listed(module, weights, device, beta, linear_beta)


def check_listed(module, weights, device, beta, linear_beta):
    """The spec engine's fronts over several consecutive calls (both routing-list
    parities and the count clearing): "listed" (`WideRoute` appends the routing
    lists and writes the plan; the experts build their windows from them) is
    bit-identical to "grouped" (routing, plan and unit-table kernels): pair order
    inside a unit does not change any token's FC1/FC2 column. "sharded" (one FP32-out
    GEMM over the router rows and latent down shard, multicast into the mailbox)
    differs from them only by the GEMM's accumulation order."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import wide
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe.experts_runtime import (  # noqa: E501
        PersistentExperts,
    )

    layers = {}
    saved = wide.ROUTING
    try:
        for kind in ("sharded", "listed", "grouped"):
            wide.ROUTING = f"65-1024:{kind}"
            experts = PersistentExperts(*weights, beta=beta, linear_beta=linear_beta)
            state = wide.WideScratch(TP, device, LISTED_ROWS)
            layers[kind] = wide.WideMoE(module, experts, state, beta, linear_beta)
            name = "spec" if kind == "grouped" else kind
            assert all(layers[kind].plans[b][0] == name for b in LISTED_ROWS)
    finally:
        wide.ROUTING = saved
    for call in range(5):
        rows = LISTED_ROWS[call % 2]
        g = torch.Generator(device="cuda").manual_seed(2000 + call)
        x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
        outs = {k: layer.run(x).float().clone() for k, layer in layers.items()}
        torch.accelerator.synchronize()
        assert torch.isfinite(outs["sharded"]).all()
        assert torch.equal(outs["listed"], outs["grouped"]), (call, rows)
        ref = outs["listed"]
        err = float((outs["sharded"] - ref).norm() / ref.norm())
        assert err < 5e-3, (call, rows, err)
    if torch.distributed.get_rank() == 0:
        print(f"listed == grouped (bit-exact), sharded rel err {err:.2e}, 5 calls")


def run_worker(rank, world, local_rank, init_method):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    config = make_config(CONFIG, max(ROWS))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
        init_workspace_manager(device)
        try:
            check(config, device)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    INIT is None or CONFIG is None or not os.path.isdir(CONFIG or ""),
    reason="needs an srun -n8 launch (KIMI_K3_SD_TP8_INIT) and a K3.1 config "
    "(KIMI_K31_CONFIG)",
)
def test_wide_moe_matches_kimi_moe_tp8_launched():
    """One rank of an `srun -n8` launch (TP8: 2 nodes x 4 GPUs)."""
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    assert world == TP, world
    local = int(os.environ["SLURM_LOCALID"])
    run_worker(rank, world, local, f"file://{INIT}")
