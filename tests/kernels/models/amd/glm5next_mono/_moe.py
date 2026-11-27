# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flash MonoKernel's FFN stages against vLLM's own `Glm5NextMoE`.

vLLM builds the layer from the model config with its Quark MXFP4 quantization; the
test fills the parameters in their loaded layout with random values, runs
`process_weights_after_loading` and hands the kernel the views
`layer_weights` takes of the result, as in serving.

Both the kernel and vLLM (its mHC ops, `Glm5NextMoE.forward`) run on the kernel's
own input to each stage and are measured against the fp64 reference of
`reference.py` on the raw MXFP4 shards. vLLM quantizes expert inputs to MXFP4 and
the kernel to MXFP8, so the kernel must be well under vLLM's MoE error, and
within 1% of the fp64 MoE with both expert GEMMs' inputs rounded to MXFP8 as it
rounds them (what is left is its accumulation order)."""

from types import SimpleNamespace

import ray
import torch

from tests.utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
)

from ._ffn import (
    POLL_TIMEOUT_US,
    _ffn_input,
    _fp64_pre,
    _routed,
    _teardown,
    _vllm_pre,
    _vllm_seam,
)
from .reference import Bound, Numerics, dequant, mhc_post
from .reference import moe as fp64_moe

MODEL = "amd/GLM-5.3-Flash-Quark-MXFP4"
PREFIX = "language_model.model.layers.7.mlp"


def _random_moe(moe, gd, gr) -> None:
    """Random MXFP4 experts (nibbles, E8M0 scales) in the loaded layout, this rank's
    shards from `gd`; the replicated router from `gr`, alike on every rank."""
    u8 = torch.uint8
    e = moe.experts.routed_experts
    for name, lo, hi in (("weight", 0, 256), ("weight_scale", 118, 123)):
        for w in ("w13", "w2"):
            t = getattr(e, f"{w}_{name}").data.view(u8)
            t.copy_(
                torch.randint(lo, hi, t.shape, generator=gd, device=gd.device, dtype=u8)
            )
    if moe.shared_experts is not None:
        for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
            for t, lo, hi in ((lin.weight, 0, 256), (lin.weight_scale, 118, 123)):
                v = t.data.view(u8)
                v.copy_(
                    torch.randint(
                        lo, hi, v.shape, generator=gd, device=gd.device, dtype=u8
                    )
                )
    gate = moe.gate.weight
    gate.data.copy_(
        (torch.randn(gate.shape, generator=gr, device=gr.device) * 0.02).to(gate.dtype)
    )
    bias = moe.gate.e_score_correction_bias
    bias.data.copy_(
        (torch.rand(bias.shape, generator=gr, device=gr.device) - 0.5) * 0.2
    )


def _worker(widths: tuple[int, ...], bounds: dict[str, Bound]):
    @ray.remote(num_gpus=1, max_calls=1)
    def run(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.v1.worker.workspace import init_workspace_manager

        with monkeypatch.context() as m:
            m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
            # the serving flags: AITER MoE with the shared expert fused in
            for k in (
                "VLLM_ROCM_USE_AITER",
                "VLLM_ROCM_USE_AITER_MOE",
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
            ):
                m.setenv(k, "1")
            rocm_aiter_ops.refresh_env_variables()
            device = torch.device(f"cuda:{rank}")
            torch.accelerator.set_device_index(device)
            init_test_distributed_environment(
                tp_size, pp_size, rank, distributed_init_port
            )
            ensure_model_parallel_initialized(tp_size, pp_size)
            init_workspace_manager(device)
            _check(rank, tp_size, device, widths, bounds)
            _teardown()

    return run


def _check(
    rank: int, tp: int, device, widths: tuple[int, ...], bounds: dict[str, Bound]
) -> None:
    import torch.distributed as dist

    from vllm.config import set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        GLOBAL_INTER,
        HC_MULT,
        HIDDEN,
        N_EXPERTS,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
        FlashFfnOp,
        FlashFfnRuntime,
    )
    from vllm.models.glm5next.amd.specialized_decode.runtime.weights import (
        layer_weights,
    )
    from vllm.models.glm5next.common.model import Glm5NextMoE
    from vllm.utils.torch_utils import set_default_torch_dtype

    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tp,
        load_format="dummy",
        trust_remote_code=True,
        max_model_len=8192,
        kv_cache_dtype="bfloat16",
    ).create_engine_config()
    cfg = vc.model_config.hf_text_config
    with set_current_vllm_config(vc), set_default_torch_dtype(torch.bfloat16), device:
        moe = Glm5NextMoE(cfg, vc.parallel_config, vc.quant_config, prefix=PREFIX)
    _random_moe(
        moe,
        torch.Generator(device=device).manual_seed(100 + rank),
        torch.Generator(device=device).manual_seed(99),
    )
    # the raw shards, for the exact reference, before post-load processing
    e = moe.experts.routed_experts
    w13, s13, w2, s2 = (
        getattr(e, n).data.view(torch.uint8).clone()
        for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    )
    sh = None
    if moe.shared_experts is not None:
        sh = tuple(
            t.data.view(torch.uint8).clone()
            for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj)
            for t in (lin.weight, lin.weight_scale)
        )

    def expert(e: int):
        """This rank's fp64 shard of expert e; the shared one at N_EXPERTS."""
        q = sh if e == N_EXPERTS and sh is not None else (w13[e], s13[e], w2[e], s2[e])
        return dequant(q[0], q[1]), dequant(q[2], q[3])

    with set_current_vllm_config(vc):
        process_weights_after_loading(moe, vc.model_config, device)

    g = torch.Generator(device=device).manual_seed(7)
    nmix = HC_MULT * (2 + HC_MULT)
    lay = SimpleNamespace(
        mlp=moe,
        hc_ffn_fn=torch.randn(nmix, HC_MULT * HIDDEN, generator=g, device=device)
        * 0.01,
        hc_ffn_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_ffn_base=torch.randn(nmix, generator=g, device=device) * 0.1,
        post_attention_layernorm=SimpleNamespace(
            weight=(1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device))
            .bfloat16()
            .contiguous()
        ),
    )
    inter = GLOBAL_INTER // tp
    weights = layer_weights(lay, inter)  # refuses any other layout
    seam_lay = SimpleNamespace(
        fn=lay.hc_ffn_fn,
        scale=lay.hc_ffn_scale,
        base=lay.hc_ffn_base,
        gamma=lay.post_attention_layernorm.weight,
    )
    group = get_tp_group().cpu_group
    num = Numerics(f"Glm5NextMoE layer, TP{tp}", bounds, rank)
    for S in widths:
        rt = FlashFfnRuntime(S, rank, tp, group, inter, 1, POLL_TIMEOUT_US, False)
        op = FlashFfnOp(rt, weights, 0)
        parts = [
            torch.randn(S, HIDDEN, generator=g, device=device).bfloat16()
            for _ in range(tp)
        ]
        x_attn = torch.stack([p.float() for p in parts]).sum(0).bfloat16()
        res = torch.randn(S, HC_MULT, HIDDEN, generator=g, device=device).bfloat16()
        post = torch.rand(S, HC_MULT, 1, generator=g, device=device) * 2
        comb = torch.softmax(
            torch.randn(S, HC_MULT, HC_MULT, generator=g, device=device), -1
        )
        dist.barrier(group)
        x, r, p, c = (t.clone() for t in op.launch(parts[rank], res, post, comb))
        torch.accelerator.synchronize()
        assert not rt.words.any(), f"S={S}: expired waits {rt.words.tolist()}"

        # the seam: R' on the attention partials, the pre on the kernel's own R'
        v_r = _vllm_seam(seam_lay, x_attn, res, post, comb)[0]
        x64 = torch.stack([p.double() for p in parts]).sum(0)
        num.add("residual", S, r, mhc_post(x64, res, post, comb), v_r)
        p64, c64, in64 = _fp64_pre(seam_lay, r)
        v_p, v_c, v_x = _vllm_pre(seam_lay, r)
        k_in = _ffn_input(rt, S)
        num.add("post mix", S, p, p64, v_p)
        num.add("comb mix", S, c, c64, v_c)
        num.add("FFN input", S, k_in, in64, v_x)

        # the MoE: vLLM's Glm5NextMoE and the fp64 reference on the kernel's own
        # FFN input; each rank's reference is its intermediate shard's partial
        with set_current_vllm_config(vc), set_forward_context(None, vc):
            y = moe(k_in.clone())
        if not getattr(moe.experts, "reduce_results", True):
            dist.all_reduce(y)
        gate = moe.gate
        exact, mx8, order = fp64_moe(
            k_in, gate.weight, gate.e_score_correction_bias, expert, N_EXPERTS
        )
        dist.all_reduce(exact)
        dist.all_reduce(mx8)
        # routing: the kernel's top 8 is exactly the fp64 router's
        assert torch.equal(_routed(rt, S), order.int().sort(-1).values), (
            f"S={S}: the kernel routes other experts than the fp64 router"
        )
        num.add("MoE", S, x, exact, y)
        num.add("MoE (MXFP8 inputs)", S, x, mx8)
        dist.barrier(group)
        rt.close()
    num.check()
