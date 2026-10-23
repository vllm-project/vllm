# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flash MonoKernel's FFN stages against vLLM's own ``Glm5NextMoE``.

vLLM builds the layer from the model config with its Quark MXFP4 quantization; the
test fills the parameters in their loaded layout with random values, runs
``process_weights_after_loading`` and hands the kernel the views
``layer_weights`` takes of the result, as in serving. The seam is checked against
vLLM's mHC op, the MoE against ``Glm5NextMoE.forward`` on the kernel's own FFN
input, and both against an fp32 MoE on the dequantized weights and unquantized
inputs. vLLM quantizes expert inputs to MXFP4 and the kernel to MXFP8, so the gap
between them is vLLM's own error (~0.2 relative on random weights): the test checks
that, that the kernel is the closer of the two to the exact result, and that both
route alike."""

from types import SimpleNamespace

import ray
import torch

from tests.utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
)

from ._ffn import _dequant, _rel, _vllm_seam

MODEL = "amd/GLM-5.3-Flash-Quark-MXFP4"
PREFIX = "language_model.model.layers.7.mlp"
ROWS = (2, 4, 8)


def _random_moe(moe, gd, gr) -> None:
    """Random MXFP4 experts (nibbles, E8M0 scales) in the loaded layout, this rank's
    shards from ``gd``; the replicated router from ``gr``, alike on every rank."""
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


def _exact_moe(x, gate, bias, w13, s13, w2, s2, shared):
    """``shared``: the shared expert's slot in the routed tensors, or its own
    (gate_up, scale, down, scale) MXFP4 shards."""
    """fp32 MoE on dequantized MXFP4 weights and unquantized inputs, vLLM's
    routing (sigmoid + bias, top-8, renormalized, x2.5) and SwiGLU clamp."""
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        ROUTE_SCALE,
        SWIGLU_LIMIT,
        TOP_K,
    )

    scores = torch.sigmoid(x.float() @ gate.float().T)
    order = torch.argsort(-(scores + bias), dim=-1, stable=True)[:, :TOP_K]
    picked = torch.gather(scores, 1, order)
    weights = picked / picked.sum(-1, keepdim=True) * ROUTE_SCALE
    out = torch.zeros_like(x, dtype=torch.float32)

    def expert(e):
        if isinstance(e, tuple):
            return e
        return w13[e], s13[e], w2[e], s2[e]

    for s in range(x.shape[0]):
        for e, wt in zip([shared] + order[s].tolist(), [1.0] + weights[s].tolist()):
            a13, b13, a2, b2 = expert(e)
            gu = _dequant(a13, b13) @ x[s].float()
            n = gu.shape[0] // 2
            g = gu[:n].clamp(max=SWIGLU_LIMIT)
            u = gu[n:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            out[s] += wt * (_dequant(a2, b2) @ (torch.nn.functional.silu(g) * u))
    return out


@ray.remote(num_gpus=1, max_calls=1)
def _worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
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
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)
        _check(rank, tp_size, device)


def _check(rank: int, tp: int, device) -> None:
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
        MOE_SLOTS,
        N_EXPERTS,
        TOP_K,
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
    shared = N_EXPERTS
    if moe.shared_experts is not None:
        sh = moe.shared_experts
        shared = tuple(
            t.data.view(torch.uint8).clone()
            for lin in (sh.gate_up_proj, sh.down_proj)
            for t in (lin.weight, lin.weight_scale)
        )
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
    for S in ROWS:
        rt = FlashFfnRuntime(S, rank, tp, group, inter, 1, 20_000_000, False)
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

        # seam: vLLM's mHC op chain
        v_r, v_p, v_c, v_x = _vllm_seam(seam_lay, x_attn, res, post, comb)
        seam = (_rel(r, v_r), _rel(p, v_p), _rel(c, v_c))
        assert max(seam) < 1e-4, (S, seam)
        a = rt.scratch[rt.sc["a"] :].view(torch.int32)[: S * HIDDEN : 2]
        k_in = a.contiguous().view(torch.bfloat16).view(S, HIDDEN)
        assert _rel(k_in, v_x) < 1e-2, f"S={S}: FFN input {_rel(k_in, v_x)}"

        # MoE: vLLM's Glm5NextMoE on the kernel's own FFN input
        with set_current_vllm_config(vc), set_forward_context(None, vc):
            y = moe(k_in.clone())
        if not getattr(moe.experts, "reduce_results", True):
            dist.all_reduce(y)
        ref = _exact_moe(
            k_in,
            moe.gate.weight,
            moe.gate.e_score_correction_bias,
            w13,
            s13,
            w2,
            s2,
            shared,
        )
        dist.all_reduce(ref)  # this rank's intermediate shard
        e_kv = _rel(x, y)
        msg = f"TP{tp} S={S}: kernel vs vLLM {e_kv:.4f}"
        e_k, e_v = _rel(x, ref), _rel(y, ref)
        msg += f", vs exact: kernel {e_k:.4f} vLLM {e_v:.4f}"
        print(msg, flush=True)
        # vLLM's MXFP4 expert inputs make its own error the gap between the two
        assert e_k < 0.06 and e_k < 0.5 * e_v, msg
        assert abs(e_kv - e_v) < 0.25 * e_v, msg
        # routing: the kernel's top-8 is vLLM's (sigmoid + bias) on the same input
        scores = torch.sigmoid(k_in.float() @ moe.gate.weight.float().T)
        order = torch.argsort(
            -(scores + moe.gate.e_score_correction_bias), dim=-1, stable=True
        )[:, :TOP_K]
        sel = rt.scratch[rt.sc["sel"] :].view(torch.int32)[: 2 * S * MOE_SLOTS : 2]
        routed = sel.view(S, MOE_SLOTS)[:, 1:].sort(-1).values
        assert torch.equal(routed, order.int().sort(-1).values), f"S={S}: routing"
        dist.barrier(group)
        rt.close()
