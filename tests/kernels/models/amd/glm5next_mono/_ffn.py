# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash FFN-layer MonoKernel on CDNA4, one process a TP rank.

Random weights in the post-load AITER W4A4 layout. The seam is checked against
vLLM's own mHC op chain, the MoE against an fp32 reference of the same routing on
the kernel's own FFN input (so routing cannot differ), with the expert inputs
quantized as the kernel declares (MXFP8)."""

import os

import ray
import torch

from tests.utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
)
from vllm.platforms import current_platform

REPLAYS = 300
ROWS = (2, 4, 8, 16)
# serving waits time out after a minute; the stalled-rank checks after 0.5 s
POLL_TIMEOUT_US = 60_000_000
FAULT_POLL_TIMEOUT_US = 500_000


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


def _rel(a, b) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


_FP4 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _dequant(q, s):
    fp4 = torch.tensor(_FP4, device=q.device)
    v = torch.stack([fp4[(q & 15).long()], fp4[(q >> 4).long()]], -1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def _mxfp8(a):
    """The kernel's activation quant: per 32, e8m0 scale ceil(amax / 448), e4m3."""
    shp, a = a.shape, a.float().reshape(-1, 32)
    bits = (a.abs().amax(-1, keepdim=True) / 448.0).view(torch.int32)
    e = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(1, 254)
    scale = torch.exp2(e.float() - 127)
    q = (a / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(shp)


def _moe_reference(x, w_r, bias, w13, w2, inter):
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        HIDDEN,
        N_EXPERTS,
        ROUTE_SCALE,
        SWIGLU_LIMIT,
        TOP_K,
    )

    scores = torch.sigmoid(x.float() @ w_r.float().T)
    order = torch.argsort(-(scores + bias), dim=-1, stable=True)[:, :TOP_K]
    picked = torch.gather(scores, 1, order)
    weights = picked / picked.sum(-1, keepdim=True) * ROUTE_SCALE
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    xin = _mxfp8(x)
    for s in range(x.shape[0]):
        for e, wt in zip([N_EXPERTS] + order[s].tolist(), [1.0] + weights[s].tolist()):
            gu = w13[e] @ xin[s]
            gate = gu[:inter].clamp(max=SWIGLU_LIMIT)
            up = gu[inter:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            out[s] += wt * (w2[e] @ _mxfp8(torch.nn.functional.silu(gate) * up))
    return out, order


class _Layer:
    """One layer's random weights (every rank builds the same) and this rank's shard."""

    def __init__(self, rank: int, tp: int, device) -> None:
        from aiter.ops.shuffle import shuffle_scale, shuffle_weight

        from vllm.models.glm5next.amd.specialized_decode.mono.config import (
            GLOBAL_INTER,
            HC_MULT,
            HIDDEN,
            N_EXPERTS,
        )
        from vllm.models.glm5next.amd.specialized_decode.mono.packing import pack_bf16

        g = torch.Generator(device=device).manual_seed(0)
        E, full, self.inter = N_EXPERTS + 1, GLOBAL_INTER, GLOBAL_INTER // tp

        def mxfp4(rows, k):
            q = torch.randint(0, 256, (E, rows, k // 2), generator=g, device=device)
            s = torch.randint(119, 124, (E, rows, k // 32), generator=g, device=device)
            return q.to(torch.uint8), s.to(torch.uint8)

        self.w_r = (
            torch.randn(N_EXPERTS, HIDDEN, generator=g, device=device) * 0.02
        ).bfloat16()
        self.bias = (torch.rand(N_EXPERTS, generator=g, device=device) - 0.5) * 0.2
        q13, s13 = mxfp4(2 * full, HIDDEN)
        q2, s2 = mxfp4(HIDDEN, full)
        self.w13, self.w2 = _dequant(q13, s13), _dequant(q2, s2)
        lo, hi, n = rank * self.inter, (rank + 1) * self.inter, self.inter
        fp4 = torch.float4_e2m1fn_x2

        def rows(t):
            return torch.cat([t[:, lo:hi], t[:, full + lo : full + hi]], 1).contiguous()

        self.k_w13 = shuffle_weight(rows(q13).view(fp4), layout=(16, 16))
        self.k_s13 = shuffle_scale(rows(s13).view(E * 2 * n, -1)).view(E, 2 * n, -1)
        self.k_w2 = shuffle_weight(
            q2[:, :, lo // 2 : hi // 2].contiguous().view(fp4), layout=(16, 16)
        )
        self.k_s2 = shuffle_scale(
            s2[:, :, lo // 32 : hi // 32].contiguous().view(E * HIDDEN, -1)
        ).view(E, HIDDEN, -1)
        self.k_wr = pack_bf16(self.w_r)
        # the shared expert as vLLM stores it apart: row-major, scale per row
        self.sh_rows = (
            rows(q13)[N_EXPERTS],
            rows(s13)[N_EXPERTS],
            q2[N_EXPERTS, :, lo // 2 : hi // 2].contiguous(),
            s2[N_EXPERTS, :, lo // 32 : hi // 32].contiguous(),
        )
        nmix = HC_MULT * (2 + HC_MULT)
        self.fn = torch.randn(nmix, HC_MULT * HIDDEN, generator=g, device=device) * 0.01
        self.scale = torch.rand(3, generator=g, device=device) + 0.5
        self.base = torch.randn(nmix, generator=g, device=device) * 0.1
        self.gamma = (
            1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)
        ).bfloat16()
        self.g = g

    def weights(self, separate: bool = False):
        from vllm.models.glm5next.amd.specialized_decode.mono.config import N_EXPERTS
        from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import (
            FlashFfnWeights,
        )
        from vllm.models.glm5next.amd.specialized_decode.mono.packing import (
            pack_shared_expert,
        )

        routed = (self.k_w13, self.k_s13, self.k_w2, self.k_s2)
        if separate:
            shared = pack_shared_expert(*self.sh_rows)
            routed = tuple(t[:N_EXPERTS].contiguous() for t in routed)
        else:
            shared = tuple(t[N_EXPERTS] for t in routed)
        return FlashFfnWeights(
            w_r=self.k_wr,
            bias=self.bias,
            w_ug=routed[0],
            s_ug=routed[1],
            w_dn=routed[2],
            s_dn=routed[3],
            w_sh_ug=shared[0],
            s_sh_ug=shared[1],
            w_sh_dn=shared[2],
            s_sh_dn=shared[3],
            hc_fn=self.fn,
            hc_scale=self.scale,
            hc_base=self.base,
            g_norm=self.gamma,
        )


def _vllm_seam(lay, x_attn, res, post, comb):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp, _apply_mhc_norm
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        EPS,
        HC_EPS,
        HC_POST_MULT,
        SINKHORN_ITERS,
    )

    with set_current_vllm_config(VllmConfig()):
        r, p, c, li = MHCFusedPostPreOp().forward_native(
            x_attn,
            res,
            post,
            comb,
            lay.fn,
            lay.scale,
            lay.base,
            EPS,
            HC_EPS,
            HC_EPS,
            HC_POST_MULT,
            SINKHORN_ITERS,
            1,
            1,
            lay.gamma,
            EPS,
        )
        return r, p, c, _apply_mhc_norm(li, lay.gamma, EPS)


def _runtime(width, rank, tp, poll_timeout_us, launches=1, early_out=False):
    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashFfnRuntime

    return FlashFfnRuntime(
        width,
        rank,
        tp,
        get_tp_group().cpu_group,
        2048 // tp,
        launches,
        poll_timeout_us,
        poll_early_out=early_out,
    )


def _check(rank: int, tp: int, device, separate: bool) -> None:
    import torch.distributed as dist

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.config import (
        HC_MULT,
        HIDDEN,
        MOE_SLOTS,
    )
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashFfnOp

    lay = _Layer(rank, tp, device)
    group = get_tp_group().cpu_group
    # 16 TP2 rows overflow the down stage's LDS
    for S in ROWS if tp > 2 else ROWS[:-1]:
        rt = _runtime(S, rank, tp, POLL_TIMEOUT_US)
        op = FlashFfnOp(rt, lay.weights(separate), 0)
        g = lay.g
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

        # every rank holds the same result
        gathered = [torch.empty_like(x).cpu() for _ in range(tp)]
        dist.all_gather(gathered, x.cpu(), group=group)
        assert all(torch.equal(gathered[0], o) for o in gathered), (
            f"S={S}: ranks differ"
        )

        # the seam against vLLM's mHC op chain
        v_r, v_p, v_c, v_x = _vllm_seam(lay, x_attn, res, post, comb)
        assert _rel(r, v_r) < 1e-4, f"S={S}: R' {_rel(r, v_r)}"
        assert _rel(p, v_p) < 1e-5 and _rel(c, v_c) < 1e-5, (
            S,
            _rel(p, v_p),
            _rel(c, v_c),
        )
        a = rt.scratch[rt.sc["a"] :].view(torch.int32)[: S * HIDDEN : 2]
        k_in = a.contiguous().view(torch.bfloat16).view(S, HIDDEN)
        assert _rel(k_in, v_x) < 1e-2, f"S={S}: FFN input {_rel(k_in, v_x)}"

        # the MoE on the kernel's own FFN input: identical routing, then numerics
        ref, order = _moe_reference(k_in, lay.w_r, lay.bias, lay.w13, lay.w2, 2048)
        sel = rt.scratch[rt.sc["sel"] :].view(torch.int32)[: 2 * S * MOE_SLOTS : 2]
        routed = sel.view(S, MOE_SLOTS)[:, 1:].sort(-1).values
        assert torch.equal(routed, order.int().sort(-1).values), f"S={S}: routing"
        assert _rel(x, ref) < 5e-3, f"S={S}: MoE {_rel(x, ref)}"

        # the packed shared expert computes exactly what the fused slot does
        if separate:
            dist.barrier(group)
            rt.advance_step()
            fused = FlashFfnOp(rt, lay.weights(), 0)
            y = [t.clone() for t in fused.launch(parts[rank], res, post, comb)]
            torch.accelerator.synchronize()
            assert all(torch.equal(a, b) for a, b in zip(y, (x, r, p, c))), (
                f"S={S}: packed shared expert differs from the fused one"
            )

        # a non-finite padding row stays in its own row
        bad = parts[rank].clone()
        bad[-1] = float("nan")
        res_bad = res.clone()
        res_bad[-1] = float("inf")
        dist.barrier(group)
        rt.advance_step()
        y = [t.clone() for t in op.launch(bad, res_bad, post, comb)]
        torch.accelerator.synchronize()
        for got, want in zip(y, (x, r, p, c)):
            assert torch.equal(got[:-1], want[:-1]), f"S={S}: a NaN row leaked"

        # back-to-back graph replays, every one bit for bit the first
        dist.barrier(group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rt.advance_step()
            outs = op.launch(parts[rank], res, post, comb)
        for i in range(REPLAYS):
            graph.replay()
            if i % 100 == 99 or i == REPLAYS - 1:
                torch.accelerator.synchronize()
                assert all(torch.equal(o, w) for o, w in zip(outs, (x, r, p, c))), (
                    f"S={S}: replay {i} differs"
                )
        assert not rt.words.any()
        dist.barrier(group)
        rt.close()


def _check_fault(rank: int, tp: int, device, early_out: bool = False) -> None:
    import tempfile

    import torch.distributed as dist

    # read at kernel build time; a private cache keeps the stalling build apart
    os.environ["VLLM_MONO_FAULT_INJECT"] = "rank=1,step=0,us=2000000"
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix="mono_fault_")

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.config import HC_MULT, HIDDEN
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashFfnOp

    lay = _Layer(rank, tp, device)
    rt = _runtime(2, rank, tp, FAULT_POLL_TIMEOUT_US, early_out=early_out)
    op = FlashFfnOp(rt, lay.weights(), 0)
    x = torch.randn(2, HIDDEN, device=device).bfloat16()
    res = torch.randn(2, HC_MULT, HIDDEN, device=device).bfloat16()
    post = torch.ones(2, HC_MULT, 1, device=device)
    comb = torch.full((2, HC_MULT, HC_MULT), 1.0 / HC_MULT, device=device)
    group = get_tp_group().cpu_group
    dist.barrier(group)
    op.launch(x, res, post, comb)
    torch.accelerator.synchronize()
    dist.barrier(group)
    report = [None] * tp
    dist.all_gather_object(report, rt.words.tolist(), group=group)
    if rank == 0:
        waiting = [r for r in range(tp) if r != 1]
        flags = rt.flags.tolist()
        assert all(any(report[r]) for r in waiting), report
        assert all(flags[r] for r in waiting), flags
    rt.close()


def _check_slow_rank(rank: int, tp: int, device) -> None:
    """A rank 10 s late into a launch (as a host stall on one rank makes it) at
    the serving poll settings: the others wait it out, nothing expires, and the
    launch computes what an on-time one does."""
    import tempfile

    import torch.distributed as dist

    # read at kernel build time; a private cache keeps the stalling build apart
    os.environ["VLLM_MONO_FAULT_INJECT"] = "rank=1,step=0,us=10000000"
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = tempfile.mkdtemp(prefix="mono_slow_")

    from vllm.distributed import get_tp_group
    from vllm.models.glm5next.amd.specialized_decode.mono.config import HC_MULT, HIDDEN
    from vllm.models.glm5next.amd.specialized_decode.mono.ffn.op import FlashFfnOp
    from vllm.models.glm5next.amd.specialized_decode.runtime.spec import DecodeConfig

    lay = _Layer(rank, tp, device)
    rt = _runtime(2, rank, tp, DecodeConfig(tp_size=tp).poll_timeout_us, early_out=True)
    op = FlashFfnOp(rt, lay.weights(), 0)
    x = torch.randn(2, HIDDEN, device=device).bfloat16()
    res = torch.randn(2, HC_MULT, HIDDEN, device=device).bfloat16()
    post = torch.ones(2, HC_MULT, 1, device=device)
    comb = torch.full((2, HC_MULT, HC_MULT), 1.0 / HC_MULT, device=device)
    group = get_tp_group().cpu_group
    dist.barrier(group)
    late = [t.clone() for t in op.launch(x, res, post, comb)]
    torch.accelerator.synchronize()
    dist.barrier(group)
    rt.advance_step()
    on_time = [t.clone() for t in op.launch(x, res, post, comb)]
    torch.accelerator.synchronize()
    dist.barrier(group)
    report = [None] * tp
    dist.all_gather_object(report, rt.words.tolist(), group=group)
    if rank == 0:
        assert not any(any(w) for w in report), report
        assert not any(rt.flags.tolist()), rt.flags.tolist()
    assert all(torch.equal(a, b) for a, b in zip(late, on_time))
    rt.close()


def _worker(check, *args):
    @ray.remote(num_gpus=1, max_calls=1)
    def run(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
        with monkeypatch.context() as m:
            m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
            device = torch.device(f"cuda:{rank}")
            torch.accelerator.set_device_index(device)
            init_test_distributed_environment(
                tp_size, pp_size, rank, distributed_init_port
            )
            ensure_model_parallel_initialized(tp_size, pp_size)
            check(rank, tp_size, device, *args)

    return run
