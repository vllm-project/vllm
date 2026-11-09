# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's specialized decode in the engine, on the 8-layer dummy Kimi-K3
(KDA layers 1, 2 and 4-6 run K1 + K2; MLA layers 3 and 7 M1 + K2 with
VLLM_ROCM_USE_AITER_FP4BMM=1, else vLLM's MLA then K2) with its DSpark draft
at TP8 and DCP8.

- layers: eager verify steps, every stage of every kernel layer against the op
  vLLM's decoder layer runs for it, on the kernel's own input to that stage,
  the engine's KDA state slots and loaded weights (test_numerics' bounds);
- graphs: under FULL graphs every pure verify step runs the kernels on every
  rank as a graph replay, whether a request fills the 8-row graph (7
  speculative tokens) or one request of two replays it padded (3).
"""

import math

import pytest

from tests.models.kimi_k3.amd_specialized_decode.test_numerics import needs_gfx950
from tests.utils import multi_gpu_test
from vllm import SamplingParams

MODEL = "riverclouds/Kimi-K3-8L-dummy"
DRAFT = "riverclouds/Kimi-K3-DSpark-for-8L-dummy"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
    "1, 2, 3, 4, 5,",
]
PARAMS = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
LAYER_LAUNCHES = 28

# Activations at the checkpoint's scale for the layer test: scales every
# float dummy weight as the dummy loader draws it, 1-D norm weights to
# U(0.9, 1.1), matrices to U(+-sqrt(3 / fan_in)), before vLLM's post-load
# processing.
_SITECUSTOMIZE = """\
import importlib.abc
import importlib.machinery
import sys

_MODULE = "vllm.model_executor.model_loader.dummy_loader"


def _patch(module):
    init = module.initialize_single_dummy_weight

    def scaled(param, *args, name="", **kwargs):
        init(param, *args, name=name, **kwargs)
        if (
            not param.is_floating_point()
            or param.element_size() == 1
            or getattr(param, "dummy_weight_value", None) is not None
        ):
            return
        if param.dim() == 1 and name == "weight":
            param.data.mul_(100).add_(1)
        elif param.dim() > 1:
            param.data.mul_(1e3 * (3 / param[0].numel()) ** 0.5)

    module.initialize_single_dummy_weight = scaled


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def patched_exec(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = patched_exec
        return spec


sys.meta_path.insert(0, _Finder())
"""


@pytest.fixture(scope="module", autouse=True)
def _env():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ROCM_USE_AITER", "1")
        mp.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "asm")
        mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        yield


def _decode(worker):
    return worker.model_runner.model_state.language_model.specialized_decode


def _scaled_weights(worker) -> bool:
    from vllm.model_executor.model_loader import dummy_loader

    return dummy_loader.initialize_single_dummy_weight.__name__ == "scaled"


def _count_steps(worker, q: int) -> None:
    """Counts the scheduled steps that are pure verify steps (every request
    scheduled with its `q - 1` draft tokens), and records the width of every
    step plan() hands the kernels: under FULL graphs plan() runs only for steps
    that are not replayed."""
    decode, runner = _decode(worker), worker.model_runner
    plan, decode.eager_widths = decode.plan, []
    execute, decode.verify_steps = runner.execute_model, 0

    def counted_plan(*args):
        step = plan(*args)
        if step is not None:
            decode.eager_widths.append(step.width)
        return step

    def counted_execute(scheduler_output, *args, **kwargs):
        spec = scheduler_output.scheduled_spec_decode_tokens
        if not kwargs.get("dummy_run") and all(
            n == q and len(spec.get(r, ())) == q - 1
            for r, n in scheduler_output.num_scheduled_tokens.items()
        ):
            decode.verify_steps += scheduler_output.total_num_scheduled_tokens > 0
        return execute(scheduler_output, *args, **kwargs)

    decode.plan, runner.execute_model = counted_plan, counted_execute


def _steps(worker) -> tuple[int, int, int]:
    """Forwards that ran the kernels (each bumps the device epoch once), those of
    them that were not replayed, and the pure verify steps."""
    decode = _decode(worker)
    return int(decode.runner.epoch[0]), len(decode.eager_widths), decode.verify_steps


def _engine(vllm_runner, num_spec: int, max_num_seqs: int, **kwargs):
    return vllm_runner(
        MODEL,
        tensor_parallel_size=8,
        decode_context_parallel_size=8,
        dcp_comm_backend="a2a",
        language_model_only=True,
        load_format="dummy",
        kv_cache_dtype="fp8",
        max_model_len=2048,
        enable_chunked_prefill=True,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=0.15,
        # extend_kv_cache re-binds the KV views without prepare_kv, which
        # plan() rejects in eager steps (#66370)
        enable_extensible_kv_cache=False,
        attention_config={"mla_prefill_backend": "ROCM_AITER_FA"},
        speculative_config={
            "method": "dspark",
            "model": DRAFT,
            "num_speculative_tokens": num_spec,
            "attention_backend": "ROCM_AITER_MLA",
            "draft_sample_method": "probabilistic",
            "rejection_sample_method": "block",
        },
        additional_config={"decode_backend": "latency_specialized"},
        **kwargs,
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "num_spec, max_num_seqs, batches",
    [(7, 1, [PROMPTS]), (3, 2, [PROMPTS[:1], PROMPTS[1:]])],
    ids=["8_rows", "padded"],
)
def test_verify_steps_replay_the_kernels(vllm_runner, num_spec, max_num_seqs, batches):
    """Only 8-row FULL graphs are captured: every pure verify step, including
    a single request of 4 rows (`padded`, first batch), runs the kernels on
    every rank, each step a replay."""
    with _engine(
        vllm_runner,
        num_spec,
        max_num_seqs,
        compilation_config={
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [8],
        },
    ) as runner:
        llm = runner.llm
        llm.collective_rpc(_count_steps, args=(num_spec + 1,))
        for prompts in batches:
            before = llm.collective_rpc(_steps)
            llm.generate(prompts, PARAMS, use_tqdm=False)
            after = llm.collective_rpc(_steps)
            deltas = {tuple(a - b for a, b in zip(*ab)) for ab in zip(after, before)}
            assert len(deltas) == 1, deltas
            kernel_steps, eager, verify_steps = deltas.pop()
            assert kernel_steps == verify_steps > 0 and eager == 0


def _byte_perm(shuffle, shape):
    """The permutation a byte shuffle applies to a [shape] uint8 tensor: its
    output's byte i is input byte perm[i] (-1: padding), from shuffling each
    byte of the indices."""
    import torch

    idx = torch.arange(math.prod(shape), device="cuda")
    perm = sum(
        shuffle(((idx >> 8 * b) & 255).to(torch.uint8).view(shape)).flatten().long()
        << 8 * b
        for b in range(4)
    )
    ones = shuffle(torch.ones(shape, dtype=torch.uint8, device="cuda")).flatten()
    return torch.where(ones == 1, perm, -1)


def _raw_experts(experts) -> dict:
    """The routed experts' MXFP4 bytes and E8M0 codes [E, N, K / 2 | K / 32]
    (gate rows, then up), undoing the per-expert shuffles vLLM's K3 SiTU path
    applies at load (oracle/mxfp4.py)."""
    import torch
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops as ops

    fp4, e8 = torch.float4_e2m1fn_x2, torch.float8_e8m0fnu
    assert not ops.is_fused_moe_situv2_gate_up_interleaved()
    raw = {}
    for name in ("w13", "w2"):
        w = getattr(experts, f"{name}_weight").view(torch.uint8)
        e, n, kb = w.shape
        ks = kb // 16

        def shuffle_w(t):
            return ops.shuffle_weight_a16w4(t.view(fp4), 16, False).view(torch.uint8)

        def shuffle_s(t, name=name, ks=ks):
            t = t.view(e8).view(-1, ks)
            t = (
                ops.shuffle_scale_a16w4(t, 1, False)
                if name == "w13"
                else e8m0_shuffle(t)
            )
            return t.view(torch.uint8)

        for key, shuffle, cols, src in (
            (name, shuffle_w, kb, w),
            (name + "s", shuffle_s, ks, getattr(experts, f"{name}_weight_scale")),
        ):
            perm = _byte_perm(shuffle, (1, n, cols))
            keep = perm >= 0
            out = torch.empty(e, n * cols, dtype=torch.uint8, device="cuda")
            out[:, perm[keep]] = src.view(torch.uint8).reshape(e, -1)[:, keep]
            raw[key] = out.view(e, n, cols)
    return raw


def _gemm_parts(x, weight) -> int:
    """How many bf16-rounded K partials vLLM's bf16 linear of x @ weight^T
    sums on gfx950 (rocm_unquantized_gemm -> AITER's tuned gemm): a FlyDSL
    hgemm's split-K CTAs (_ks<n>) times its slice-K waves (_w<m>x<n>x<k>);
    1 for the other kernels (fp32 accumulation)."""
    import regex as re
    import torch
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    bf = str(torch.bfloat16)
    cfg = get_GEMM_A16W16_config(
        M=x.size(0), N=weight.size(0), K=weight.size(1), bias=False, dtype=bf, otype=bf
    )
    name = str(cfg.get("kernelName") or "")
    if cfg["libtype"] != "flydsl" or "hgemm" not in name:
        return 1
    ks, waves = re.search(r"_ks(\d+)", name), re.search(r"_w\d+x\d+x(\d+)", name)
    assert ks and waves, name
    return int(ks.group(1)) * int(waves.group(1))


def _k2_ops(layer, runner, rec) -> tuple[dict, dict, dict]:
    """(w, mw, ops) of _k2_vllm_check from the layer's modules: its weights,
    and the ops vLLM's own forward runs (each module's kernel)."""
    import torch

    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.models.kimi_k3.amd.linear import _apply_attn_res
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.stages.moe import UP_N

    m = layer.mlp
    moe = m.experts  # the latent MoE runner
    w_up = m.routed_expert_up_proj.weight
    if not getattr(m.routed_output_transform, "row_sharded", False):
        w_up = w_up.narrow(0, runner.rank * UP_N, UP_N)
    o_proj, sd = layer.self_attn.o_proj, m.shared_experts.down_proj
    s = rec["core"].size(0)
    w = {
        "core": rec["core"],
        "w_o": o_proj.weight,
        "prefix": rec["prefix"],
        "blocks": rec["blocks"][:s],
        "ares_nw": layer.mlp_res_norm.weight,
        "ares_qk": layer.mlp_res_proj.weight.view(HIDDEN),
        "in_nw": layer.post_attention_layernorm.weight,
    }
    mw = {
        "w_gate": m.gate.weight,
        "bias": m.gate.e_score_correction_bias,
        "w_ld": m.routed_expert_down_proj.weight,
        "w_sgu": m.shared_experts.gate_up_proj.weight,
        "w_sd": sd.weight,
        "ln_w": m.routed_expert_norm.weight,
        "w_up_shard": w_up,
        "raw": _raw_experts(moe.routed_experts),
    }

    def first(y):
        return y[0] if isinstance(y, tuple) else y

    def ares(prefix, blocks):
        return _apply_attn_res(
            prefix.clone(),
            blocks.contiguous(),
            layer.mlp_res_proj,
            layer.mlp_res_norm,
            blocks.size(1),
            output_norm=layer.post_attention_layernorm,
        )

    def select(x, logits):
        wts, ids = moe.router.select_experts(
            hidden_states=x,
            router_logits=logits,
            topk_indices_dtype=moe._quant_method.topk_indices_dtype,
        )
        return ids.long(), wts.float()

    def experts(lat, wts, ids):
        return moe.routed_experts.forward_modular(
            x=lat, topk_weights=wts.contiguous(), topk_ids=ids.to(torch.int32)
        )

    x = rec["core"].new_empty(s, HIDDEN)
    parts = {
        "ar": _gemm_parts(rec["core"], o_proj.weight),
        "lat": _gemm_parts(x, mw["w_ld"]),
        "sgu": _gemm_parts(x, mw["w_sgu"]),
    }
    ops = {
        "parts": parts,
        "partial": lambda core: o_proj.quant_method.apply(o_proj, core),
        "all_reduce": tensor_model_parallel_all_reduce,
        "attn_res": ares,
        "router": lambda x: first(m.gate(x)),
        "select": select,
        "lat": lambda x: first(m.routed_expert_down_proj(x)),
        "sgu": lambda x: first(m.shared_experts.gate_up_proj(x)),
        "situ": m.shared_experts.act_fn,
        "experts": experts,
        "lnorm": lambda lat: first(m.routed_expert_norm(lat)),
        "up": lambda ln, shp: shp.clone().addmm_(ln, w_up.t()),
        "sdown": lambda h: sd.quant_method.apply(sd, h),
    }
    return w, mw, ops


def _k1_check_engine(layer, runner, step, ins, rec) -> dict:
    """K1 of one KDA launch: AttnRes and in_proj as _k1_vllm_check; the conv,
    recurrence and gated norm as vLLM runs them on gfx950, AITER's fused spec
    decode (fp32 q, k, v and o), on the kernel's proj and copies of the
    states: each path against its own arithmetic's interval (the kernel's: q,
    k, v, o rounded to bf16), the two then differing by that rounding."""
    import torch
    from einops import rearrange

    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.kimi_k3.amd.linear import _apply_attn_res
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import (
        BETA_0,
        FA_0,
        G2_0,
        HD,
        HIDDEN,
        NH,
        PROJ,
        QKV,
    )

    attn, bf = layer.self_attn, torch.bfloat16
    md = step.attn_metadata[attn.prefix]
    S, q = md.spec_state_indices_tensor.numel(), md.spec_state_indices_tensor.size(1)
    nb = layer.prev_valid_blocks
    wi = layer.block_write_idx if layer.is_block_write_layer else -1
    # the step's slots, renumbered 1 .. n into small copies of the states
    slots = ins["slots"].unique()
    renum = torch.zeros(int(slots.max()) + 1, dtype=torch.int32, device="cuda")
    renum[slots] = torch.arange(1, slots.numel() + 1, dtype=torch.int32, device="cuda")
    st_idx = renum[md.spec_state_indices_tensor.long()]

    def small(states):
        return [torch.cat([s[:1] * 0, s]) for s in states]

    conv0, rs0 = small(ins["before"])
    conv1, rs1 = small(ins["after"])
    w = {
        "delta": ins["delta"],
        "ares_nw": layer.self_attention_res_norm.weight,
        "ares_qk": layer.self_attention_res_proj.weight.view(HIDDEN),
        "in_nw": layer.input_layernorm.weight,
        "w_in": attn.in_proj_qkvgfab.weight,
        "w_fb": attn.f_b_proj.weight,
        "conv_w": attn.conv1d.weight.view(QKV, 4),
        "a_log": attn.A_log,
        "dt_bias": attn.dt_bias,
        "on_w": attn.o_norm.weight,
        "conv_state": conv0,
        "rstate": rs0,
        "st_idx": st_idx,
        "num_acc": md.num_accepted_tokens,
        "lower_bound": attn.gate_lower_bound,
        "onorm_eps": attn.o_norm.eps,
    }
    tag = (int(runner.epoch.item()) % (1 << 22)) * 256 + layer.layer_idx + 1
    taps = N._k1_taps(w, {"scratch": rec["k1_scratch"]}, nb, wi, tag)
    m: dict = {}

    def check(name, got, c, r, fmt=bf):
        m[name] = max(m.get(name, 0.0), N._outside(got, c, r, fmt))

    prefix, blocks = ins["prefix"].clone(), ins["blocks"].clone()
    x = _apply_attn_res(
        prefix,
        blocks,
        layer.self_attention_res_proj,
        layer.self_attention_res_norm,
        nb,
        delta=ins["delta"],
        output_norm=layer.input_layernorm,
        block_write_idx=wi,
    )
    # a raise here would leave the other ranks in K2's all-reduces: recorded
    m["prefix_bits"] = float(not torch.equal(prefix, rec["k1_prefix"]))
    m["blocks_bits"] = float(not torch.equal(blocks, rec["blocks"]))
    c, v = N._attn_res_var(
        prefix,
        blocks[:, :nb],
        w["ares_nw"],
        w["ares_qk"],
        w["in_nw"],
        layer.self_attention_res_norm.variance_epsilon,
        layer.input_layernorm.variance_epsilon,
    )
    N._both(m, "x", taps["x"], x, c, N._r(v))
    c, v = N._dot_var(taps["x"], w["w_in"])
    proj_v = attn.in_proj_qkvgfab(taps["x"])[0][:, : c.size(1)]
    parts = _gemm_parts(taps["x"], w["w_in"])
    r_v = N._r(v + N._split_var(taps["x"], w["w_in"], parts))
    N._both(m, "proj", taps["proj"], proj_v, c, N._r(v), r_v=r_v)
    proj = taps["proj"].to(bf)
    g1 = attn.f_b_proj(proj[:, FA_0 : FA_0 + HD].contiguous())[0]
    conv_v, rs_v = conv0.clone(), rs0.clone()
    core_v = torch.empty(1, S, NH, HD, dtype=bf, device="cuda")
    q_, k_, v_ = (
        rearrange(t, "n (h d) -> 1 n h d", d=HD)
        for t in proj[:, :QKV].contiguous().split(PROJ, -1)
    )
    rocm_aiter_ops.kda_fused_recurrent_spec_decode(
        q=q_,
        k=k_,
        v=v_,
        g=rearrange(g1, "n (h d) -> 1 n h d", d=HD),
        beta=proj[None, :, BETA_0 : BETA_0 + NH].contiguous(),
        A_log=attn.A_log,
        dt_bias=attn.dt_bias,
        lower_bound=attn.gate_lower_bound,
        initial_state=rs_v,
        cu_seqlens=torch.arange(0, S + 1, q, dtype=torch.int32, device="cuda"),
        ssm_state_indices=st_idx,
        num_accepted_tokens=md.num_accepted_tokens,
        conv_state=conv_v,
        conv_weight=attn.decode_conv1d_weight,
        out_gate=rearrange(proj[:, G2_0 : G2_0 + PROJ], "n (h d) -> 1 n h d", d=HD),
        norm_weight=attn.o_norm.weight,
        norm_eps=attn.o_norm.eps,
        out=core_v,
    )
    # NaN-aware: blocks committed during warmup keep warmup contents, NaNs
    # included, in state entries the step neither reads nor writes
    same = (conv_v == conv1) | (conv_v.isnan() & conv1.isnan())
    m["conv_bits"] = float(not bool(same.all()))
    kernel = N._k1_path(w, taps, rounded=True)
    aiter = N._k1_path(w, taps, rounded=False, g1=g1)
    for rq, (c, r) in enumerate(kernel["qk"]):
        check("qk", taps["qk"][rq * q : (rq + 1) * q], c, r)
    for name, path, rs, core in (
        ("", kernel, rs1, rec["core"]),
        ("_v", aiter, rs_v, core_v.view(S, NH * HD)),
    ):
        for slot, (c, r) in path["state"].items():
            check("state" + name, rs[slot], c, r, torch.float32)
        for row, (c, r) in path["core"].items():
            check("core" + name, core[row].view(NH, HD), c, r)
    for row, (c, _) in aiter["core"].items():
        m["core_ulps"] = max(
            m.get("core_ulps", 0.0),
            N._ulps(rec["core"][row], core_v[0, row].reshape(-1), c.reshape(-1)),
        )
    return m


def _beyond(got, lo, hi, ulp) -> float:
    """How far `got` lies outside [lo, hi], in units of `ulp` (NaN: NaN)."""
    g = got.double()
    out = ((lo - g).clamp(min=0) + (g - hi).clamp(min=0)) / ulp
    return out.max().item() if out.numel() else 0.0


def _bf16_ulp(c):
    from tests.models import mono_reference as R

    return R.ulp(c.double().abs().clamp(min=2.0**-126))


def _e4m3_ulp(c):
    """The E4M3 spacing at |c| (3 mantissa bits, normal exponents >= -6)."""
    import torch

    e = torch.floor(torch.log2(c.double().abs().clamp(min=2.0**-6)))
    return torch.exp2(e - 3)


def _e4m3_ends(c, r):
    """[lo, hi]: what a value in c +- r rounds to through bf16, then E4M3
    (saturating at 448, RNE): both monotone."""
    from tests.models import mono_reference as R
    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N

    lo, hi = N._rounds(c, r)
    return R.to_e4m3_rne(lo), R.to_e4m3_rne(hi)


def _rms_var(x, w, eps, depth: int):
    """RMSNorm of exact bf16 rows x (float64) and both paths' fp32 error
    variance: the sum of squares `depth` deep over both paths, the rsq and the
    two products 4 U a path (K2's `ln`)."""
    import torch

    from tests.models import mono_reference as R
    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N

    x, n = x.double(), x.size(-1)
    y = R.rmsnorm(x, w, eps)
    ms = x.square().sum(-1, keepdim=True)
    rstd = torch.rsqrt(ms / n + eps)
    v_ms = N._sum_var((x**4).sum(-1, keepdim=True), ms, depth)
    v = (x * w.double()) ** 2 * (rstd**3 / 2 / n) ** 2 * v_ms + 2 * N._fn_var(y, 4)
    return y, v


def _even_codes(amax):
    """AITER's even MXFP4 scale code of an amax (`_mxfp4_scale_from_amax`,
    M1's `even_code`: the exponent rounded at mantissa bit 21, minus 2), at
    least 1 as M1 clamps it."""
    import torch

    bits = amax.float().view(torch.int32)
    return ((((bits + 0x200000) >> 23) & 0xFF) - 2).clamp(min=1)


def _mx_even_alts(base, lo, hi) -> tuple:
    """MXFP4 (the even code, E2M1 RNE: M1's and AITER's `batched_gemm_a16wfp4`
    prequant) of bf16 rows `base` [..., K], each value possibly another bf16
    value in [lo, hi]: (dequantized `base`, [(row, delta over K)]), as
    N._mx_alts with the even rule: each code a group's amax allows and,
    under it, each element's quant of `base` or of an end."""
    import torch

    from tests.models import mono_reference as R

    vb = base.unflatten(-1, (-1, 32))
    ends = [e.unflatten(-1, (-1, 32)) for e in (lo, hi)]
    code = _even_codes(vb.abs().amax(-1))
    alts = []

    def quant(c, groups):
        sc = R.e8m0(c)[..., None]
        q = R.to_e2m1_rne(vb / sc) * sc
        for end in ends:
            e = R.to_e2m1_rne(end / sc) * sc
            for *row, g, j in ((e != q) & groups[..., None]).nonzero().tolist():
                delta = torch.zeros_like(base[tuple(row)])
                delta[g * 32 + j] = (e - q)[(*row, g, j)]
                alts.append((tuple(row), delta))
        return q

    q = quant(code, torch.ones_like(code, dtype=torch.bool))
    small = torch.where(lo * hi > 0, torch.minimum(lo.abs(), hi.abs()), 0)
    big = torch.maximum(lo.abs(), hi.abs())
    c_lo, c_hi = (_even_codes(t.unflatten(-1, (-1, 32)).amax(-1)) for t in (small, big))
    for off in range(int((c_lo - code).min()), int((c_hi - code).max()) + 1):
        groups = (c_lo <= code + off) & (code + off <= c_hi)
        if not off or not groups.any():
            continue
        q2 = quant(code + off, groups)
        for *row, g in groups.nonzero().tolist():
            delta = torch.zeros_like(base[tuple(row)])
            delta[g * 32 : g * 32 + 32] = q2[(*row, g)] - q[(*row, g)]
            alts.append((tuple(row), delta))
    return q.flatten(-2), alts


def _mfma_var(x, w, blocks: int):
    """Rows x [..., h, K] against each head's rows w [h, n, K] (dequantized
    MXFP4) as FP4 MFMA chains over 128-blocks: (float64 [..., h, n], both
    paths' variance),
    each product through its MFMA's 128 adds (within 129 U) and the chain's
    `blocks` (K2's `mfma_var`)."""
    import torch

    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N

    c = torch.einsum("thk,hnk->thn", x, w)
    t2 = torch.einsum("thk,hnk->thn", x * x, w * w)
    return c, 2 * N._sum_var(t2, c, blocks + 129)


def _sigmoid_ends(lo, hi):
    """[lo, hi] of fp32 sigmoid(z) for z in [lo, hi]: exp2, the add and v_rcp
    (N._sigmoid_rel at 2 U a division) on each end."""
    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N

    U = N.U
    return (
        lo.sigmoid() * (1 - N._sigmoid_rel(lo, 2 * U)),
        hi.sigmoid() * (1 + N._sigmoid_rel(hi, 2 * U)),
    )


def _combine_var(o, lse, seq_lens, qsl, q: int):
    """The DCP LSE combine of this rank's heads (`_a2a_combine`'s masks), its
    float64 latent [S, 12, 512] and both paths' fp32 error variance, given
    every sender's partials o [R, S, 12, 512] (bf16), lse [R, S, 12] (fp32,
    base e) and local KV lengths seq_lens [R, requests].

    A weight w_r = exp(lse_r - lse) is formed from lse_r, the max m and the
    log-sum glse (M1b: exp2((lse_r - m) log2 e) / sum; vLLM: exp(lse_r -
    glse)): two roundings within U A_r, A_r = |lse_r - m| + |lse_r - glse|,
    and exp's 2 U, independent a weight; the normalization shared by a row's
    weights (the R-deep sum, rcp or log 2 U, glse's add U |glse| and log 2 U
    |log sum|); then the products and the R-deep sum of w_r o_r. First order:
    d lat = sum_r w_r (o_r - lat) e_r + lat eta."""
    import torch

    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N

    U = N.U
    R_, S = o.size(0), o.size(1)
    t = torch.arange(S, device=o.device)
    empty = (seq_lens[:, t // q] == 0) | (t >= qsl[S // q])
    lse = lse.double().masked_fill(empty.unsqueeze(-1), float("-inf"))
    lse = torch.where(lse.isnan() | (lse == float("inf")), float("-inf"), lse)
    live = lse > float("-inf")
    m = lse.max(0).values
    m = torch.where(m == float("-inf"), 0.0, m)
    e = torch.where(live, (lse - m).exp(), 0.0)
    tot = e.sum(0)
    has = tot > 0
    w = torch.where(has, e / torch.where(has, tot, 1.0), 0.0)
    glse = torch.where(has, m + torch.where(has, tot, 1.0).log(), 0.0)
    p = torch.where(live.unsqueeze(-1), o.double(), 0.0)
    wo = w.unsqueeze(-1) * p
    lat = wo.sum(0)
    a = torch.where(live, (lse - m).abs() + (lse - glse).abs(), 0.0)
    v_e = (2 * (U * a) ** 2 + (2 * U) ** 2) / 3
    tot1 = torch.where(has, tot, 1.0)
    log_s = tot1.log()
    v_eta = N._sum_var(e.square().sum(0), tot, R_) / tot1**2
    v_eta = v_eta + ((2 * U) ** 2 + (U * glse) ** 2 + (2 * U * log_s) ** 2) / 3
    v_eta = torch.where(has, v_eta, 0.0)
    dev = torch.where(live.unsqueeze(-1), p - lat, 0.0)
    var = (w.square() * v_e).unsqueeze(-1).mul(dev.square()).sum(0)
    var = var + lat.square() * v_eta.unsqueeze(-1)
    var = var + (U * wo).square().sum(0) / 3
    var = var + N._sum_var(wo.square().sum(0), lat, R_)
    return lat, 2 * var


def _m1_check_engine(layer, runner, rec) -> dict:
    """M1 of one MLA launch (`FfnLayer`) against vLLM's MLA ops on the
    kernel's input to each stage, as _k1_check_engine:

    - M1a: AttnRes (`x`), fused_qkv_a (`qa`) and g_proj (`gate`: the kernel
      writes bf16(sigmoid); vLLM keeps the logits, its sigmoid fused into the
      gate multiply below), the q / kv RMSNorm (`qc`; `kv`: the fp8 cache
      entry, `kpe_bits` its k_pe bytes), q_b (`qn`) and the fp8 query of this
      rank's heads, fp4 W_UK and q_pe (`q`, vLLM's batched_gemm_a16wfp4 and
      decode concat-quant);
    - M1b: the DCP combine (`lat_v`, vLLM's a2a combine; the kernel's latent
      stays in LDS), fp4 W_UV and the gate (`core`: on the kernel's gate;
      `core_v`: vLLM's `_v_up_proj` on its latent, fused_sigmoid_mul on its
      logits), the latent's MXFP4 from its interval (each code / value the two
      may round apart allowed its exact effect)."""
    import torch
    from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul

    from tests.models import mono_reference as R
    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.linear import _apply_attn_res
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_back
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.kda import HIDDEN
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention.mla import (
        KVL,
        NH,
        NOPE,
        QHD,
        QKV_A,
        QL,
        ROPE,
        scratch_layout,
    )
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import mla

    bf = torch.bfloat16
    wr = layer.self_attn.mla_attn
    a = wr.mla_attn
    pre, post = rec["m1a"], rec["m1b"]
    key, S, rank = pre["key"], pre["prefix"].size(0), runner.rank
    m: dict = {}  # M1a's
    mb: dict = {}  # M1b's
    heads = slice(rank * NH, (rank + 1) * NH)

    # M1b's collectives first (every rank checks the same launches)
    tp = get_tp_group()
    o_all = tp.all_gather(post["o"].contiguous().unsqueeze(0), dim=0)
    lse_all = tp.all_gather(post["lse"].contiguous().unsqueeze(0), dim=0)
    sl_all = tp.all_gather(post["seq_lens"].unsqueeze(0), dim=0)
    lat_v = a.dcp_manager.combine(
        post["o"], post["lse"], seq_lens=post["seq_lens"], query_start_loc=post["qsl"]
    )

    # M1a taps: x and the normed q_c plain bf16 rows, qkv_a and q_nope pairs
    lay = scratch_layout(key)
    sc = pre["scratch"]
    tag = (int(runner.epoch.item()) % (1 << 22)) * 256 + layer.layer_idx + 1
    x = R.words(sc[lay["x"][0] :], S * HIDDEN, bf).view(S, HIDDEN)
    qa = N._tap(sc[lay["qa"][0] :], S * QKV_A // 2, tag, bf).view(S, QKV_A)
    qc = R.words(sc[lay["qc"][0] :], S * QL, bf).view(S, QL)
    qn = N._tap(sc[lay["qn"][0] :], S * NH * NOPE // 2, tag, bf).view(S, NH, NOPE)

    nb = layer.prev_valid_blocks
    prefix, blocks = pre["prefix_in"].clone(), pre["blocks"].clone()
    x_v = _apply_attn_res(
        prefix,
        blocks,
        layer.self_attention_res_proj,
        layer.self_attention_res_norm,
        nb,
        delta=pre["delta"],
        output_norm=layer.input_layernorm,
    )
    m["prefix_bits"] = float(not torch.equal(prefix, pre["prefix"]))
    c, v = N._attn_res_var(
        prefix,
        blocks[:, :nb],
        layer.self_attention_res_norm.weight,
        layer.self_attention_res_proj.weight.view(HIDDEN),
        layer.input_layernorm.weight,
        layer.self_attention_res_norm.variance_epsilon,
        layer.input_layernorm.variance_epsilon,
    )
    N._both(m, "x", x, x_v, c, N._r(v))

    w_qkv, w_g = mla.qkv_gate_weights(wr)
    qkv_v = wr.fused_qkv_a_proj(x)[0]
    c, v = N._dot_var(x, w_qkv)
    parts = _gemm_parts(x, wr.fused_qkv_a_proj.weight)
    r_v = N._r(v + N._split_var(x, w_qkv, parts))
    N._both(m, "qa", qa, qkv_v[:, :QKV_A], c, N._r(v), r_v=r_v)
    # the gate: bf16(sigmoid(bf16 logits)), each monotone
    if wr.g_proj is not None:
        logits_v = wr.g_proj(x)[0]
        parts = _gemm_parts(x, wr.g_proj.weight)
    else:
        logits_v = qkv_v[:, QKV_A:]
    c_g, v_g = N._dot_var(x, w_g)
    r_g, r_gv = N._r(v_g), N._r(v_g + N._split_var(x, w_g, parts))
    s_lo, s_hi = _sigmoid_ends(*N._rounds(c_g, r_g))
    gate = pre["gate"]
    m["gate"] = _beyond(
        gate, R.bf16_rne(s_lo), R.bf16_rne(s_hi), _bf16_ulp(c_g.sigmoid())
    )
    m["gate_v"] = N._outside(logits_v, c_g, r_gv)

    # q / kv RMSNorm: the kernel's sums 4 a thread, 64 lanes, then 6 (q) or 2
    # (kv) waves; AITER's dual norm's taken K deep
    q_c, kv_lora = qa.split([QL, KVL + ROPE], -1)
    kv_c, k_pe = kv_lora.split([KVL, ROPE], -1)
    qc_v, kvn_v = wr._normalize_q_kv(q_c, kv_c)
    c, v = _rms_var(q_c, wr.q_a_layernorm.weight, key.q_eps, 3 + 6 + 5 + QL)
    N._both(m, "qc", qc, qc_v, c, N._r(v))
    own = pre["slots"] >= 0
    small = torch.zeros(S, 1, pre["cache"].size(-1), dtype=pre["cache"].dtype,
                        device=x.device)  # fmt: skip
    slots_v = torch.where(own, torch.arange(S, device=x.device), -1)
    a.impl.do_kv_cache_update(
        kvn_v,
        k_pe.unsqueeze(1),
        small,
        slots_v,
        a.kv_cache_dtype,
        a._k_scale,
        a.kv_cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2") and a._k_scale_float == 1.0,
    )
    fp8 = torch.float8_e4m3fn
    rows, rows_v = pre["cache"], small[own, 0]
    c, v = _rms_var(kv_c, wr.kv_a_layernorm.weight, key.kv_eps, 3 + 6 + 1 + KVL)
    c, r = c[own], N._r(v[own])
    lo, hi = _e4m3_ends(c, r)
    for name, got in (("kv", rows), ("kv_v", rows_v)):
        m[name] = _beyond(got[:, :KVL].view(fp8), lo, hi, _e4m3_ulp(c))
    kpe = R.to_e4m3_rne(k_pe[own].double())
    m["kpe_bits"] = float(
        not torch.equal(rows[:, KVL:], rows_v[:, KVL:])
        or not torch.equal(rows[:, KVL:].view(fp8).double(), kpe)
    )

    # q_b (nope rows -> q_nope pairs, pe rows -> the fp8 query)
    w_qb = wr.q_b_proj.weight.view(NH, QHD, QL)
    w_nope, w_pe = (w.reshape(-1, QL) for w in w_qb.split([NOPE, ROPE], 1))
    q_v = wr.q_b_proj(qc)[0].view(S, NH, QHD)
    parts = _gemm_parts(qc, wr.q_b_proj.weight)
    c, v = N._dot_var(qc, w_nope)
    r_v = N._r(v + N._split_var(qc, w_nope, parts))
    qn_v = q_v[..., :NOPE].reshape(S, -1)
    N._both(m, "qn", qn.view(S, -1), qn_v, c, N._r(v), r_v=r_v)
    c_pe, v_pe = N._dot_var(qc, w_pe)
    v_pe_v = v_pe + N._split_var(qc, w_pe, parts)
    # W_UK: the kernel's q_nope's MXFP4 (exact) against W_K, one 128-K MFMA;
    # this rank's heads (each rank checks its own)
    w_k = R.dequant_mxfp4(a.W_K.view(torch.uint8), a.W_K_scale.view(torch.uint8))
    xq = _mx_even_alts(qn.double(), qn.double(), qn.double())[0]
    c_uk, v_uk = _mfma_var(xq, w_k, 1)
    c_q = torch.cat([c_uk, c_pe.view(S, NH, ROPE)], -1)
    q8 = pre["q"][:, heads].view(fp8)
    ql_v = rocm_aiter_ops.batched_gemm_a16wfp4(
        qn.transpose(0, 1),
        a.W_K,
        a.W_K_scale,
        transpose_bm=True,
        prequant=True,
        y_scale=a._q_scale,
    )
    q8_v = a._decode_concat_quant_fp8_op(ql_v, q_v[..., NOPE:], a._q_scale)
    for name, got, vp in (("q", q8, v_pe), ("q_v", q8_v.view(fp8), v_pe_v)):
        r = N._r(torch.cat([v_uk, vp.view(S, NH, ROPE)], -1))
        lo, hi = _e4m3_ends(c_q, r)
        m[name] = _beyond(got, lo, hi, _e4m3_ulp(c_q))

    # M1b: the combine, the latent's MXFP4 (the values and codes its interval
    # allows), W_UV (4 128-K MFMAs), bf16, then the gate
    lse = lse_all[:, :, heads].float()
    if not a.impl.lse_base_on_e:
        lse = lse * math.log(2)
    c_l, v_l = _combine_var(
        o_all[:, :, heads], lse, sl_all, post["qsl"], runner.num_spec + 1
    )
    lo_l, hi_l = N._rounds(c_l, N._r(v_l))
    mb["lat_v"] = N._outside(lat_v, c_l, N._r(v_l))
    lat_q, alts = _mx_even_alts(R.bf16_rne(c_l), lo_l, hi_l)
    w_v = R.dequant_mxfp4(a.W_V.view(torch.uint8), a.W_V_scale.view(torch.uint8))
    c_u, v_u = _mfma_var(lat_q, w_v, KVL // 128)
    allow = torch.zeros_like(c_u)
    for (t, h), d in alts:
        allow[t, h] += (w_v[h] @ d).abs()
    mb["mx_alts"] = len(alts)
    lo_u, hi_u = N._rounds(c_u, N._r(v_u) + allow)
    c_u, lo_u, hi_u = (t.reshape(S, -1) for t in (c_u, lo_u, hi_u))
    g = post["gate"].double()
    ulp = _bf16_ulp(R.bf16_rne(c_u) * g)
    ends = torch.stack([lo_u * g, hi_u * g])
    mb["core"] = _beyond(
        rec["core"], R.bf16_rne(ends.amin(0)), R.bf16_rne(ends.amax(0)), ulp
    )
    uv_v = torch.empty(S, NH * mla_back.VD, dtype=bf, device=x.device)
    a._v_up_proj(lat_v, out=uv_v)
    core_v = fused_sigmoid_mul(uv_v, logits_v.contiguous(), out=torch.empty_like(uv_v))
    sg_lo, sg_hi = _sigmoid_ends(*N._rounds(c_g, r_gv))
    ends = torch.stack([u * s for u in (lo_u, hi_u) for s in (sg_lo, sg_hi)])
    # the fp32 product's rounding
    lo, hi = ends.amin(0), ends.amax(0)
    lo, hi = lo - N.U * lo.abs(), hi + N.U * hi.abs()
    mb["core_v"] = _beyond(core_v, R.bf16_rne(lo), R.bf16_rne(hi), ulp)
    return {f"m1a_{k}": v for k, v in m.items()} | {
        f"m1b_{k}": v for k, v in mb.items()
    }


def _m1b_inputs(kw: dict) -> dict:
    """Copies of M1b's attention partials, their metadata and the gate."""
    return {k: kw[k].clone() for k in ("o", "lse", "seq_lens", "qsl", "gate")}


def _check_layers(worker, launches: int) -> None:
    """For the first `launches` kernel launches, each stage of the kernels
    against the op vLLM's layer runs for it on the kernel's own input to that
    stage (test_numerics' *_matches_vllm_ops checks, with the engine's
    modules, weights and KDA states). Checked launches run the plain K2
    (`opt=0`, bit-exact to `K2_OPT`, test_numerics): K2_OPT's sentinel
    all-reduces re-arm the words they read, leaving no taps."""
    import dataclasses

    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N
    from vllm.distributed import get_tp_group
    from vllm.models.kimi_k3.amd.specialized_decode.mono import layer as k2_mod
    from vllm.models.kimi_k3.amd.specialized_decode.mono.abi import FAILED
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla as m1_mod
    from vllm.models.kimi_k3.amd.specialized_decode.mono.attention import mla_back
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import layers

    decode = _decode(worker)
    decode.layer_errors = []
    k2_key = layers._k2_key

    def plain_key(*args, **kwargs):
        return dataclasses.replace(k2_key(*args, **kwargs), opt=0)

    for kernels in decode.kernels.values():
        for kernel in kernels.values():
            for b in kernel.builds():
                if isinstance(b, k2_mod.K2Build):
                    layers.compile_build(dataclasses.replace(b, opt=0))
    get_tp_group().barrier()

    def wrap(kernel):
        launch = kernel.launch

        def checked(step, positions, prefix_sum, block_residual, prefix_delta):
            if len(decode.layer_errors) >= launches:
                return launch(step, positions, prefix_sum, block_residual, prefix_delta)
            layer, runner = kernel.layer, kernel.runner
            kda = isinstance(kernel, layers.WholeLayer)
            ffn = isinstance(kernel, layers.FfnLayer)
            ins = {
                "prefix": prefix_sum.clone(),
                "delta": None if prefix_delta is None else prefix_delta.clone(),
                "blocks": block_residual.clone(),
            }
            if kda:
                md = step.attn_metadata[layer.self_attn.prefix]
                ins["slots"] = md.spec_state_indices_tensor.flatten().unique()
                ins["before"] = [c[ins["slots"]].clone() for c in kernel.caches()]
            rec: dict = {}
            k2 = layers._k2

            def spy(layer_, runner_, core, prefix, blocks, reset=False, **m1):
                # K1's scratch and K2's inputs, before K2 reuses the scratch
                rec["k1_scratch"] = runner_.scratch.clone()
                rec["core"], rec["prefix"] = core.clone(), prefix.clone()
                rec["blocks"] = blocks.clone()
                fused = m1.get("m1b_in") is not None
                if fused:
                    rec["m1b"] = _m1b_inputs(m1["m1b_in"])
                rec["key"] = layers._k2_key(layer_, runner_, core.size(0), reset, fused)
                out = k2(layer_, runner_, core, prefix, blocks, reset, **m1)
                if fused:
                    # M1b in K2's launch wrote o_proj's input
                    rec["core"] = core.clone()
                return out

            pre, post = m1_mod.mla_pre, mla_back.mla_post

            def pre_spy(key, **kw):
                # M1a's inputs and outputs, its scratch before K2 reuses it
                rec["m1a"] = {
                    "key": key,
                    "prefix_in": kw["prefix"].clone(),
                    "delta": None if kw["delta"] is None else kw["delta"].clone(),
                    "blocks": kw["blocks"].clone(),
                    "slots": kw["slots"].clone(),
                }
                pre(key, **kw)
                cache, slots = kw["kv_cache"], kw["slots"]
                own = slots[slots >= 0]
                rec["m1a"].update(
                    prefix=kw["prefix"].clone(),
                    scratch=kw["scratch"].clone(),
                    q=kw["q_out"].clone(),
                    gate=kw["gate_out"].clone(),
                    cache=cache[own // cache.size(1), own % cache.size(1)].clone(),
                )

            def post_spy(key, **kw):
                rec["m1b"] = _m1b_inputs(kw)
                post(key, **kw)

            layers._k2, layers._k2_key = spy, plain_key
            m1_mod.mla_pre, mla_back.mla_post = pre_spy, post_spy
            try:
                got = launch(step, positions, prefix_sum, block_residual, prefix_delta)
            finally:
                layers._k2, layers._k2_key = k2, k2_key
                m1_mod.mla_pre, mla_back.mla_post = pre, post
            # a wait that gave up (abi.WAIT_SECONDS) left stale taps
            assert not int(runner.health[FAILED]), (layer.layer_idx, "a wait gave up")
            errs = {}
            if kda:
                ins["after"] = [c[ins["slots"]].clone() for c in kernel.caches()]
                rec["k1_prefix"] = prefix_sum if rec["key"].reset else rec["prefix"]
                errs.update(_k1_check_engine(layer, runner, step, ins, rec))
            if ffn:
                errs.update(_m1_check_engine(layer, runner, rec))
            w, mw, ops = _k2_ops(layer, runner, rec)
            tag = (
                (int(runner.epoch.item()) % (1 << 22)) * 256 + 128 + layer.layer_idx + 1
            )
            k2m = N._k2_vllm_check(runner, rec["key"], w, mw, got[0], got[2], ops, tag)
            errs.update({f"k2_{k}": v for k, v in k2m.items()})
            decode.layer_errors.append((layer.layer_idx, type(kernel).__name__, errs))
            # a peer's next launch must not wait (abi.WAIT_SECONDS, then go on
            # with stale words) on a rank still checking, nor push into the
            # regions it reads
            get_tp_group().barrier()
            return got

        kernel.launch = checked

    for kernels in decode.kernels.values():
        for kernel in kernels.values():
            wrap(kernel)


def _layer_errors(worker):
    return _decode(worker).layer_errors


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_layer_kernels_match_vllm_layers(vllm_runner, tmp_path_factory):
    """Eager verify steps of two requests at 3 speculative tokens, on scaled
    dummy weights: every stage of every kernel layer (K1 of the KDA layers,
    M1a and M1b of the MLA layers M1 runs, K2 of all) and the op vLLM's layer
    runs for it within the stage's interval, on every rank."""
    import os

    from tests.models.kimi_k3.amd_specialized_decode.test_numerics import K2_STAGES
    from vllm import envs

    site = tmp_path_factory.mktemp("kimi_k3_scaled_dummy")
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    path = os.pathsep.join(filter(None, [str(site), os.getenv("PYTHONPATH")]))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", path)
        with _engine(vllm_runner, 3, 2, enforce_eager=True) as runner:
            llm = runner.llm
            assert all(llm.collective_rpc(_scaled_weights))
            llm.collective_rpc(_check_layers, args=(LAYER_LAUNCHES,))
            llm.generate(PROMPTS[:2], PARAMS, use_tqdm=False)
            ranks = llm.collective_rpc(_layer_errors)
    k1 = ("prefix_bits", "blocks_bits", "conv_bits", "x", "x_v", "proj", "proj_v")
    k1 = (*k1, "qk", "state", "state_v", "core", "core_v")
    k2 = tuple(f"k2_{s}" for s in K2_STAGES)
    m1a = ("prefix_bits", "x", "x_v", "qa", "qa_v", "gate", "gate_v", "qc", "qc_v")
    m1a = (*m1a, "kv", "kv_v", "kpe_bits", "qn", "qn_v", "q", "q_v")
    m1 = (*(f"m1a_{s}" for s in m1a), "m1b_lat_v", "m1b_core", "m1b_core_v")
    # the MLA layers run M1 with AITER's MXFP4 bmms, else vLLM's ops (m1_reasons)
    mla = "FfnLayer" if envs.VLLM_ROCM_USE_AITER_FP4BMM else "MlaOpsLayer"
    for errors in ranks:
        assert len(errors) == LAYER_LAUNCHES
        assert {i for i, _, _ in errors} == set(range(1, 8))
        for i, kind, errs in errors:
            assert kind == (mla if i in (3, 7) else "WholeLayer"), (i, kind)
            stages = {"WholeLayer": (*k1, *k2), "FfnLayer": (*m1, *k2)}.get(kind, k2)
            assert all(errs[k] == 0 for k in stages), (i, errs)
