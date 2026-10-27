# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3's specialized decode in the engine, on the 8-layer dummy Kimi-K3
(KDA layers 1, 2 and 4-6 run K1 + K2, MLA layers 3 and 7 vLLM's MLA then K2)
with its DSpark draft at TP8 and DCP8.

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
        mp.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
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
    scheduled with its ``q - 1`` draft tokens), and records the width of every
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
    a single request of 4 rows (``padded``, first batch), runs the kernels on
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
    m["conv_bits"] = float(not torch.equal(conv_v, conv1))
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


def _check_layers(worker, launches: int) -> None:
    """For the first ``launches`` kernel launches, each stage of the kernels
    against the op vLLM's layer runs for it on the kernel's own input to that
    stage (test_numerics' *_matches_vllm_ops checks, with the engine's
    modules, weights and KDA states)."""
    from tests.models.kimi_k3.amd_specialized_decode import test_numerics as N
    from vllm.models.kimi_k3.amd.specialized_decode.runtime import layers

    decode = _decode(worker)
    decode.layer_errors = []

    def wrap(kernel):
        launch = kernel.launch

        def checked(step, positions, prefix_sum, block_residual, prefix_delta):
            if len(decode.layer_errors) >= launches:
                return launch(step, positions, prefix_sum, block_residual, prefix_delta)
            layer, runner = kernel.layer, kernel.runner
            kda = hasattr(kernel, "caches")
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

            def spy(layer_, runner_, core, prefix, blocks, reset=False):
                # K1's scratch and K2's inputs, before K2 reuses the scratch
                rec["k1_scratch"] = runner_.scratch.clone()
                rec["core"], rec["prefix"] = core.clone(), prefix.clone()
                rec["blocks"] = blocks.clone()
                rec["key"] = layers._k2_key(layer_, runner_, core.size(0), reset)
                return k2(layer_, runner_, core, prefix, blocks, reset)

            layers._k2 = spy
            try:
                got = launch(step, positions, prefix_sum, block_residual, prefix_delta)
            finally:
                layers._k2 = k2
            errs = {}
            if kda:
                ins["after"] = [c[ins["slots"]].clone() for c in kernel.caches()]
                rec["k1_prefix"] = prefix_sum if rec["key"].reset else rec["prefix"]
                errs.update(_k1_check_engine(layer, runner, step, ins, rec))
            w, mw, ops = _k2_ops(layer, runner, rec)
            tag = (
                (int(runner.epoch.item()) % (1 << 22)) * 256 + 128 + layer.layer_idx + 1
            )
            k2m = N._k2_vllm_check(runner, rec["key"], w, mw, got[0], got[2], ops, tag)
            errs.update({f"k2_{k}": v for k, v in k2m.items()})
            decode.layer_errors.append((layer.layer_idx, errs))
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
    K2 of all) and the op vLLM's layer runs for it within the stage's
    interval, on every rank."""
    import os

    from tests.models.kimi_k3.amd_specialized_decode.test_numerics import K2_STAGES

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
    for errors in ranks:
        assert len(errors) == LAYER_LAUNCHES
        assert {i for i, _ in errors} == set(range(1, 8))
        for i, errs in errors:
            stages = k2 if i in (3, 7) else (*k1, *k2)
            assert all(errs[k] == 0 for k in stages), (i, errs)
