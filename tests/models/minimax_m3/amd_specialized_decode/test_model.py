# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer- and model-level precision of --monokernel on a mini MiniMax M3 at
the real per-layer shapes, TP4, with random weights (no checkpoint): layers
0-2 dense and stock, 3-4 sparse and fused, the last one ending in the step-end
handshake. Engines load dummy weights, so vLLM's real MXFP4 / AITER post-load
path runs; every rank then overwrites them in place at sensible scales,
distinct per layer and per TP shard, the experts generated in the checkpoint's
MXFP4 format and converted by that same path.

- layer: each sparse ``MiniMaxM3DecoderLayer``'s FFN half (seam all-reduce and
  residual, post-attention norm, MoE) on native's captured decode-step inputs;
- model: teacher-forced decode (trace replay), --monokernel eager: every fused
  launch of the engine's real steps against vLLM's own ops on that launch's
  inputs, within the a-priori bounds of both paths' arithmetic (``_launch_check``);
  under FULL graphs (with frequent epoch resets) the full-vocabulary logprobs
  equal eager's bit for bit;
- fail-stop: a late rank fault under FULL graphs returns no token.

End to end, a ulp-level difference of a launch is amplified past any a-priori
bound: native reruns differ alike, as native's MoE sums its terms with BF16
atomics in run-dependent order, and attention over the FP8 KV cache and top-k
near-ties grow such differences. So the logprobs are compared against
--monokernel eager, which is deterministic, not against native."""

import zlib
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseMetadata
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

LAYERS = 5
SPARSE = (3, 4)
VOCAB = 4096
STEPS = 8
# decode widths that are graph capture sizes: no padded rows, whose stock ops
# are not batch invariant
BATCHES = (1, 4, 16)
FALLBACK_SEQS = 20
FALLBACK_SHORT = 8
SEED = 0
ENV = {
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "1",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT": "1",
    # deterministic BF16 GEMMs (hipBLASLt) for the stock layers: the split-K
    # skinny and AITER tuned GEMMs reduce in a run-dependent order
    "VLLM_ROCM_USE_SKINNY_GEMM": "0",
    "VLLM_ROCM_USE_AITER_LINEAR": "0",
}


def _supported() -> bool:
    if not current_platform.is_rocm() or torch.accelerator.device_count() < 4:
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


pytestmark = [
    pytest.mark.distributed(num_gpus=4),
    pytest.mark.skipif(not _supported(), reason="needs 4 gfx950 GPUs"),
]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3Config

    layers = [0] * 3 + [1] * (LAYERS - 3)
    mx = dict(
        dtype="fp4",
        qscheme="per_group",
        ch_axis=-1,
        group_size=32,
        scale_type="float",
        scale_format="e8m0",
        scale_calculation_mode="even",
        round_method="half_even",
    )
    path = tmp_path_factory.mktemp("minimax_m3_mini")
    MiniMaxM3Config(
        text_config=dict(
            num_hidden_layers=LAYERS,
            vocab_size=VOCAB,
            moe_layer_freq=layers,
            sparse_attention_config=dict(
                use_sparse_attention=True,
                sparse_index_dim=128,
                sparse_num_index_heads=4,
                sparse_topk_blocks=16,
                sparse_block_size=128,
                sparse_disable_index_value=layers,
                sparse_score_type="max",
                sparse_init_block=0,
                sparse_local_block=1,
                sparse_attention_freq=layers,
            ),
        ),
        architectures=["MiniMaxM3SparseForCausalLM"],
        torch_dtype="bfloat16",
        # the checkpoint's: MXFP4 experts (A16W4 on this path), BF16 elsewhere
        quantization_config=dict(
            quant_method="quark",
            global_quant_config=dict(
                weight={**mx, "is_dynamic": False},
                input_tensors={**mx, "is_dynamic": True},
                output_tensors=None,
                bias=None,
            ),
            layer_quant_config={},
            layer_type_quant_config={},
            export=dict(kv_cache_group=[], pack_method="reorder"),
            exclude=[
                "lm_head",
                "re:.*self_attn.*",
                r"re:.*\.mlp\..*",
                r"re:.*block_sparse_moe\.gate$",
            ],
        ),
    ).save_pretrained(path)
    return str(path)


@torch.no_grad()
def _randomize(worker):
    """Weights at unit-activation scales; TP shards seeded per rank, replicated
    tensors (norms, router, bias) alike on every rank."""
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
    )
    from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_rocm

    rank = get_tensor_model_parallel_rank()
    model = worker.model_runner.get_model()

    def generator(name, sharded=True):
        key = f"{SEED}/{name}/{rank if sharded else 0}"
        return torch.Generator(device="cuda").manual_seed(zlib.crc32(key.encode()))

    for name, p in model.named_parameters():
        if ".experts." in name:
            continue
        replicated = p.ndim == 1 or ".gate." in name
        if p.ndim == 1:
            std = 0.1  # Gemma norms scale by 1 + w; the router bias
        elif "embed_tokens" in name:
            std = 1.0
        else:
            std = p.shape[1] ** -0.5
        g = generator(name, not replicated)
        p.copy_(torch.randn(p.shape, device=p.device, generator=g).mul(std))
    for name, b in model.named_buffers():
        # dummy loading leaves the sparse layers' FP8 KV scales random
        if name.endswith(("_k_scale", "_v_scale", "_q_scale", "_prob_scale")):
            b.fill_(1.0)
    for layer in model.model.layers:
        # q = 0, uniform attention: the FP8 KV cache rounds native's run-to-run
        # last-bit differences into whole FP8 steps, which softmax over q.k
        # would amplify; the FFN under test needs no attention content
        layer.self_attn.q_norm.weight.fill_(-1.0)
        if layer.layer_id not in SPARSE:
            continue
        if layer.layer_id == SPARSE[-1]:
            # its uniform attention would average V over the prompt, whose
            # FFNs ran stock with native's BF16 atomics: decode then depends
            # on run-to-run rounding; layer 3's seam carries the attention
            layer.self_attn.o_proj.weight.zero_()
        moe = layer.block_sparse_moe
        if moe.gate._bf16x3_weight is not None:
            moe.gate._bf16x3_weight.copy_(
                bf16x3_router_gemm_rocm.split_bf16x3(moe.gate.weight)
            )
        experts = moe.experts.routed_experts
        g = generator(f"experts.{layer.layer_id}")
        new = {}
        for key in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            dst = getattr(experts, key)
            # e8m0 2^-9..2^-7: unit gate/up pre-activations, MoE outputs on the
            # residual's scale (native's BF16 atomic sums round to its size)
            lo, hi = (118, 121) if "scale" in key else (0, 256)
            new[key] = torch.nn.Parameter(
                torch.randint(
                    lo, hi, dst.shape, dtype=torch.uint8, device="cuda", generator=g
                ),
                requires_grad=False,
            )
        converted = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
            layer=torch.nn.Module(),
            **new,
        )
        for key, src in zip(new, converted):
            getattr(experts, key).view(torch.uint8).copy_(src.view(torch.uint8))
    torch.accelerator.synchronize()


def _requests():
    set_random_seed(SEED)
    requests = []
    for n in BATCHES:
        prompts = [
            {"prompt_token_ids": torch.randint(VOCAB, (int(size),)).tolist()}
            for size in torch.randint(16, 48, (n,))
        ]
        traces = [torch.randint(VOCAB, (STEPS,)).tolist() for _ in range(n)]
        requests.append((n, prompts, traces))
    return requests


def _fallback_requests():
    """One batch of 4 requests more than ``FALLBACK_SEQS``, the last
    ``FALLBACK_SHORT`` admitted done after 3 tokens: a decode step of all
    ``FALLBACK_SEQS`` rows (also when the requests reach the engine over a few
    steps), then the waiting requests' prefills next to decodes, then 16 rows
    and fewer."""
    set_random_seed(SEED + 2)
    n = FALLBACK_SEQS + 4
    short = range(FALLBACK_SEQS - FALLBACK_SHORT, FALLBACK_SEQS)
    prompts = [
        {"prompt_token_ids": torch.randint(VOCAB, (int(size),)).tolist()}
        for size in torch.randint(16, 48, (n,))
    ]
    traces = [
        torch.randint(VOCAB, (3 if i in short else STEPS,)).tolist() for i in range(n)
    ]
    return [(n, prompts, traces)]


def _decode(llm, capture=False, requests=None):
    """Full-vocabulary logprobs of every decode step (the first output token
    comes from prefill, which runs stock), every token forced; ``capture``:
    also each sparse layer's inputs at the first decode step of each batch."""
    from vllm import SamplingParams

    rows = []
    for n, prompts, traces in requests or _requests():
        params = [
            SamplingParams(
                max_tokens=len(trace),
                temperature=0,
                logprobs=-1,
                trace_decode_token_ids=trace,
                detokenize=False,
                ignore_eos=True,
            )
            for trace in traces
        ]
        if capture:
            llm.collective_rpc(_install, args=(n,))
        outputs = llm.generate(prompts, params, use_tqdm=False)
        if capture:
            assert llm.collective_rpc(_stash, args=(n,)) == [list(SPARSE)] * 4
        for out, trace in zip(outputs, traces):
            assert list(out.outputs[0].token_ids) == trace
            for step in out.outputs[0].logprobs[1:]:
                rows.append([step[v].logprob for v in range(VOCAB)])
    return torch.tensor(rows)


# vLLM's Gemma norm on ROCm (an f32 weight: the native torch ops): the f32
# mean of squares in any order and its division within (H + 1) U, halved by
# the rsqrt; the add, rsqrt (2 U), two products and 1 + w
TORCH_NORM_ETA = (0.5 * (6144 + 1) + 6) * 2.0**-24


def _scratch(runner, s):
    """A launch's X, router logits (its parts summed as its route stage does)
    and route from its scratch; ``tags``: every pair carries the launch's."""
    from tests.models import mono_reference as R
    from tests.models.minimax_m3.amd_specialized_decode import test_ops as O
    from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn

    tag = int(runner.epoch[0])

    def pairs(name, n):
        p = runner.region(s, name)[: n * 8].view(torch.int32).view(n, 2)
        return p[:, 0].clone(), bool((p[:, 1] == tag).all())

    route, route_tags = pairs("route", s * 2 * ffn.TOPK)
    parts, part_tags = pairs("logit", s * ffn.EXPERTS * ffn.ROUTER_PARTS)
    parts = parts.view(torch.float32).view(s, ffn.EXPERTS, 4, -1).double()
    route = route.view(s, -1)
    return dict(
        x=R.words(runner.region(s, "xn"), s * ffn.HIDDEN, torch.bfloat16).view(s, -1),
        logits=O._f32_sum(O._f32_sum(parts)),
        ids=route[:, : ffn.TOPK].contiguous(),
        wts=route[:, ffn.TOPK :].contiguous().view(torch.float32),
        tags=route_tags and part_tags,
    )


def _launch_check(layer, part, residual, out, res_out, scratch):
    """One fused launch against vLLM's own ops on the launch's own inputs to
    each stage, every rank alike: the residual exactly; X within a step of the
    norm (and its flips within their expected count); the router logits and,
    on rows clear of a 4th/5th near-tie, the route within both paths' bounds;
    the output against vLLM's experts on X and the launch's route (an exact
    tie or a near-tie is then the same choice), within the bf16 rounding
    budget of the two paths' expert sums and the TP reduction."""
    from tests.models import mono_reference as R
    from tests.models.minimax_m3.amd_specialized_decode import test_ops as O
    from vllm.distributed import tensor_model_parallel_all_reduce as all_reduce
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        fused_allreduce_gemma_rms_norm,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.mono import ffn

    moe = layer.block_sparse_moe
    routed = moe.experts.routed_experts
    x, ids_m, wts_m = scratch["x"], scratch["ids"], scratch["wts"]
    want_x, want_res = fused_allreduce_gemma_rms_norm(
        part.clone(), residual.clone(), layer.post_attention_layernorm
    )
    m = dict(residual=torch.equal(res_out, want_res), tags=scratch["tags"])
    flips = O._norm_flips(x, want_x, O.MONO_NORM_ETA + TORCH_NORM_ETA)
    m["x_steps"], m["x_flips"], m["x_limit"] = flips

    logits, _ = moe.gate(x)
    err = O._router_err(x, moe.gate.weight)
    m["logits"] = O._ratio((scratch["logits"] - logits.double()).abs(), err)
    # the route stage on the launch's own logits: each path's scores within
    # SCORE_ERR, its weights within their own bound
    own = scratch["logits"].float()
    wts, ids = moe.experts.router.select_experts(
        x, own, routed.quant_method.topk_indices_dtype
    )
    m["shared"] = bool(
        (ids[:, ffn.TOPK] == ffn.SHARED).all() and (wts[:, ffn.TOPK] == 1).all()
    )
    clear = O._clear_choices(own, moe.e_score_correction_bias, O.SCORE_ERR)
    got_ids, order = ids_m.long().sort(-1)
    want_ids, want_order = ids[:, : ffn.TOPK].long().sort(-1)
    m["clear"], m["rows"] = int(clear.sum()), len(clear)
    m["route_ids"] = torch.equal(got_ids[clear], want_ids[clear])
    ek = O._weight_err(own[clear].double().gather(1, got_ids[clear]))
    got_w = wts_m.gather(1, order)[clear].double()
    want_w = wts[:, : ffn.TOPK].gather(1, want_order)[clear].double()
    m["route_w"] = (
        O._ratio((got_w - want_w).abs(), 2 * ek * want_w) if m["clear"] else 0
    )

    fids = torch.cat([ids_m.to(ids.dtype), ids[:, ffn.TOPK :]], 1)
    fwts = torch.cat([wts_m, wts[:, ffn.TOPK :]], 1)

    def experts(cols):
        return routed.forward_modular(
            x=x,
            topk_weights=fwts[:, cols].contiguous(),
            topk_ids=fids[:, cols].contiguous(),
        )

    want = all_reduce(experts(list(range(ffn.TOPK + 1))))
    terms = torch.stack([experts([k]) for k in range(ffn.TOPK + 1)]).double()
    bound, var = O._combine_budget(terms)
    total = terms.abs().sum(0)
    bound, var, total = (all_reduce(t.float()).double() for t in (bound, var, total))
    # the ranks' partials summed in f32 by both (3 adds, each within U of sum
    # |P|), then each rounded once to bf16: at most a step apart
    step = R.ulp(total)
    bound = bound + 6 * O.U * total + step
    var = var + 2 * step.square() / 12
    m.update(O._combine_stats(out, want, bound, var, total))
    return m


def _assert_launches(results, label):
    """Every launch's checks (``_launch_check``) on every rank; the route
    compared on at least half the rows, and the mean square and bias over
    all of rank 0's launches (the ranks' outputs are the same all-reduced
    sums, not independent samples) within the variance model."""
    from tests.models.minimax_m3.amd_specialized_decode import test_ops as O

    flat = [m for rank in results for m in rank]
    assert flat
    for m in flat:
        assert m["residual"] and m["tags"] and m["shared"]
        assert m["x_steps"] <= 1 and m["x_flips"] <= m["x_limit"]
        assert m["logits"] <= 1 and m["route_ids"] and m["route_w"] <= 1
        assert m["ratio"] <= 1
    assert sum(m["clear"] for m in flat) >= sum(m["rows"] for m in flat) / 2
    sums = ("n", "sq", "var", "b2var", "bias", "proj")
    merged = {k: sum(m[k] for m in results[0]) for k in sums}
    merged.update({k: max(m[k] for m in flat) for k in ("ratio", "ulps", "b2max")})
    flips = sum(m["x_flips"] for m in flat)
    print(f"{label}: {len(flat)} launches, X flips {flips}")
    O._assert_combine(merged, label)


def _backbone(worker):
    return worker.model_runner.get_model().model


def _install(worker, s):
    worker._ffn_cap, worker._ffn_hooks = {}, []
    for layer in _backbone(worker).layers:
        if layer.layer_id not in SPARSE:
            continue

        def pre(mod, args, lid=layer.layer_id):
            if args[1].shape[0] == s and lid not in worker._ffn_cap:
                worker._ffn_cap[lid] = [args[1].clone(), args[2].clone(), None]

        def post(mod, args, kwargs, out, lid=layer.layer_id):
            cap = worker._ffn_cap.get(lid)
            if cap is not None and cap[2] is None:
                cap[2] = out.clone()

        worker._ffn_hooks += [
            layer.register_forward_pre_hook(pre),
            layer.self_attn.register_forward_hook(post, with_kwargs=True),
        ]


def _stash(worker, s):
    for hook in worker._ffn_hooks:
        hook.remove()
    worker._ffn_caps = getattr(worker, "_ffn_caps", {})
    worker._ffn_caps[s] = worker._ffn_cap
    return sorted(worker._ffn_cap)


def _replay(worker):
    """Each captured sparse layer's FFN half, native and fused, on every rank."""
    from vllm.forward_context import set_forward_context
    from vllm.models.minimax_m3.amd.specialized_decode.runtime.executor import (
        FfnExecutor,
    )
    from vllm.models.minimax_m3.amd.specialized_decode.runtime.weights import (
        MonoLayerWeights,
    )

    layers = {
        layer.layer_id: layer
        for layer in _backbone(worker).layers
        if layer.layer_id in SPARSE
    }
    weights = {i: MonoLayerWeights.from_layer(layer) for i, layer in layers.items()}
    executor = FfnExecutor(list(layers.values()), weights, BATCHES, 4)
    results = []
    with torch.no_grad(), set_forward_context(None, worker.vllm_config):
        for s, caps in worker._ffn_caps.items():
            for i, (hidden, residual, part) in caps.items():
                layer = layers[i]
                _, residual = layer.input_layernorm(hidden, residual)
                slots = torch.arange(s, dtype=torch.int64, device=part.device)
                out, res_out = executor.runner(part, residual, weights[i], slots)
                scratch = _scratch(executor.runner, s)
                results.append(
                    _launch_check(layer, part, residual, out, res_out, scratch)
                )
    failed = executor.health().failed
    executor.close()
    return results, failed


def _engine(vllm_runner, model_dir, mono, graphs=False, in_process=None, **extra):
    """``in_process``: a MonkeyPatch to run the engine core in this process."""
    if in_process is not None:
        in_process.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    extra.setdefault("max_num_seqs", max(BATCHES))
    return vllm_runner(
        model_dir,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=4,
        kv_cache_dtype="fp8",
        block_size=128,
        attention_backend="TRITON_ATTN",
        attention_config={"indexer_kv_dtype": "fp8"},
        kernel_config={
            "enable_jit_warmup": False,
            "moe_backend": "aiter",
            "monokernel": mono,
        },
        enforce_eager=not graphs,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY" if graphs else "NONE"},
        enable_prefix_caching=False,
        enable_trace_replay=True,
        max_logprobs=-1,
        gpu_memory_utilization=0.1,
        **extra,
    )


@pytest.fixture(scope="module", autouse=True)
def _env():
    with pytest.MonkeyPatch.context() as mp:
        for name, value in ENV.items():
            mp.setenv(name, value)
        yield


@pytest.fixture(scope="module")
def native(vllm_runner, model_dir):
    """Native eager: each sparse layer's FFN inputs at the first decode step
    of each batch, and their replays."""
    with _engine(vllm_runner, model_dir, mono=False) as runner:
        runner.llm.collective_rpc(_randomize)
        _decode(runner.llm, capture=True)
        return runner.llm.collective_rpc(_replay)


@pytest.fixture(scope="module")
def mono_eager(vllm_runner, model_dir):
    """--monokernel eager, teacher-forced, the engine core in-process (each
    batch's requests all join its first step, so the steps are the same every
    run): the logprobs, the engagement and every launch's checks."""
    with (
        pytest.MonkeyPatch.context() as mp,
        _engine(vllm_runner, model_dir, mono=True, in_process=mp) as runner,
    ):
        llm = runner.llm
        llm.collective_rpc(_randomize)
        start = llm.collective_rpc(_count_launches)
        llm.collective_rpc(_record_forwards)
        llm.collective_rpc(_record_launches)
        logprobs = _decode(llm)
        after = llm.collective_rpc(_reset_state)
        fused, stocked = _engaged(llm, start, llm.collective_rpc(_forwards))
        launches = llm.collective_rpc(_check_launches)
    return SimpleNamespace(
        logprobs=logprobs, after=after, fused=fused, stocked=stocked, launches=launches
    )


def test_sparse_layers_match_native(native):
    """Every rank, both sparse layers, decode widths 1, 4 and 16: the fused
    launch on native's captured inputs against vLLM's own ops
    (``_launch_check``)."""
    assert not any(failed for _, failed in native)
    _assert_launches([results for results, _ in native], "replay")


def _specialized(worker):
    return worker.model_runner.model_state.backbone.specialized_decode


def _executor(worker):
    return _specialized(worker).executor


def _count_launches(worker):
    """From here on, the device's launches: its epoch, summed over the epoch
    resets."""
    executor = _executor(worker)
    clear = executor._clear
    executor.cleared = -_epoch(worker)

    def count_and_clear():
        executor.cleared += _epoch(worker)
        clear()

    executor._clear = count_and_clear
    return _engagement(worker)


def _engagement(worker):
    """Sampled steps, mono and stock forwards, each sparse layer's launches and
    the device's."""
    decode = _specialized(worker)
    executor = decode.executor
    return (
        executor.outputs,
        decode.mono_steps,
        decode.stock_steps,
        *executor.launches.values(),
        executor.cleared + _epoch(worker),
    )


def _engaged(llm, start, forwards):
    """Since ``start`` (``_count_launches``), alike on every rank: each forward
    ``_record_forwards`` saw routed as ``decide_step`` decides, and every
    sampled step but the stock ones (eager forwards or FULL graph replays)
    ran one launch a sparse layer on the device. Returns the fused steps and
    the stock forwards' rows, prefills and decodes."""
    from vllm.models.minimax_m3.amd.specialized_decode.mono.ffn import MAX_TOKENS

    assert all(rank == forwards[0] for rank in forwards)
    moved = {
        tuple(a - b for a, b in zip(after, before))
        for after, before in zip(llm.collective_rpc(_engagement), start)
    }
    assert len(moved) == 1
    outputs, mono, stock, *launches, device = moved.pop()
    stocked = []
    for tokens, _, prefills, decodes, fused, _ in forwards[0]:
        assert fused == (not prefills and len(tokens) <= MAX_TOKENS)
        if not fused:
            stocked.append((len(tokens), prefills, decodes))
    assert (mono, stock) == (len(forwards[0]) - len(stocked), len(stocked))
    assert launches == [mono] * len(SPARSE)
    assert device == len(SPARSE) * (outputs - stock)
    return outputs - stock, stocked


def _record_forwards(worker):
    """Every backbone forward from here on: its tokens, positions, prefill and
    decode counts, whether it ran the fused launches, and on decode steps
    EAGLE3's auxiliary states."""
    from vllm.forward_context import get_forward_context

    backbone = _backbone(worker)
    decode = backbone.specialized_decode
    name = backbone.layers[SPARSE[0]].self_attn.layer_name
    worker._forwards = []

    def pre(mod, args):
        worker._mono_steps = decode.mono_steps if decode else 0

    def post(mod, args, out):
        metadata = get_forward_context().attn_metadata[name]
        assert isinstance(metadata, MiniMaxM3SparseMetadata)
        prefills, decodes = metadata.num_prefills, metadata.num_decodes
        mono = decode is not None and decode.mono_steps > worker._mono_steps
        aux = out[1] if isinstance(out, tuple) else None
        if prefills or not aux:
            aux = None
        else:
            aux = torch.stack(aux).float().cpu().numpy().tobytes()
        worker._forwards.append(
            (args[0].tolist(), args[1].tolist(), prefills, decodes, mono, aux)
        )

    worker._forward_hooks = [
        backbone.register_forward_pre_hook(pre),
        backbone.register_forward_hook(post),
    ]


def _forwards(worker):
    for hook in worker._forward_hooks:
        hook.remove()
    return worker._forwards


def _record_launches(worker):
    """From here on, every fused launch as the module after it reads it: the
    residual and attention partial it took, its output and residual out, and
    its scratch."""
    backbone = _backbone(worker)
    executor = _executor(worker)
    layers = list(backbone.layers)
    taken: dict[int, list] = {}
    worker._launches = []

    def took(mod, args, out, lid):
        taken[lid] = [executor.launches[lid], out[1].clone(), None]

    def attended(mod, args, kwargs, out, lid):
        taken[lid][2] = out.clone()

    def read(mod, args, lid):
        took = taken.pop(lid, None)
        if took is None or executor.launches[lid] == took[0]:
            return
        _, residual, part = took
        scratch = _scratch(executor.runner, part.shape[0])
        out, res_out = args[0].clone(), args[1].clone()
        worker._launches.append((lid, part, residual, out, res_out, scratch))

    hooks = []
    for i, layer in enumerate(layers):
        if layer.layer_id not in SPARSE:
            continue
        lid = layer.layer_id
        reader = layers[i + 1].input_layernorm if i + 1 < len(layers) else None
        hooks += [
            layer.input_layernorm.register_forward_hook(partial(took, lid=lid)),
            layer.self_attn.register_forward_hook(
                partial(attended, lid=lid), with_kwargs=True
            ),
            (reader or backbone.norm).register_forward_pre_hook(partial(read, lid=lid)),
        ]
    worker._launch_hooks = hooks


def _check_launches(worker):
    """Every launch ``_record_launches`` saw, through ``_launch_check``."""
    from vllm.forward_context import set_forward_context

    for hook in worker._launch_hooks:
        hook.remove()
    layers = {layer.layer_id: layer for layer in _backbone(worker).layers}
    with torch.no_grad(), set_forward_context(None, worker.vllm_config):
        return [
            _launch_check(layers[lid], *tensors) for lid, *tensors in worker._launches
        ]


def _launch_states(worker):
    """Each recorded launch's output plus residual out, as EAGLE3 takes it."""
    return [
        (out + res_out).float().cpu().numpy().tobytes()
        for _, _, _, out, res_out, _ in worker._launches
    ]


def _reset_every(worker, steps):
    executor = _executor(worker)
    executor._reset_launches = steps * len(executor._layers)
    executor._since_reset = 0
    return executor.resets


def _reset_state(worker):
    executor = _executor(worker)
    return executor.resets, executor.health().failed


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "full_graphs"])
def test_model_matches_native(vllm_runner, model_dir, mono_eager, graphs, monkeypatch):
    """Teacher-forced decode with --monokernel. Eager: every launch matches
    vLLM's own ops (``_launch_check``). Under FULL graphs (async scheduling)
    the logprobs equal eager's bit for bit, the epoch resets every 3 sampled
    steps on every rank at the same steps, and no wait times out."""
    ref = mono_eager
    assert ref.fused >= len(BATCHES) * (STEPS - 1)
    assert all(prefills for _, prefills, _ in ref.stocked)
    assert not any(failed for _, failed in ref.after)
    if not graphs:
        assert all(len(rank) == len(SPARSE) * ref.fused for rank in ref.launches)
        _assert_launches(ref.launches, "eager")
        return
    with _engine(
        vllm_runner,
        model_dir,
        mono=True,
        graphs=True,
        in_process=monkeypatch,
        async_scheduling=True,
    ) as runner:
        llm = runner.llm
        llm.collective_rpc(_randomize)
        before = llm.collective_rpc(_reset_every, args=(3,))
        start = llm.collective_rpc(_count_launches)
        llm.collective_rpc(_record_forwards)
        logprobs = _decode(llm)
        after = llm.collective_rpc(_reset_state)
        fused, stocked = _engaged(llm, start, llm.collective_rpc(_forwards))
    assert fused == ref.fused
    assert torch.equal(logprobs, ref.logprobs)
    assert not any(failed for _, failed in after)
    resets = {n - b for (n, _), b in zip(after, before)}
    assert len(resets) == 1 and resets.pop() >= len(BATCHES) * STEPS // 3 - 1


def test_fallback_steps_match_native(vllm_runner, model_dir):
    """--monokernel eager over steps it hands to stock: decode steps of over
    16 rows and mixed prefill / decode steps, between plain decode steps that
    run the fused launches; every launch, on the state the stock steps left,
    matches vLLM's own ops (``_launch_check``)."""
    from vllm.models.minimax_m3.amd.specialized_decode.mono.ffn import MAX_TOKENS

    with _engine(
        vllm_runner, model_dir, mono=True, max_num_seqs=FALLBACK_SEQS
    ) as runner:
        llm = runner.llm
        llm.collective_rpc(_randomize)
        start = llm.collective_rpc(_count_launches)
        llm.collective_rpc(_record_forwards)
        llm.collective_rpc(_record_launches)
        _decode(llm, requests=_fallback_requests())
        forwards = llm.collective_rpc(_forwards)
        fused, stocked = _engaged(llm, start, forwards)
        launches = llm.collective_rpc(_check_launches)
    assert any(rows > MAX_TOKENS and not prefills for rows, prefills, _ in stocked)
    assert any(prefills and decodes for _, prefills, decodes in stocked)
    assert forwards[0][-1][4]
    assert all(len(rank) == len(SPARSE) * fused for rank in launches)
    _assert_launches(launches, "fallback")


def _arm_late_fault(worker, rank, ticks):
    _executor(worker).runner.inject.copy_(
        torch.tensor([rank + 1, ticks], dtype=torch.int32)
    )


def test_late_rank_fault_returns_no_tokens_from_full_graphs(
    vllm_runner, model_dir, capfd
):
    """During a replayed step, rank 2 fails 5 ms after the rest of the step's
    last launch, so the output rank is ahead of it; that step returns no
    token."""
    from vllm import SamplingParams
    from vllm.v1.engine.exceptions import EngineDeadError

    prompt = {"prompt_token_ids": list(range(1, 33))}
    params = SamplingParams(max_tokens=8, temperature=0, detokenize=False)
    with _engine(vllm_runner, model_dir, mono=True, graphs=True) as runner:
        llm = runner.llm
        assert len(llm.generate(prompt, params)[0].outputs[0].token_ids) == 8
        capfd.readouterr()
        llm.collective_rpc(_arm_late_fault, args=(2, 500_000))
        with pytest.raises(EngineDeadError):
            llm.generate(prompt, params)
    logs = "".join(capfd.readouterr())
    # the error-word assert trapped the queue ahead of the step's sampling
    assert "hipErrorLaunchFailure" in logs or "HSA_STATUS_ERROR_EXCEPTION" in logs


# EAGLE3 verify steps: 1 + k rows a request, up to 4 requests
SPEC_BATCHES = (1, 2, 3, 4)
# the draft's auxiliary states: after layer 1 (stock) and layers 3 and 4 (fused)
AUX_LAYERS = (2, 4, 5)
# prompt lengths apart by more than a request's decode positions
PROMPT_STRIDE = 16


@pytest.fixture(scope="module")
def draft_dir(tmp_path_factory):
    """A random one-layer EAGLE3 draft at the checkpoint draft's attention
    shapes and the mini model's vocabulary, its auxiliary states taken after
    layer 1 (stock) and layers 3 and 4 (fused)."""
    import json

    path = tmp_path_factory.mktemp("minimax_m3_eagle3_mini")
    config = dict(
        architectures=["LlamaForCausalLMEagle3"],
        model_type="llama",
        hidden_size=6144,
        intermediate_size=4096,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=64,
        head_dim=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=5000000,
        max_position_embeddings=4096,
        vocab_size=VOCAB,
        draft_vocab_size=VOCAB,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        fc_norm=True,
        norm_output=True,
        eagle_aux_hidden_state_layer_ids=list(AUX_LAYERS),
    )
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _epoch(worker):
    return int(_executor(worker).runner.epoch[0])


def _spec_requests():
    set_random_seed(SEED + 1)
    return [
        [
            {"prompt_token_ids": torch.randint(VOCAB, (PROMPT_STRIDE * i,)).tolist()}
            for i in range(1, n + 1)
        ]
        for n in SPEC_BATCHES
    ]


@pytest.mark.parametrize(
    "graphs, spec_tokens",
    [(True, 3), (False, 1), (False, 5)],
    ids=["full_graphs-k3", "eager-k1", "eager-k5"],
)
def test_eagle3_verify_steps_match_native(
    vllm_runner, model_dir, draft_dir, capfd, graphs, spec_tokens
):
    """EAGLE3 (random draft) with --monokernel: the verify steps of 1-4
    requests run the fused launches up to 16 rows (k=5: 18 and 24 rows run
    stock). Eager: every launch matches vLLM's own ops (``_launch_check``) and
    EAGLE3's auxiliary states after the fused layers are the launches' outputs
    plus residuals. Under FULL graphs a late rank fault during a verify step
    then returns no token."""
    from vllm import SamplingParams
    from vllm.models.minimax_m3.amd.specialized_decode.mono.ffn import MAX_TOKENS
    from vllm.v1.engine.exceptions import EngineDeadError

    greedy = SamplingParams(
        max_tokens=STEPS, temperature=0, logprobs=-1, detokenize=False, ignore_eos=True
    )
    recorded: list[list] = [[] for _ in range(4)]
    with _engine(
        vllm_runner,
        model_dir,
        mono=True,
        graphs=graphs,
        speculative_config={
            "method": "eagle3",
            "model": draft_dir,
            "num_speculative_tokens": spec_tokens,
        },
    ) as runner:
        llm = runner.llm
        llm.collective_rpc(_randomize)
        start = llm.collective_rpc(_count_launches)
        if not graphs:
            llm.collective_rpc(_record_launches)
        for prompts in _spec_requests():
            llm.collective_rpc(_record_forwards)
            llm.generate(prompts, greedy, use_tqdm=False)
            for rank, seen in zip(recorded, llm.collective_rpc(_forwards)):
                rank += seen
        fused_steps, stocked = _engaged(llm, start, recorded)
        wide = [rows for rows, prefills, _ in stocked if not prefills]
        assert bool(wide) == ((1 + spec_tokens) * max(SPEC_BATCHES) > MAX_TOKENS)
        if not graphs:
            states = llm.collective_rpc(_launch_states)[0]
            launches = llm.collective_rpc(_check_launches)
        else:
            capfd.readouterr()
            llm.collective_rpc(_arm_late_fault, args=(2, 500_000))
            with pytest.raises(EngineDeadError):
                llm.generate(_spec_requests()[0], greedy)
    if graphs:
        logs = "".join(capfd.readouterr())
        assert "hipErrorLaunchFailure" in logs or "HSA_STATUS_ERROR_EXCEPTION" in logs
        return
    _assert_launches(launches, f"EAGLE3 k={spec_tokens}")
    fused = [aux for *_, mono, aux in recorded[0] if mono]
    assert len(fused) == fused_steps and len(states) == len(SPARSE) * fused_steps
    # AUX_LAYERS[1:]: after layers 3 and 4, the fused ones
    for step, aux in enumerate(fused):
        aux = torch.frombuffer(bytearray(aux), dtype=torch.float32)
        wants = states[2 * step : 2 * step + 2]
        for got, want in zip(aux.view(len(AUX_LAYERS), -1)[1:], wants):
            want = torch.frombuffer(bytearray(want), dtype=torch.float32)
            assert torch.equal(got, want)
