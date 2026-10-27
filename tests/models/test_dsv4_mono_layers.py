# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4-Pro mono decode path through the real model: a five-layer
DeepSeek-V4-Pro at TP8 on CDNA4 with ``--monokernel``, random weights. At every
decode-only step each decoder layer runs the mono path, then vLLM's own stages,
each on the mono path's input of that stage (``MonoProbe._both``): unlike
``test_dsv4_mono_numerics.py``, which builds the ops itself, this runs
``DeepseekV4DecoderLayer.forward`` in the engine -- the model's hook, the
per-step path choice, the weights taken off the layer, the metadata, the
compressor and the indexer's top-k -- so a change to the decoder layer that the
kernels do not follow fails here. Fed the same inputs, the two outputs of a
stage differ by its arithmetic only, so every bound is derived
(``test_dsv4_mono_numerics``' parity bounds), not measured. Scope: eager steps
of one token a request; speculative decoding's multi-token rows and graph
replays are not covered.

The checkpoint's own first five layers (``compress_ratios`` 128, 128, 4, 128,
4; ``num_hash_layers`` 3) hold every variant the kernels build, each taking the
whole layer (K1 + K2):
- 0: hash-routed HCA, the first layer (K1 takes the embedding, no fold)
- 1: hash-routed HCA
- 2: hash-routed CSA (vLLM's indexer between K1 and K2)
- 3: bias-routed HCA
- 4: bias-routed CSA
A 32K ``max_model_len`` builds the HCA kernels 256 keys wide; the prompts
(under 8.3K) keep vLLM's c128a decode table at its 128-wide floor, so the
kernels read a table narrower than they are built for.
"""

import math
import zlib

import pytest
import torch

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.platforms import current_platform

MODEL = "deepseek-ai/DeepSeek-V4-Pro"
TP = 8
LOAD_FORMAT = "dsv4_mono_random"
TRUNCATED = {
    "num_hidden_layers": 5,
    # one entry a layer, the MTP layer's included
    "compress_ratios": [128, 128, 4, 128, 4, 0],
}
LAYERS = (0, 1, 2, 3, 4)
STEPS = 5  # decode steps after the prefill's token


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


@torch.no_grad()
def _fill(name: str, t: torch.Tensor) -> None:
    """A random value for every weight, seeded by its name (alike on every
    rank and in every run), in the ranges the checkpoint's formats hold."""
    if t.device.type == "meta":
        return  # materialized and filled by the dummy loader's own path
    g = torch.Generator(device=t.device).manual_seed(zlib.crc32(name.encode()))

    def randn(scale=1.0, shift=0.0):
        return torch.randn(t.shape, generator=g, device=t.device) * scale + shift

    if t.dtype in (torch.uint8, torch.float8_e8m0fnu) and "scale" in name:
        # E8M0 2^-12 .. 2^-8: one a 128 x 128 FP8 block, one a 32 MXFP4
        # values. Larger random layers make the stack chaotic: atomics' order
        # alone moves vLLM's own logits run to run.
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(115, 120, u8.shape, generator=g, device=t.device))
    elif t.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(0, 256, u8.shape, generator=g, device=t.device))
    elif t.dtype == torch.float8_e4m3fn:
        t.copy_(randn().clamp(-6, 6))
    elif name.endswith("tid2eid"):
        # a hash layer's table: distinct experts (of 384) a token
        ids = torch.rand(t.shape[0], 384, generator=g, device=t.device)
        t.copy_(ids.argsort(1)[:, : t.shape[1]])
    elif not torch.is_floating_point(t):
        return
    elif "norm" in name:
        t.copy_(randn(0.1, 1.0))
    elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
        t.copy_(randn(0.5).abs() + 0.5)
    elif name.endswith(("hc_attn_base", "hc_ffn_base", "attn_sink")):
        t.copy_(randn(0.1))
    elif name.endswith("embed_tokens.weight"):
        t.copy_(randn())
    else:
        t.copy_(randn(0.02))


@register_model_loader(LOAD_FORMAT)
class RandomWeightLoader(DummyModelLoader):
    """Dummy loading with weights in the checkpoint's ranges, before post-load
    processing, so every processed copy follows (plain dummy weights leave the
    E8M0 scales zero, and both paths would agree on zeros)."""

    def load_weights(self, model, model_config) -> None:
        from vllm.distributed import get_tensor_model_parallel_world_size

        super().load_weights(model, model_config)
        for name, t in model.state_dict().items():
            _fill(name, t)
        # the routed experts' intermediate past this rank's share, if padded,
        # is the loader's zero padding: the kernels skip it
        inter = model_config.hf_text_config.moe_intermediate_size
        inter //= get_tensor_model_parallel_world_size()
        for m in model.modules():
            w13, w2 = getattr(m, "w13_weight", None), getattr(m, "w2_weight", None)
            if w13 is None or w2 is None or w13.device.type == "meta":
                continue
            padded = w13.shape[1] // 2
            if padded > inter:
                w13, w2 = w13.data.view(torch.uint8), w2.data.view(torch.uint8)
                for half in (0, padded):
                    w13[:, half + inter : half + padded] = 0
                w2[..., inter // 2 :] = 0


class MonoProbe:
    """Worker extension (``worker_extension_cls``): it also brings this module,
    and so the loader, into the workers."""

    def _decoder_layers(self) -> dict:
        from vllm.model_executor.models.utils import extract_layer_index
        from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer

        model = self.model_runner.model  # type: ignore[attr-defined]
        return {
            extract_layer_index(m.attn.prefix): m
            for m in model.modules()
            if isinstance(m, DeepseekV4DecoderLayer)
        }

    def capture_steps(self) -> None:
        """At every decode-only step each decoder layer runs the mono path,
        then vLLM's stages on the mono path's own stage inputs (``_both``)."""
        from vllm.forward_context import get_forward_context

        self._captured: list = []

        def decode_only(layer) -> bool:
            md = get_forward_context().attn_metadata
            if not isinstance(md, dict):
                return False
            swa = md.get(layer.attn.swa_cache_layer.prefix)
            return swa is not None and swa.num_prefills == 0

        for i, layer in self._decoder_layers().items():

            def forward(x, positions, input_ids, post_mix=None, res_mix=None,
                        residual=None, i=i, layer=layer,
                        run=layer.forward):  # fmt: skip
                if not decode_only(layer):
                    return run(x, positions, input_ids, post_mix, res_mix, residual)
                return self._both(i, layer, run, x, positions, input_ids,
                                  post_mix, res_mix, residual)  # fmt: skip

            layer.forward = forward

    def _both(self, i, layer, run, x, positions, input_ids, post_mix, res_mix,
              residual):  # fmt: skip
        """The layer by the mono path, then by vLLM's stages each on the mono
        path's input of that stage: the attention seam on the layer's inputs,
        the attention on the kernels' normed rows (a CSA layer's top-k forced
        to the kernels' own: the indexer reads each path's qr), the FFN seam on
        the kernels' attention output, the MoE on the kernels' FFN input. A
        stage's two outputs then differ by its own arithmetic only. vLLM's
        attention rewrites the step's KV record with its own: later steps read
        either path's records alike."""
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.models.deepseek_v4.amd import mono_decode
        from vllm.models.deepseek_v4.amd.mono.dims import TOPK, Dims
        from vllm.models.deepseek_v4.amd.mono.layer import (
            peer_half_bytes,
            scratch_layout,
        )
        from vllm.models.deepseek_v4.amd.mono.stages.allreduce import region_bytes
        from vllm.models.deepseek_v4.amd.mono.stages.moe import ROUTE_COPIES

        M, attn, first = x.shape[0], layer.attn, residual is None
        ins = [None if t is None else t.clone() for t in (x, residual, post_mix,
                                                          res_mix)]  # fmt: skip
        before = mono_decode._runner
        e = -1 if before is None else int(before.epoch[0])
        out = run(x, positions, input_ids, post_mix, res_mix, residual)
        runner = mono_decode._runner
        assert runner is not None and int(runner.epoch[0]) == max(e, 0) + 1, i
        e, tp = max(e, 0), runner.tp
        csa = attn.indexer is not None
        width = 0 if csa else layer.mono.hca_width
        layout, scratch = scratch_layout(M, tp, width), runner.scratch(M, width)

        def region(name, n):
            off = layout[name][0]
            return scratch[off : off + n]

        # the kernels' stage outputs: the attention's (the peer partials of
        # every rank, summed in rank order as the FFN seam reads them) and the
        # MoE's route
        half = peer_half_bytes(tp) * (e & 1)
        pairs = runner.peer.bytes[half : half + region_bytes(M, Dims(tp))]
        part = pairs.view(torch.int32).view(tp + 1, M, -1, 2)[:tp, ..., 0]
        part = part.contiguous().view(torch.bfloat16).view(tp, M, -1).float()
        attn_m = part.sum(0).to(torch.bfloat16)
        route = region("route", ROUTE_COPIES * M * 2 * TOPK * 8).view(torch.int32)
        route = route.view(ROUTE_COPIES, M, 2 * TOPK, 2)
        assert (route[..., 1] == 2 * e + 2).all(), (i, "route tags")
        m = dict(
            res_mid=runner.res_mid[:M].clone(),
            post_a=runner.post_a[:M].clone(),
            comb_a=runner.comb_a[:M].clone(),
            normed=runner.normed[:M].clone(),
            attn=attn_m,
            xk=region("normed", M * 7168 * 2).view(torch.bfloat16).view(M, -1).clone(),
            route=route[0, :, :TOPK, 0].clone(),
            out=[t.clone() for t in out[:4]],
        )
        mono_topk = attn.topk_indices_buffer[:M].clone() if csa else None

        v: dict = {}
        if first:
            v["normed"], v["post_a"], v["comb_a"] = layer.hc_pre(
                ins[0], layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                norm_weight=layer.attn_norm.weight,
                norm_eps=layer.attn_norm.variance_epsilon,
            )  # fmt: skip
            v["res_mid"] = ins[0]
        else:
            v["res_mid"], v["post_a"], v["comb_a"], v["normed"] = self._seam(
                layer, "attn", *ins
            )
        if csa:
            mqa = attn.forward_mqa

            def forced(q, kv, pos, o):
                attn.topk_indices_buffer[:M].copy_(mono_topk)
                return mqa(q, kv, pos, o)

            attn.forward_mqa = forced
        try:
            v["attn"] = attn(positions, m["normed"].clone(), None)
        finally:
            if csa:
                del attn.forward_mqa
        v["res_out"], v["post_f"], v["comb_f"], v["xk"] = self._seam(
            layer, "ffn", attn_m, m["res_mid"], m["post_a"].view(M, -1, 1),
            m["comb_a"],
        )  # fmt: skip
        router = layer.ffn.experts.router
        select = router.select_experts

        def routed(*a, **k):
            w, ids = select(*a, **k)
            v["route"] = ids.clone()
            return w, ids

        shared: list = []
        hook = layer.ffn.shared_experts.register_forward_hook(
            lambda mod, a, o: shared.append(o)
        )
        router.select_experts = routed
        try:
            v["x"] = layer.ffn(m["xk"].clone(), input_ids)
        finally:
            del router.select_experts
            hook.remove()
        v["shared"] = tensor_model_parallel_all_reduce(shared[0].float())
        if self.rank == 0:  # type: ignore[attr-defined]
            self._captured.append(
                dict(
                    layer=i,
                    hashed=layer.ffn.gate.tid2eid is not None,
                    ins=[None if t is None else t.cpu() for t in ins],
                    m={k: [t.cpu() for t in t_] if isinstance(t_, list)
                       else t_.cpu() for k, t_ in m.items()},
                    v={k: t_.cpu() for k, t_ in v.items()},
                )
            )  # fmt: skip
        return out

    @staticmethod
    def _seam(layer, sub, x, residual, post, comb):
        return layer.mhc_fused_post_pre(
            x.clone(), residual.clone(), post.clone(), comb.clone(),
            getattr(layer, f"hc_{sub}_fn"), getattr(layer, f"hc_{sub}_scale"),
            getattr(layer, f"hc_{sub}_base"), layer.rms_norm_eps, layer.hc_eps,
            layer.hc_eps, layer.hc_post_alpha, layer.hc_sinkhorn_iters,
            norm_weight=getattr(layer, f"{sub}_norm").weight,
            norm_eps=getattr(layer, f"{sub}_norm").variance_epsilon,
        )  # fmt: skip

    def save_capture(self, path: str) -> None:
        if self.rank != 0:  # type: ignore[attr-defined]
            return
        params = {}
        for i, layer in self._decoder_layers().items():
            p = {k: getattr(layer, k).detach().cpu() for k in (
                "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn",
                "hc_ffn_scale", "hc_ffn_base")}  # fmt: skip
            for sub in ("attn", "ffn"):
                p[f"{sub}_norm"] = getattr(layer, f"{sub}_norm").weight.detach().cpu()
            p["gate"] = layer.ffn.gate.weight.detach().cpu()
            bias = layer.ffn.gate.e_score_correction_bias
            p["bias"] = None if bias is None else bias.detach().cpu()
            p["eps"] = (layer.rms_norm_eps, layer.hc_eps, layer.hc_sinkhorn_iters,
                        layer.attn_norm.variance_epsilon)  # fmt: skip
            params[i] = p
        torch.save((self._captured, params), path)

    def mono_layers(self) -> dict:
        from vllm.models.deepseek_v4.amd import mono_decode

        layers = {
            i: None if m.mono is None else m.mono._weights is not None
            for i, m in self._decoder_layers().items()
        }
        runner = mono_decode._runner
        return {
            "layers": layers,
            "epoch": -1 if runner is None else int(runner.epoch[0]),
        }


def _decode(vllm_runner, monkeypatch, prompts, path: str) -> list:
    """Greedy decode of ``prompts`` with the mono path on, both paths captured
    at every decode-only step (``MonoProbe``) into ``path``; the workers' mono
    layers."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    # the deployment the mono path takes (``MonoDecodeLayer.create``)
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS", "0")
    monkeypatch.setenv("VLLM_ROCM_FP8_PADDING", "1")
    with vllm_runner(
        MODEL,
        tensor_parallel_size=TP,
        load_format=LOAD_FORMAT,
        hf_overrides=TRUNCATED,
        worker_extension_cls=f"{__name__}.MonoProbe",
        moe_backend="aiter",
        monokernel=True,
        kv_cache_dtype="fp8",
        # the serving page (vllm_runner's default 16 holds no C128 row)
        block_size=256,
        tokenizer_mode="deepseek_v4",
        # eager: the probe runs vLLM's stages at every step (the kernels take
        # eager steps as they take FULL graph replays)
        enforce_eager=True,
        max_model_len=32768,
        max_num_seqs=len(prompts),
        # the steps mixing prefill and decode take vLLM's path (the runner
        # turns chunked prefill off: a step's tokens must cover the model
        # length)
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.15,
    ) as runner:
        runner.llm.collective_rpc("capture_steps")
        params = SamplingParams(temperature=0.0, max_tokens=STEPS + 1)
        runner.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
        runner.llm.collective_rpc("save_capture", args=(path,))
        return runner.llm.collective_rpc("mono_layers")


def _check(records: list, params: dict) -> dict:
    """Every layer's stages, the kernels against vLLM on the same inputs ->
    each check's worst over its bound (``within``: <= 1) or its derived
    expectation (``units``: <= ``UNITS_OK``)."""
    from .test_dsv4_mono_numerics import (
        ATTN_GRIDS,
        BF16_RN,
        MID_FLIP,
        MOE_BF16,
        TWO_GRIDS,
        _rows_rel,
        seam_parity,
    )

    within: dict = {}
    units: dict = {}
    moe: dict = {}

    def worst(table, key, value):
        table[key] = max(table.get(key, 0.0), value)

    for r in records:
        i, ins, m, v = r["layer"], r["ins"], r["m"], r["v"]
        p = params[i]
        x, res, post, comb = ins
        first = res is None
        for sub, args, got, want in (
            ("attn", (x, x if first else res, post, comb, first),
             (m["res_mid"], m["post_a"], m["comb_a"], m["normed"]),
             (v["res_mid"], v["post_a"], v["comb_a"], v["normed"])),
            ("ffn", (m["attn"], m["res_mid"], m["post_a"], m["comb_a"], False),
             m["out"][1:] + [m["xk"]],
             (v["res_out"], v["post_f"], v["comb_f"], v["xk"])),
        ):  # fmt: skip
            for k, ratio in seam_parity(p, sub, *args, got, want).items():
                worst(within, (i, sub, k), ratio)
        # the attention: four activations on different grids, gain 1 each
        rel = _rows_rel(m["attn"], v["attn"]) / (math.sqrt(ATTN_GRIDS) * TWO_GRIDS)
        worst(units, (i, "attention"), rel.max().item())
        # the MoE: a hash layer's picks are the token's; a bias-routed layer's
        # as sets, but where vLLM's f64 margin is within both paths' score
        # errors (``test_dsv4_mono_numerics._route_check``)
        same = m["route"].long().sort(1).values == v["route"].long().sort(1).values
        same = same.all(1)
        if r["hashed"]:
            assert same.all(), (i, "hash route")
        else:
            _route_ties(p, m["xk"], m["route"], v["route"], same, i)
        # the kernels round the SwiGLU to bf16 before its MXFP8, AITER's tuned
        # stage 1 does not (``MID_FLIP``); the shared expert's two roundings
        # are on different grids (sqrt(3) of ``TWO_GRIDS``: the input's reach
        # both SwiGLU operands); the partial sums' bf16 roundings on each path
        # (``MOE_BF16``)
        x_v = v["x"].double()[same]
        d = (m["out"][0].double()[same] - x_v).square().sum(1)
        e = (MID_FLIP**2 * (x_v - v["shared"].double()[same]).square().sum(1)
             + 3 * (TWO_GRIDS * v["shared"].double()[same]).square().sum(1)
             + 2 * MOE_BF16 * (BF16_RN * x_v).square().sum(1))  # fmt: skip
        acc = moe.setdefault(i, [0.0, 0.0])
        acc[0] += d.sum().item()
        acc[1] += e.sum().item()
    # the MoE's flips are a few hundred a token, their energy in the largest
    # binades: the expectation holds over the layer's tokens, not each one
    for i, (d, e) in moe.items():
        units[(i, "MoE")] = math.sqrt(d / e)
    return {"within": within, "units": units}


def _route_ties(p, x, route_m, route_v, same, layer) -> None:
    """Rows the two paths route differently: each a near-tie of score + bias
    (f64), within both paths' score errors: the kernels' logits
    (``test_dsv4_mono_io``'s router bound, 242 u |x| |w|), vLLM's fp32 router
    GEMM in any order (7168 u |x| |w|), through sqrt(softplus)'s slope."""
    from .test_dsv4_mono_numerics import HIDDEN, U

    if same.all():
        return
    xd, gw = x.double()[~same], p["gate"].double()
    logits = xd @ gw.T
    sp = torch.nn.functional.softplus(logits)
    sel = sp.sqrt() + p["bias"].double()
    top = sel.topk(route_m.shape[1] + 1, dim=1).values
    margin = top[:, -2] - top[:, -1]
    dl = (33 * HIDDEN // 1024 + 11 + HIDDEN) * U * (xd.abs() @ gw.abs().T)
    tie = 2 * (torch.sigmoid(logits) / (2 * sp.sqrt()) * dl + 4 * U * sel.abs())
    assert (margin < tie.amax(1)).all(), (layer, "route", margin, tie.amax(1))


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
def test_mono_decode_matches_the_model(vllm_runner, monkeypatch, tmp_path):
    if torch.accelerator.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    pytest.importorskip("flydsl")
    g = torch.Generator().manual_seed(0)
    # contexts past the CSA layers' 1024 top-k (4 tokens a compressed entry),
    # so the indexer selects
    jitter = torch.randint(256, (8,), generator=g).tolist()
    lens = [4400 + 512 * i + r for i, r in enumerate(jitter)]
    prompts = [torch.randint(1000, 60000, (n,), generator=g).tolist() for n in lens]

    probe = _decode(vllm_runner, monkeypatch, prompts, f"{tmp_path}/capture")
    # every layer took the mono path on every rank, at every decode step
    for rank in probe:
        assert rank["layers"] == dict.fromkeys(LAYERS, True), rank
        assert rank["epoch"] >= len(LAYERS) * STEPS, rank
        assert rank["epoch"] % len(LAYERS) == 0, rank
    records, params = torch.load(f"{tmp_path}/capture", weights_only=False)
    assert {r["layer"] for r in records} == set(LAYERS), records
    assert len(records) >= len(LAYERS) * STEPS, len(records)
    from .test_dsv4_mono_numerics import UNITS_OK

    out = _check(records, params)
    bad = {k: r for k, r in out["within"].items() if not r <= 1.0}
    bad |= {k: r for k, r in out["units"].items() if not r <= UNITS_OK}
    assert not bad, (bad, out)
