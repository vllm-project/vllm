# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode path through the real model: a five-layer
DeepSeek-V4.1-Flash at TP2 and TP4 on CDNA4, and at TP4 on gfx942, served with
``--monokernel`` on random weights. Every decode step of every layer also runs
vLLM's path on the step's own inputs and caches, and the two must agree stage
by stage (the attention output, the FFN seam, the MoE) within what each path's
own rounding allows. Unlike
``test_dsv41_mono_numerics.py``, which checks the kernels against float64
references one stage at a time, this runs ``DeepseekV4DecoderLayer.forward``
as the engine does -- the model's hooks, the per-step path choice, the weights
taken off the layer, the metadata and the indexer's top-k -- so a change to the
decoder layer that the kernels do not follow fails here.

On gfx942 the indexer is vLLM's FP8 indexer, because the MXFP4 indexer cache
needs CDNA4, and the kernels read the copies of the weights that the model
makes when it loads them. A decode step of 8 rows there takes two GEMV passes
of 6 rows, the second one padded, and a step of 42 rows, the most that the
gfx942 kernels take, takes seven.

Every layer takes the whole-layer path (K1 + K2); the five hold each variant:
- 0: the first layer, sliding window only (K1 takes the embedding settled)
- 1: ratio-2 KV and index source (vLLM's compressor and indexer between K1
  and K2)
- 2: ratio-2 consumer with an Engram (K1 takes the residual settled)
- 3: ratio-1 KV, index and candidate source
- 4: ratio-1 index source over 3's candidates and KV

On CDNA4 the same run with DSpark (5 drafts, 8 requests: 48-row verify steps)
also runs the three draft layers whole on the draft's own runner, and a run in
FULL CUDA graphs, its 5 requests padded to an 8-row graph, must give the eager
run's tokens.
"""

import zlib

import pytest
import torch

from tests.models import mono_reference as R
from tests.models.utils import check_logprobs_close
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.platforms import current_platform

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
LOAD_FORMAT = "dsv41_mono_random"
TRUNCATED = {
    "num_hidden_layers": 5,
    # one entry a layer, the three MTP layers' included
    "compress_ratios": [0, 2, 2, 1, 1, 0, 0, 0],
    "kv_source_layer_ids": [1, 3],
    # 4: an indexer of its own over 3's candidates (a REINDEX layer)
    "index_source_layer_ids": [1, 3, 4],
    "candidate_source_layer_id": 3,
    # an Engram on a whole layer (K1 then takes its residual settled), its
    # tables shrunk: 24 prime-sized ranges from 1000
    "engram_layer_ids": [2],
    "engram_num_embeddings": [32768],
    "engram_vocab_size": 1000,
    "dspark_target_layer_ids": [2, 3, 4],
}
HIDDEN = 5120
# DeepseekV4DecoderLayer.forward's positional arguments
FORWARD_ARGS = ("x", "positions", "input_ids", "pre_mix", "post_mix", "res_mix",
                "residual")  # fmt: skip
WHOLE, FFN = (0, 1, 2, 3, 4), ()
DRAFT = (5, 6, 7)  # the DSpark draft's layers
STEPS = 5  # decode steps after the prefill's token
DSPARK_TOKENS = 5
DSPARK = {"method": "dspark", "num_speculative_tokens": DSPARK_TOKENS}


def _truncate(config):
    """``TRUNCATED`` as an override that the DSpark draft's config also takes
    (a dict override reaches the target's only)."""
    for key, value in TRUNCATED.items():
        setattr(config, key, value)
    return config


def _arch() -> str:
    """The GPU kind that the mono kernels build for, "cdna4" or "gfx942", or
    an empty string on any other GPU."""
    if not current_platform.is_rocm():
        return ""
    from vllm.platforms.rocm import get_cdna_version, on_gfx942

    if get_cdna_version() == 4:
        return "cdna4"
    return "gfx942" if on_gfx942() else ""


# The (tensor parallel size, decode rows) cases of each GPU. The gfx942 kernels
# build for TP4 only. 8 rows are two GEMV passes of 6 rows there, the second
# one padded, and 42 rows are the most that they take.
CASES = {"cdna4": [(2, 8), (4, 8)], "gfx942": [(4, 8), (4, 42)]}


@torch.no_grad()
def _fill(name: str, t: torch.Tensor) -> None:
    """A random value for every weight, seeded by its name (alike on every
    rank and in every run), in the ranges the checkpoint's formats hold."""
    if t.device.type == "meta":
        return  # materialized and filled by the dummy loader's own path
    g = torch.Generator(device=t.device).manual_seed(zlib.crc32(name.encode()))

    def randn(scale=1.0, shift=0.0):
        return torch.randn(t.shape, generator=g, device=t.device) * scale + shift

    if t.dtype == torch.uint8 and "scale" in name:
        # E8M0 2^-12 .. 2^-8, alike over each 32 rows as in the checkpoint (the
        # loader folds per-row scales into 32 x 32 blocks only then). Larger
        # random layers make the stack chaotic: atomics' order alone moves
        # vLLM's own logits run to run.
        rows = t.shape[-2] if t.dim() > 1 else 1
        shape = (*t.shape[:-2], -(-rows // 32), t.shape[-1]) if t.dim() > 1 else t.shape
        e8m0 = torch.randint(115, 120, shape, generator=g, device=t.device)
        if t.dim() > 1:
            e8m0 = e8m0.repeat_interleave(32, dim=-2)[..., :rows, :]
        t.copy_(e8m0)
    elif t.dtype in (torch.uint8, torch.float4_e2m1fn_x2):
        u8 = t.view(torch.uint8)
        u8.copy_(torch.randint(0, 256, u8.shape, generator=g, device=t.device))
    elif t.dtype == torch.float8_e4m3fn:
        t.copy_(randn().clamp(-6, 6))
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
        # the routed experts' intermediate past this rank's share is the
        # loader's zero padding (576 -> 640 at TP4): the kernels skip it
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
        from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

        runner = self.model_runner  # type: ignore[attr-defined]
        models = [runner.model, runner.get_draft_model()]
        return {
            m.attn.layer_id: m
            for model in models
            if model is not None
            for m in model.modules()
            if isinstance(m, DeepseekV4DecoderLayer)
        }

    def capture_steps(self, compare: bool) -> None:
        """Count each decoder layer's decode-only steps (the steps the mono path
        takes). With ``compare``, each such step of a mono layer then also runs
        vLLM's path on the step's own inputs and caches, so every stage compares
        on equal inputs. Two results of the mono step are forced into vLLM's:
        an index source's top-k (a near-tie of index scores at the 512th entry
        is a discontinuity) and the attention output (compared on its own, it
        would otherwise carry its rounding into the FFN seam). The MoE runs
        again on the mono step's own MoE input, its rows tiled to MAX_ROWS: at
        some row counts AITER's fused stage 1 takes another MXFP8 scale rule
        (https://github.com/ROCm/aiter/issues/6548), which this run sees as a
        1.6-1.9% difference of vLLM's own MoE output at 3 rows. vLLM's path
        writes the caches last, which no comparison depends on."""
        from vllm.forward_context import get_forward_context
        from vllm.models.deepseek_v41.amd import mono_decode
        from vllm.models.deepseek_v41.amd.mono.layer import (
            peer_half_bytes,
            scratch_layout,
        )

        self._steps: dict[int, int] = {}
        self._records: list = []
        self._consts: dict = {}
        st: dict = {}

        def decode_only(layer) -> bool:
            if torch.cuda.is_current_stream_capturing():
                return False
            md = get_forward_context().attn_metadata
            if not isinstance(md, dict):
                return False
            swa = md.get(layer.attn.swa_cache_layer.prefix)
            return swa is not None and swa.num_prefills == 0

        def pre_hook(layer_id, layer):
            def hook(module, args, kwargs):
                st.clear()
                if not decode_only(layer):
                    return
                self._steps[layer_id] = self._steps.get(layer_id, 0) + 1
                if compare and layer.mono is not None:
                    clone = lambda v: v.clone() if torch.is_tensor(v) else v  # noqa: E731
                    st.update(
                        layer=layer_id,
                        mode="mono",
                        args=[clone(a) for a in args],
                        kwargs={k: clone(v) for k, v in kwargs.items()},
                    )

            return hook

        def mono_outputs(layer, rows: int) -> dict:
            """The step's MoE input and attention output (the ranks' wo_b
            partials summed in rank order), as the kernels left them."""
            runner = (
                mono_decode._draft_runner if layer.mono.draft else mono_decode._runner
            )
            tp = runner.tp
            lay = scratch_layout(
                rows, tp, runner.experts, runner.topk, runner.ug_groups
            )
            off = lay["normed"][0]
            moe_in = runner.scratch(rows)[off : off + rows * HIDDEN * 2]
            ep = int(runner.epoch[0]) - 1
            n = tp * rows * HIDDEN // 2
            peer = runner.peer.bytes[(ep & 1) * peer_half_bytes(tp) :]
            parts = R.pairs(peer, n, 2 * ep + 2).view(torch.bfloat16)
            parts = parts.view(tp, rows, HIDDEN).float()
            attn = parts[0]
            for r in range(1, tp):
                attn = attn + parts[r]
            return dict(
                moe_in=moe_in.view(torch.bfloat16).view(rows, HIDDEN).clone(),
                attn=attn.to(torch.bfloat16),
                res_mid=runner.res_mid[:rows].clone(),
                post_a=runner.post_a[:rows].clone(),
                comb_a=runner.comb_a[:rows].clone(),
                pre_a=runner.pre_a[:rows].clone(),
            )

        def hook(module, args, kwargs, out):
            if st.get("mode") != "mono":
                return
            layer_id, layer = st["layer"], module
            rows = out[0].shape[0]
            got = mono_outputs(layer, rows)
            st["attn"] = got["attn"]
            st["mode"] = "vllm"
            mono, layer.mono = layer.mono, None
            try:
                want = layer.forward(*st["args"], **st["kwargs"])
            finally:
                layer.mono = mono
            st["mode"] = "moe"
            fc = get_forward_context()
            kw = dict(zip(FORWARD_ARGS, st["args"])) | st["kwargs"]
            ids = kw["input_ids"]
            tile = torch.arange(mono_decode.MAX_ROWS, device=out[0].device) % rows
            pad = fc.is_padding
            fc.is_padding = None if pad is None else pad[:rows][tile].contiguous()
            try:
                moe = layer.ffn(
                    got["moe_in"][tile].contiguous(), None if ids is None else ids[tile]
                )[:rows]
            finally:
                fc.is_padding = pad
            cpu = lambda ts: [t.cpu() for t in ts]  # noqa: E731
            self._records.append(
                dict(
                    layer=layer_id,
                    mono=cpu(out),
                    vllm=cpu(want),
                    attn=cpu((got["attn"], st["attn_vllm"])),
                    moe=cpu((out[0], moe)),
                    moe_in=got["moe_in"].cpu(),
                    res_mid=got["res_mid"].cpu(),
                    post_a=got["post_a"].cpu(),
                    comb_a=got["comb_a"].cpu(),
                    pre_a=got["pre_a"].cpu(),
                    moe_in_vllm=st["moe_in_vllm"].cpu(),
                    inputs=[
                        None if t is None else t.cpu()
                        for t in (
                            kw.get("x"),
                            kw.get("residual"),
                            kw.get("post_mix"),
                            kw.get("res_mix"),
                        )
                    ],
                )
            )
            if layer_id not in self._consts:
                gate = layer.ffn.gate
                self._consts[layer_id] = dict(
                    gate=gate.weight.float().cpu(),
                    bias=gate.e_score_correction_bias.float().cpu(),
                    topk=layer.ffn.n_activated_experts,
                    ffn_norm=layer.ffn_norm.weight.float().cpu(),
                    seams={
                        sub: [
                            getattr(layer, f"hc_{sub}_{n}").float().cpu()
                            for n in ("fn", "scale", "base")
                        ]
                        for sub in ("attn", "ffn")
                    },
                    eps=(layer.rms_norm_eps, layer.hc_eps, layer.hc_sinkhorn_iters),
                )
            st.clear()

        def ffn_hook(module, args, kwargs):
            if st.get("mode") == "vllm":
                st["moe_in_vllm"] = (
                    args[0] if args else kwargs["hidden_states"]
                ).clone()

        def attn_hook(module, args, kwargs, out):
            if st.get("mode") == "vllm":
                st["attn_vllm"] = out.clone()
                return st["attn"].view_as(out).to(out.dtype)
            return None

        def topk_hook(layer):
            def hook(module, args, kwargs, out):
                rows = st["args"][0].shape[0] if "args" in st else 0
                buf = layer.attn.topk_indices_buffer[:rows]
                if st.get("mode") == "mono":
                    st["topk_mono"] = buf.clone()
                elif st.get("mode") == "vllm" and "topk_mono" in st:
                    buf.copy_(st["topk_mono"])

            return hook

        for i, layer in self._decoder_layers().items():
            layer.register_forward_pre_hook(pre_hook(i, layer), with_kwargs=True)
            layer.register_forward_hook(hook, with_kwargs=True)
            layer.attn.register_forward_hook(attn_hook, with_kwargs=True)
            layer.ffn.register_forward_pre_hook(ffn_hook, with_kwargs=True)
            if layer.attn.indexer is not None:
                layer.attn.indexer.indexer_op.register_forward_hook(
                    topk_hook(layer), with_kwargs=True
                )

    def save_capture(self, path: str) -> None:
        rank = self.rank  # type: ignore[attr-defined]
        torch.save((self._records, self._consts), f"{path}.rank{rank}")

    def mono_layers(self) -> dict:
        from vllm.models.deepseek_v41.amd import mono_decode

        def copies(m) -> bool:
            """Whether the layer holds the gfx942 copies of its weights, which
            the model makes when it loads them (``mono942``)."""
            ffn = m.ffn
            held = [
                ffn.experts.routed_experts,
                ffn.shared_experts.gate_up_proj,
                ffn.shared_experts.down_proj,
            ]
            if not m.mono.ffn_only:
                held += [getattr(m.attn, n) for n in mono_decode.ATTN_LINEARS_942]
            return all(hasattr(t, "mono942") for t in held)

        layers = {
            i: None
            if m.mono is None
            else [
                "ffn" if m.mono.ffn_only else "draft" if m.mono.draft else "whole",
                m.mono._weights is not None,
                copies(m),
            ]
            for i, m in self._decoder_layers().items()
        }
        epochs = [
            -1 if r is None else int(r.epoch[0])
            for r in (mono_decode._runner, mono_decode._draft_runner)
        ]
        return {
            "layers": layers,
            "epoch": epochs[0],
            "draft_epoch": epochs[1],
            "steps": dict(getattr(self, "_steps", {})),
        }


def _decode(
    vllm_runner,
    monkeypatch,
    tp: int,
    prompts,
    path: str,
    arch: str,
    spec: bool = False,
    graphs: bool = False,
    compare: bool = False,
):
    """Greedy decode of ``prompts`` on the mono path, with DSpark (``spec``),
    in FULL CUDA graphs of 8 rows (``graphs``) or eager: (each prompt's output,
    the workers' mono layers before and after the decode). With ``compare``,
    every decode-only step's comparisons with vLLM's path (``capture_steps``)
    are saved under ``path``."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    # CDNA4 has the MXFP4 indexer cache with 128-token blocks. gfx942 keeps
    # vLLM's FP8 indexer, which needs 64-token blocks.
    indexer: dict = dict(block_size=64)
    if arch == "cdna4":
        indexer = dict(
            attention_config={
                "indexer_kv_dtype": "mxfp4",
                "indexer_sparse_logits": True,
            },
            block_size=128,
        )
    steps: dict = dict(enforce_eager=True, max_num_seqs=len(prompts))
    if graphs:
        steps = dict(
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [8],
            },
            max_num_seqs=8,
        )
    with vllm_runner(
        MODEL,
        tensor_parallel_size=tp,
        load_format=LOAD_FORMAT,
        hf_overrides=_truncate,
        worker_extension_cls=f"{__name__}.MonoProbe",
        moe_backend="aiter",
        monokernel=True,
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        **indexer,
        max_model_len=4096,
        # All prompts prefill in one step, so that no decode token shares a
        # step with a prefill and every decode step has all the rows.
        max_num_batched_tokens=max(16384, sum(len(p) for p in prompts)),
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        speculative_config=DSPARK if spec else None,
        **steps,
    ) as runner:
        runner.llm.collective_rpc("capture_steps", args=(compare,))
        before = runner.llm.collective_rpc("mono_layers")
        params = SamplingParams(temperature=0.0, max_tokens=STEPS + 1, logprobs=5)
        outs = runner.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params
        )
        if compare:
            runner.llm.collective_rpc("save_capture", args=(path,))
        probe = runner.llm.collective_rpc("mono_layers")
    return [o.outputs[0] for o in outs], before, probe


def _prompts(rows: int) -> list[list[int]]:
    """Prompts past both ratios' 512 compressed entries, so top-k selects."""
    g = torch.Generator().manual_seed(0)
    lens = torch.randperm(1700, generator=g)[:rows].add(300).tolist()
    return [torch.randint(1000, 60000, (n,), generator=g).tolist() for n in lens]


def _skip_unless(arch: str, tp: int) -> None:
    if not arch:
        pytest.skip("the mono kernels target CDNA4 and gfx942")
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")


def _engaged(
    before: list, after: list, layers: dict, steps: int, replayed: bool = False
) -> None:
    """Every rank ran ``layers`` ({layer: kind}) on the mono path, with the
    weights taken off the layer and (gfx942) its copies made at load, at least
    ``steps`` steps. Eager, each runner's epoch moved one launch for every
    decode-only step of every layer, as the capture hooks counted them, so no
    step fell back to vLLM's path. ``replayed``: every decode-only step replayed
    one CUDA graph, which the hooks do not see: one launch a layer a step."""
    copies = _arch() == "gfx942"
    for b, a in zip(before, after):
        assert a["layers"] == {i: [kind, True, copies] for i, kind in layers.items()}, a
        for epoch, draft in (("epoch", False), ("draft_epoch", True)):
            group = [i for i in layers if (i in DRAFT) == draft]
            if not group:
                continue
            moved = a[epoch] - max(b[epoch], 0)
            if replayed:
                assert not a["steps"], a
                assert moved >= len(group) * steps, (b, a)
                assert moved % len(group) == 0, (b, a)
                continue
            assert min(a["steps"].get(i, 0) for i in group) >= steps, a
            assert moved == sum(a["steps"][i] for i in group), (b, a)


# ---- the parity budgets: each path's own rounding, on equal inputs
#
# E4M3 keeps 3 mantissa bits. A value rounded to it at an arbitrary point within
# its ulp (2^-3 of its binade's base, so <= 2^-3 |v|) moves by <= 2^-3 |v| /
# sqrt(12) rms: E4M3_RMS of it, whatever the block's power-of-two scale.
E4M3_RMS = 2.0**-3 / 12**0.5
# The attention output (the ranks' wo_b outputs summed), relative L2 a row. On
# CDNA4 both paths quantize the same tensors by the same rules. Their inputs
# agree up to f32 summation order as far as the attention core, whose p and
# partials each rounds to bf16 its own way (~2^-8 relative). Past the core, the
# inverse-RoPE MXFP8 and wo_a's MXFP8 turn that into one-step E4M3 flips. With
# the two paths' roundings independent (the most they can differ), each stage
# adds 2 E4M3_RMS^2 to the relative mean square, which random linear maps keep:
# 2 E4M3_RMS = 7.2%. A value d (relative) away from a rounding boundary flips
# with odds about d / ulp, so expected is about sqrt(0.07 d) a stage: 2^-8 ->
# 1.2% -> 2.9% (2.5-3.4% seen). gfx942, unvalidated on hardware: vLLM dequantizes
# the dense linears to bf16 at load and keeps bf16 activations, the kernels
# quantize four (the layer input, the q rank, the core's and wo_a's outputs):
# four one-sided E4M3_RMS^2 (the q rank's through logits of std ~1), the same.
ATTN_REL = 2 * E4M3_RMS
# The MoE on the same bf16 input, rms over a row, in units of 2^-7 rms of the
# row (a bf16 ulp at the row's rms or above). The quantized inputs and every f32
# sum agree up to summation order; the paths differ in summing a row's six
# routed down outputs and the shared expert's in bf16: the kernels in top-k
# order, AITER's atomics in any. 2 x 6 roundings of running sums, each uniform
# within half an ulp of a sum no larger than the row's (independent terms):
# sqrt(12 / 12) ulp, split over the ranks' partials, which add in f32 and round
# once more (1 / sqrt(6)): ~1.1 expected (0.6 seen), 2 allowed. The MXFP8 scale
# rule of aiter#6548 alone is 2.0-2.4.
MOE_ULPS = 2.0
# gfx942, unvalidated on hardware: vLLM's MoE keeps bf16 activations, the
# kernels quantize the input and the intermediate (MXFP8). The input's
# E4M3_RMS reaches gate and up alike; silu(g) u carries u's error and g's times
# g silu'(g) / silu(g) = 1 + g sigmoid(-g), whose rms weighted by silu(g)^2 (the
# relative L2 gain) is <= 1.21 for g ~ N(0, s^2) at any s, and pointwise <= 1.28
# for g > 0; then the intermediate's own: sqrt(1 + 1.28^2 + 1) E4M3_RMS = 6.9%.
MOE_REL_942 = (1 + 1.28**2 + 1) ** 0.5 * E4M3_RMS
U = 2.0**-24
# the seam norm's rstd, relative (test_dsv41_mono_numerics' KAPPA["rstd"])
RSTD = (33 / 2 + 2) * U


def _ulp(v: torch.Tensor) -> torch.Tensor:
    return R.ulp(v.abs().clamp(min=2.0**-126))


def _fn64(fn: torch.Tensor) -> torch.Tensor:
    """A seam's fn as the kernels read it: bf16 hi + lo."""
    hi = fn.to(torch.bfloat16).float()
    return hi.double() + (fn - hi).to(torch.bfloat16).double()


def _mixes(res, seam, eps):
    """(the f64 mixes of the residual rows, the first-order bound on a seam's
    computed ones about them, test_dsv41_mono_numerics' _mix_bound)."""
    from tests.models.deepseek_v41.amd_specialized_decode.test_dsv41_mono_numerics import (  # noqa: E501
        _mix_bound,
    )

    fn, scale, base = seam
    rms_eps, hc_eps, iters = eps
    x = res.flatten(1).to(torch.bfloat16)
    fn = _fn64(fn)
    hc = res.shape[1]
    want = R.mhc_mixes(x, fn, scale, base, hc, rms_eps, hc_eps, hc_eps, 2.0, iters)
    return want, _mix_bound(x, fn, scale, base, rms_eps, hc_eps, iters)


def _fold_err(mid, inputs):
    """How far the two paths' attention seam residuals ``mid`` may differ on
    equal inputs (x, residual, post, comb): each path's f32 sum of post x + sum
    comb r (5 rounded products, 4 adds) within 5u of the sum of its terms, so
    the two bf16 results an ulp and those 10u apart."""
    x, res, post, comb = inputs
    err = _ulp(mid)
    if res is not None and x is not None and x.dim() == 2 and res.dim() == 3:
        x, res, post, comb = (t.double() for t in inputs)
        terms = (post.reshape(*res.shape[:2], 1) * x[:, None]).abs()
        terms = terms + torch.einsum("mij,mid->mjd", comb.abs(), res.abs())
        err = err + 10 * U * terms
    return err


def _mixes_match(tag, got, got_v, res, res_v, seam, eps) -> None:
    """A seam's (post, comb, pre) of both paths, each from its own residual:
    within the f64 mixes' difference and each path's _mix_bound."""
    (want, b), (want_v, b_v) = (_mixes(t.double(), seam, eps) for t in (res, res_v))
    for i, name, k in ((0, "pre", 2), (1, "post", 0), (2, "comb", 1)):
        a, a_v = (t.double().view_as(want[i]) for t in (got[k], got_v[k]))
        allow = (want[i] - want_v[i]).abs() + b[i] + b_v[i]
        err = (a - a_v).abs()
        assert (err <= allow).all(), (tag, name, (err / allow).max().item())


def _seam(r: dict, c: dict) -> None:
    """The FFN seam of both paths: the residual and the MoE input each within
    their roundings (bf16, f32 sums) and the attention seam's mixes, one or two
    ulps, plus the difference of the attention outputs where vLLM's path took
    its own (``attn_vllm``); the layer's mixes as ``_mixes_match``."""
    f64 = lambda ts: [t.double() for t in ts]  # noqa: E731
    attn, mid, comb_a, pre_a, post_a = f64(
        (r["attn"][0], r["res_mid"], r["comb_a"], r["pre_a"], r["post_a"])
    )
    post_a = post_a.reshape(*mid.shape[:2], 1)
    d_attn = 0
    if r.get("attn_vllm") is not None:
        d_attn = post_a * (attn - r["attn_vllm"].double()).abs()[:, None]
    _, res, *mixes = f64(r["mono"])
    _, res_v, *mixes_v = f64(r["vllm"])
    _, (bpre, bpost, bcomb) = _mixes(mid, c["seams"]["attn"], c["eps"])
    # the FFN seam's fold as the attention seam's, on the residuals and the
    # attention seam's mixes, each path's within _mix_bound of the f64
    terms = (post_a * attn[:, None]).abs() + torch.einsum(
        "mij,mid->mjd", comb_a, mid.abs()
    )
    bound = (
        _ulp(torch.maximum(res.abs(), res_v.abs()))
        + d_attn
        + 10 * U * terms
        + torch.einsum("mij,mid->mjd", comb_a, _fold_err(mid, r["inputs"]))
        + 2 * bpost[..., None] * attn[:, None].abs()
        + 2 * torch.einsum("mij,mid->mjd", bcomb, mid.abs())
    )
    err = (res - res_v).abs()
    assert (err <= bound).all(), ("residual", (err / bound).max().item())
    # the MoE input: rmsnorm(sum pre_a r) w, the sum rounded to bf16, then the
    # normed rows
    lin = torch.einsum("mi,mid->md", pre_a, res)
    d_lin = (
        _ulp(lin)
        + torch.einsum("mi,mid->md", pre_a, err)
        + torch.einsum("mi,mid->md", 2 * bpre + 8 * U * pre_a, res.abs())
    )
    rstd = torch.rsqrt(lin.square().mean(-1, keepdim=True) + c["eps"][0])
    d_rstd = (lin.abs() * d_lin).sum(-1, keepdim=True) / lin.square().sum(
        -1, keepdim=True
    ) + 2 * RSTD
    w = c["ffn_norm"].double()
    moe_in, moe_in_v = r["moe_in"].double(), r["moe_in_vllm"].double()
    bound = (
        _ulp(torch.maximum(moe_in.abs(), moe_in_v.abs()))
        + w.abs() * rstd * d_lin
        + (w * lin * rstd).abs() * (d_rstd + 4 * U)
    )
    err = (moe_in - moe_in_v).abs()
    assert (err <= bound).all(), ("MoE input", (err / bound).max().item())
    _mixes_match("FFN seam", mixes, mixes_v, res, res_v, c["seams"]["ffn"], c["eps"])


def _route_ties(x: torch.Tensor, c: dict) -> torch.Tensor:
    """Rows whose routing two f32 router GEMVs of the bf16 rows x may pick
    differently. Each sums the same exact products (bf16 x bf16) in its own
    order, each add rounding to nearest: errors uniform within half an ulp,
    2^-24 |S_i| of the running sum, so sqrt(sum S_i^2 / 3) 2^-24 rms, taken at
    6 sigma (the sequential order's S_i). A score sqrt(softplus(l)) + b moves
    by that times its slope, plus 4u of its sqrt and add; a tie where the k-th
    and (k+1)-th scores are closer than twice their two errors."""
    x, w, k = x.double(), c["gate"].double(), c["topk"]
    logit = x @ w.T
    sp = torch.nn.functional.softplus(logit)
    score = sp.sqrt() + c["bias"].double()
    edge = score.sort(-1, descending=True).indices[:, k - 1 : k + 1]
    sums = (x[:, None] * w[edge]).cumsum(-1)
    dl = 6 * 2.0**-24 * (sums.square().sum(-1) / 3).sqrt()
    lg, s = logit.gather(1, edge), sp.gather(1, edge).sqrt()
    ds = torch.sigmoid(lg) / (2 * s.clamp(min=2.0**-126)) * dl
    ds = ds + 4 * U * score.gather(1, edge).abs()
    margin = score.gather(1, edge).diff(dim=1).abs()[:, 0]
    return margin < 2 * ds.sum(-1)


def _moe_parity(got, want, moe_in, c: dict, gfx942: bool = False) -> None:
    """The MoE output of both paths on the same input ``moe_in``, a row within
    MOE_ULPS (MOE_REL_942 on gfx942) unless its routing is a tie."""
    got, want = got.double(), want.double()
    if gfx942:
        err = (got - want).norm(dim=-1) / want.norm(dim=-1)
        bad = err > MOE_REL_942
    else:
        unit = 2.0**-7 * want.square().mean(-1).sqrt()
        err = (got - want).square().mean(-1).sqrt() / unit
        bad = err > MOE_ULPS
    ties = _route_ties(moe_in, c)
    assert not (bad & ~ties).any(), ("MoE", err.max().item())


def _parity(path: str, tp: int, layers) -> None:
    """Every compared step of every layer in ``layers``: the mono step against
    vLLM's path on the same inputs and caches, stage by stage (see
    ``capture_steps``), within the budgets above. A row whose routing is a tie
    within the router's rounding may take another expert."""
    gfx942 = _arch() == "gfx942"
    for rank in range(tp):
        records, consts = torch.load(f"{path}.rank{rank}")
        assert {r["layer"] for r in records} == set(layers), sorted(consts)
        for r in records:
            c, tag = consts[r["layer"]], (rank, r["layer"])
            got, want = (t.double() for t in r["attn"])
            err = (got - want).norm(dim=-1) / want.norm(dim=-1)
            assert (err <= ATTN_REL).all(), (tag, "attention", err.max().item())
            _seam(r, c)
            got, want = r["moe"]
            _moe_parity(got, want, r["moe_in"], c, gfx942)


@pytest.mark.parametrize(("tp", "rows"), [(2, 8), (4, 8), (4, 42)])
def test_mono_decode_matches_the_model(vllm_runner, monkeypatch, tmp_path, tp, rows):
    arch = _arch()
    _skip_unless(arch, tp)
    if (tp, rows) not in CASES[arch]:
        pytest.skip(f"not a case of {arch}")
    path = f"{tmp_path}/steps"
    _, before, after = _decode(
        vllm_runner, monkeypatch, tp, _prompts(rows), path, arch, compare=True
    )
    # both layer kinds took the mono path on every rank, at every decode step,
    # and on gfx942 from the weight copies made at load
    layers = {i: "whole" if i in WHOLE else "ffn" for i in WHOLE + FFN}
    _engaged(before, after, layers, STEPS)
    _parity(path, tp, layers)


@pytest.mark.parametrize("tp", [4])
def test_mono_dspark_matches_the_model(vllm_runner, monkeypatch, tmp_path, tp):
    """DSpark with 5 drafts on 8 requests: the target's verify steps of 48
    rows on the target's runner, the three draft layers whole on the draft's
    runner (its 128 / 3 routing and non-causal SWA rows)."""
    arch = _arch()
    _skip_unless(arch, tp)
    if arch != "cdna4":
        pytest.skip("the DSpark draft layers run whole on CDNA4")
    path = f"{tmp_path}/steps"
    _, before, after = _decode(
        vllm_runner, monkeypatch, tp, _prompts(8), path, arch, spec=True,
        compare=True,
    )  # fmt: skip
    layers = {i: "whole" for i in WHOLE} | {i: "draft" for i in DRAFT}
    _engaged(before, after, layers, 1)
    _parity(path, tp, layers)


@pytest.mark.parametrize("tp", [2])
def test_mono_decode_full_graphs_match_eager(vllm_runner, monkeypatch, tmp_path, tp):
    """FULL CUDA graphs of 8 rows replaying the mono launches for 5 requests,
    the 3 pad rows inert: the eager run's tokens. The graphs' warmup builds the
    runner and its kernels eagerly, before the capture, and every decode step
    then replays a graph (no layer forward runs in Python)."""
    arch = _arch()
    _skip_unless(arch, tp)
    if arch != "cdna4":
        pytest.skip("the 8-row graph pads only on CDNA4 (gfx942: 6-row passes)")
    prompts = _prompts(5)
    eager, before_e, after_e = _decode(
        vllm_runner, monkeypatch, tp, prompts, f"{tmp_path}/eager", arch
    )
    graphs, before_g, after_g = _decode(
        vllm_runner, monkeypatch, tp, prompts, f"{tmp_path}/graphs", arch,
        graphs=True,
    )  # fmt: skip
    layers = {i: "whole" for i in WHOLE}
    _engaged(before_e, after_e, layers, STEPS)
    _engaged(before_g, after_g, layers, STEPS, replayed=True)
    check_logprobs_close(
        outputs_0_lst=[(o.token_ids, o.text, o.logprobs) for o in eager],
        outputs_1_lst=[(o.token_ids, o.text, o.logprobs) for o in graphs],
        name_0="eager",
        name_1="graphs",
    )
