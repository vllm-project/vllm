# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode path through the real model: a five-layer
DeepSeek-V4.1-Flash at TP2 and TP4 on CDNA4, and at TP4 on gfx942, served with
`--monokernel` on and off on the same random weights: every decoder layer's
outputs at every decode step must match. Unlike
`test_dsv41_mono_numerics.py`, which checks the kernels against vLLM's ops
one stage at a time, this runs `DeepseekV4DecoderLayer.forward` as the engine
does -- the model's hooks, the per-step path choice, the weights taken off the
layer, the metadata and the indexer's top-k -- so a change to the decoder layer
that the kernels do not follow fails here.

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

At TP4 on gfx950, `--monokernel` runs decode steps of 17-256 rows on the asm
decode layers (`vllm/models/deepseek_v41/amd/asm`):
`test_asm_decode_matches_the_model` checks them the same way, on a six-layer
model, decode-only and mixed steps and CUDA graph replay.
"""

import os
import zlib
from typing import Any

import pytest
import torch

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
WHOLE, FFN = (0, 1, 2, 3, 4), ()
STEPS = 5  # decode steps after the prefill's token


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
CASES = {"cdna4": [(2, 8), (4, 8), (8, 8), (8, 28)], "gfx942": [(4, 8), (4, 42)]}


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
    """Worker extension (`worker_extension_cls`): it also brings this module,
    and so the loader, into the workers."""

    def _decoder_layers(self) -> dict:
        from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

        model = self.model_runner.model  # type: ignore[attr-defined]
        return {
            m.attn.layer_id: m
            for m in model.modules()
            if isinstance(m, DeepseekV4DecoderLayer)
        }

    def capture_steps(self, lens: list[int]) -> None:
        """Keep each decoder layer's outputs (x, residual) for every decode
        token taken in a decode-only step (the steps the mono path takes),
        keyed by (layer, prompt length, position): a request's step and row
        can differ between engines (requests reach the engine core while it
        runs), its prompt length and position cannot. `lens`: the prompt
        lengths, all distinct; a row's request is the longest prompt not past
        its position."""
        from vllm.forward_context import get_forward_context

        self._captured: dict = {}
        lens = sorted(lens)

        def hook(layer_id):
            def keep(module, args, kwargs, out):
                md = get_forward_context().attn_metadata
                if not isinstance(md, dict):
                    return
                swa = md.get(module.attn.swa_cache_layer.prefix)
                if swa is None or swa.num_prefills != 0:
                    return
                pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
                for row, p in enumerate(pos.tolist()):
                    n = max(n for n in lens if n <= p)
                    kept = [t[row].float().cpu() for t in out[:2]]
                    self._captured.setdefault((layer_id, n, p), kept)

            return keep

        def gap_hook(layer_id, layer):
            def keep(ffn, args, kwargs):
                # vLLM's path only (the mono kernels route inside): each row's
                # top-6 margin, sqrt(softplus(logits)) + bias as vLLM routes
                md = get_forward_context().attn_metadata
                if not isinstance(md, dict):
                    return
                swa = md.get(layer.attn.swa_cache_layer.prefix)
                if swa is None or swa.num_prefills != 0:
                    return
                h = args[0] if args else kwargs["hidden_states"]
                logits = h.float() @ ffn.gate.weight.float().T
                score = torch.nn.functional.softplus(logits).sqrt()
                score = score + ffn.gate.e_score_correction_bias.float()
                top = score.topk(7, dim=-1).values
                gap = (top[:, 5] - top[:, 6]).tolist()
                pos = self._positions
                for row, p in enumerate(pos.tolist()):
                    n = max(n for n in lens if n <= p)
                    self._gaps.setdefault((layer_id, n, p), gap[row])

            return keep

        def pos_hook(module, args, kwargs):
            self._positions = kwargs.get(
                "positions", args[1] if len(args) > 1 else None
            )

        self._gaps: dict = {}
        for i, layer in self._decoder_layers().items():
            layer.register_forward_hook(hook(i), with_kwargs=True)
            layer.register_forward_pre_hook(pos_hook, with_kwargs=True)
            gap = gap_hook(i, layer)
            layer.ffn.register_forward_pre_hook(gap, with_kwargs=True)

    def save_capture(self, path: str) -> None:
        rank = self.rank  # type: ignore[attr-defined]
        torch.save((self._captured, self._gaps), f"{path}.rank{rank}")

    def mono_layers(self) -> dict:
        from vllm.models.deepseek_v41.amd import mono_decode

        def copies(m) -> bool:
            """Whether the layer holds the gfx942 copies of its weights, which
            the model makes when it loads them (`mono942`)."""
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
                "ffn" if m.mono.ffn_only else "whole",
                m.mono._weights is not None,
                copies(m),
            ]
            for i, m in self._decoder_layers().items()
        }
        runner = mono_decode._runner
        return {
            "layers": layers,
            "epoch": -1 if runner is None else int(runner.epoch[0]),
        }


def _decode(
    vllm_runner, monkeypatch, tp: int, mono: bool, prompts, path: str, arch: str
):
    """Greedy decode of `prompts` with the mono path on or off: (each
    prompt's output, the workers' mono layers); each decoder layer's outputs
    for the decode tokens of decode-only steps, and vLLM's routing margins,
    saved under `path`."""
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
    with vllm_runner(
        MODEL,
        tensor_parallel_size=tp,
        load_format=LOAD_FORMAT,
        hf_overrides=TRUNCATED,
        worker_extension_cls=f"{__name__}.MonoProbe",
        moe_backend="aiter",
        monokernel=mono,
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        **indexer,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=len(prompts),
        # All prompts prefill in one step, so that no decode token shares a
        # step with a prefill and every decode step has all the rows.
        max_num_batched_tokens=max(16384, sum(len(p) for p in prompts)),
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
    ) as runner:
        runner.llm.collective_rpc("capture_steps", args=([len(p) for p in prompts],))
        params = SamplingParams(temperature=0.0, max_tokens=STEPS + 1)
        outs = runner.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params
        )
        runner.llm.collective_rpc("save_capture", args=(path,))
        probe = runner.llm.collective_rpc("mono_layers")
    return [o.outputs[0] for o in outs], probe


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize(("tp", "rows"), [(2, 8), (4, 8), (4, 42), (8, 8), (8, 28)])
def test_mono_decode_matches_the_model(vllm_runner, monkeypatch, tmp_path, tp, rows):
    arch = _arch()
    if not arch:
        pytest.skip("the mono kernels target CDNA4 and gfx942")
    if (tp, rows) not in CASES[arch]:
        pytest.skip(f"not a case of {arch}")
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")
    g = torch.Generator().manual_seed(0)
    # Contexts past both ratios' 512 compressed entries, so top-k selects.
    # The captures key a decode token by (prompt length, position), so the
    # lengths are more than STEPS apart: a request whose positions reached a
    # longer prompt's length would share that prompt's keys. Skipping the
    # lengths that are too close keeps the 8-row cases' lengths as they were.
    lens: list[int] = []
    for n in torch.randperm(1700, generator=g).add(300).tolist():
        if all(abs(n - m) > STEPS for m in lens):
            lens.append(n)
        if len(lens) == rows:
            break
    prompts = [torch.randint(1000, 60000, (n,), generator=g).tolist() for n in lens]

    on, probe_on = _decode(
        vllm_runner, monkeypatch, tp, True, prompts, f"{tmp_path}/on", arch
    )
    off, probe_off = _decode(
        vllm_runner, monkeypatch, tp, False, prompts, f"{tmp_path}/off", arch
    )

    # both layer kinds took the mono path on every rank, at every decode step,
    # and on gfx942 from the weight copies made at load
    launches = len(WHOLE + FFN)  # one launch (pair) a layer a step
    copies = arch == "gfx942"
    for rank in probe_on:
        assert rank["layers"] == {
            i: ["whole" if i in WHOLE else "ffn", True, copies] for i in WHOLE + FFN
        }, rank
        assert rank["epoch"] >= launches * STEPS, rank
        assert rank["epoch"] % launches == 0, rank
    assert all(set(r["layers"].values()) == {None} for r in probe_off)
    # each decode token's input matches between the runs while the greedy
    # tokens before it do: (prompt length, position) pairs to compare
    same: set[tuple[int, int]] = set()
    for prompt, a, b in zip(prompts, on, off):
        for step in range(STEPS):
            if a.token_ids[: step + 1] != b.token_ids[: step + 1]:
                break
            same.add((len(prompt), len(prompt) + step))

    # each layer's outputs for those tokens, mono against vLLM's path, token by
    # token. Layer 0 sees the same inputs on both paths, but its FFN input
    # differs by up to test_dsv41_mono_numerics' 1.2%, which random weights
    # amplify through the MoE: 3%. Each later layer takes the previous one's
    # slightly different output, which random weights amplify (2-4% by layer
    # 4): 8%. A token whose top-6 routing is a near-tie on vLLM's path may pick
    # another expert: a 0.4% input difference moves scores near 0.8 by about
    # 3e-3, so 6th / 7th scores within 5e-3 at layer 0 and, past it, where the
    # input carries the earlier layers' differences (0.4-0.7%), within 1e-2.
    # Once it has, its residual stream differs and the later layers are exempt
    # for it. The residual stream: 3% (0.5% seen). A layer the kernels no
    # longer follow is off by far more, on every token.
    # On gfx942 vLLM's path runs the MoE with bf16 activations, while the
    # kernels quantize them to MXFP8 before both expert GEMMs, as on CDNA4.
    # With these random weights that alone moves the FFN output by 4.65%
    # (5.0% at most over 256 tokens), so layer 0 gets 6% there (5.3% seen).
    # The later layers' 8% holds on gfx942 too (5.6% seen). The residual
    # stream then differs by 1.0% after layer 1 and by up to 2.5% after layer
    # 4, still under its 3%. That moves scores near 0.8 by up to about 2e-2,
    # so past layer 0 a near-tie is a gap within 2e-2 there (1.1e-2 seen).
    # And 22% of the tokens flip an expert at some layer, at 8 and at 42 rows
    # alike, so up to a third may flip there instead of a quarter.
    first_bound = 0.06 if arch == "gfx942" else 0.03
    later_tie = 2e-2 if arch == "gfx942" else 1e-2
    flip_divisor = 3 if arch == "gfx942" else 4
    for rank in range(tp):
        got, _ = torch.load(f"{tmp_path}/on.rank{rank}")
        want, gaps = torch.load(f"{tmp_path}/off.rank{rank}")
        keys: list[tuple[int, int]] = sorted(
            same & {k[1:] for k in got} & {k[1:] for k in want}
        )
        assert len(keys) >= len(prompts), keys
        flipped: set = set()
        for i in sorted(WHOLE + FFN):
            for k in keys:
                if k in flipped:
                    continue
                (x, res), (x_ref, res_ref) = got[(i, *k)], want[(i, *k)]
                err = _rel(res, res_ref)
                assert err < 0.03, (rank, i, k, "residual", err)
                bound = first_bound if i == 0 else 0.08
                err = _rel(x, x_ref)
                if err >= bound:
                    tie = 5e-3 if i == 0 else later_tie
                    assert gaps[(i, *k)] < tie, (rank, i, k, "x", err)
                    flipped.add(k)
        assert len(flipped) <= len(keys) // flip_divisor, (rank, sorted(flipped))


# ---- The asm decode layers (`vllm/models/deepseek_v41/amd/asm`): steps of
# 17-256 rows at TP4 on gfx950, which `--monokernel` runs on them (the mono
# decode layer above keeps up to 16 rows there). A six-layer model whose layers
# take every path the real model's do: a dense layer (0), an Engram on a
# ratio-0 layer (1: our seam, vLLM's attention), an Engram on an index source
# (2), a consumer (3), the candidate source (4) and a layer indexing over its
# candidates (5). Every decoder layer's outputs must match vLLM's path:
# - on decode-only steps of every row count where the kernels change mode
#   (aimed at 256, 129, 128, 65, 64 and 17 rows), at a long --max-model-len (the
#   indexer's wide, candidate-source and gather kernels);
# - on mixed steps (a prompt's prefill next to 40 decode rows: the decode rows);
# - and the decode-only steps replayed from CUDA graphs (the indexer's narrow
#   kernels) give the eager run's tokens.
# Three engines (on, off, graph replay), so the test stays short.

ASM_TP = 4
ASM_MODEL = {
    **TRUNCATED,
    "num_hidden_layers": 6,
    # one entry a layer, the three MTP layers' included
    "compress_ratios": [0, 0, 2, 2, 1, 1, 0, 0, 0],
    "kv_source_layer_ids": [2, 4],
    # 5: an indexer of its own over 4's candidates
    "index_source_layer_ids": [2, 4, 5],
    "candidate_source_layer_id": 4,
    "engram_layer_ids": [1, 2],
    "engram_num_embeddings": [32768, 32768],
    "dspark_target_layer_ids": [3, 4, 5],
}
K1_LAYERS, K2_LAYERS = (2, 3, 4, 5), (0, 1, 2, 3, 4, 5)
# decode-only steps of these row counts, one after another: request i stops
# after its last step (requests that reach the engine core after its first
# step shift them a little)
STEP_ROWS = (256, 129, 128, 65, 64, 17)
# the kernels' modes change at 64 / 65 and 128 / 129 rows
MODE_ROWS = ((17, 64), (65, 128), (129, 255), (256, 256))
# the mixed phase: 40 requests decode, a 100-token prompt joins (140 rows)
MIXED_DECODERS, MIXED_TOKENS, LATE_LEN = 40, 16, 100
LONG_LEN, SHORT_LEN = 262144, 4096
# the indexer top-k's kernels at LONG_LEN (decode.py _idx_topk_install)
LONG_KINDS = {"candidates none", "candidates source", "candidates consumer"}
# vLLM's top-k on these steps is expected, any other fallback is not
EXPECTED_IDX_FALLBACKS = {"not a decode step", "no metadata"}


class AsmProbe(MonoProbe):
    _positions: torch.Tensor  # the layer call's (MonoProbe.capture_steps)

    def capture_decode_rows(self, lens: list[int]) -> None:
        """capture_steps, plus the decode rows of mixed steps (the batch puts
        them first), with their routing gaps."""
        from vllm.forward_context import get_forward_context

        self.capture_steps(lens)
        lens = sorted(lens)

        def mixed_rows(layer) -> int:
            md = get_forward_context().attn_metadata
            swa: Any = (
                md.get(layer.attn.swa_cache_layer.prefix)
                if isinstance(md, dict)
                else None
            )
            return swa.num_decode_tokens if swa is not None and swa.num_prefills else 0

        def keys(n: int):
            for row, p in enumerate(self._positions[:n].tolist()):
                yield row, (max(k for k in lens if k <= p), p)

        def hook(layer_id):
            def keep(layer, args, kwargs, out):
                for row, k in keys(mixed_rows(layer)):
                    kept = [t[row].float().cpu() for t in out[:2]]
                    self._captured.setdefault((layer_id, *k), kept)

            return keep

        def gap_hook(layer_id, layer):
            def keep(ffn, args, kwargs):
                n = mixed_rows(layer)
                if not n:
                    return
                h = (args[0] if args else kwargs["hidden_states"])[:n]
                logits = h.float() @ ffn.gate.weight.float().T
                score = torch.nn.functional.softplus(logits).sqrt()
                score = score + ffn.gate.e_score_correction_bias.float()
                top = score.topk(7, dim=-1).values
                gap = (top[:, 5] - top[:, 6]).tolist()
                for row, k in keys(n):
                    self._gaps.setdefault((layer_id, *k), gap[row])

            return keep

        for i, layer in self._decoder_layers().items():
            layer.register_forward_hook(hook(i), with_kwargs=True)
            layer.ffn.register_forward_pre_hook(gap_hook(i, layer), with_kwargs=True)

    def asm_state(self) -> dict:
        """What the asm layers took, on this rank (syncs)."""
        from vllm.models.deepseek_v41.amd.asm import decode

        layers = {
            i: None
            if m.mono is None or m.mono.asm is None
            else [m.mono.asm.attn_fn is not None, m.mono.asm.ffn_fn is not None]
            for i, m in self._decoder_layers().items()
        }
        logged = decode._k1["logged"]
        runners = {"wob_ar": decode._wob["wob"], "comb_ar": decode._comb["car"]}
        for j, side in enumerate(decode._eg["sides"]):
            runners |= {
                f"engram{j} rows": side.ag_rows,
                f"engram{j} gate": side.ag_gate,
            }
        # per runner with a nonzero err word: (waves, the err words' values: the
        # seq of a timed-out launch, which a peer only causes by staying away
        # for ~135 s)
        timeouts = {}
        for name, r in runners.items():
            err = r.state()[1] if r is not None else []
            if any(err):
                timeouts[name] = (
                    sum(e != 0 for e in err),
                    sorted({e for e in err if e})[:8],
                )
        return {
            "layers": layers,
            "k1": sorted(decode._k1["seen"]),
            "k2": sorted(decode._k2["seen"]),
            "engram": sorted(decode._eg["seen"]),
            "mixed": "mixed" in logged,
            "idx": sorted(decode._summary["indexer top-k"]),
            "fallbacks": sorted(
                str(k)
                for k in logged
                if isinstance(k, tuple)
                and (
                    k[0] == "tk" or k[0] == "idx" and k[2] not in EXPECTED_IDX_FALLBACKS
                )
            ),
            # flag-poll timeouts of the folded all-reduces / Engram all-gathers
            "timeouts": timeouts,
        }


def _skip_unless_gfx950_tp4() -> None:
    """Skips where the asm kernels can't run; on CI's gfx950 jobs a missing
    dependency fails instead (no green job without the asm path)."""
    if not current_platform.is_rocm():
        pytest.skip("the asm decode kernels target gfx950")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the asm decode kernels target gfx950")
    if torch.accelerator.device_count() < ASM_TP:
        pytest.skip(f"needs {ASM_TP} GPUs")
    from vllm.models.deepseek_v41.amd.asm.kernels.build import toolchain_missing

    missing = toolchain_missing()
    try:
        import flydsl  # noqa: F401
    except ImportError:
        missing = "flydsl"
    if missing:
        (pytest.fail if os.environ.get("BUILDKITE") else pytest.skip)(
            f"needs {missing} to run the asm decode layers"
        )


def _prompts(n: int, lo: int, gap: int, copies: int = 1) -> list[list[int]]:
    """N random prompts of distinct lengths from lo on, more than `gap` apart
    (the captures key a token by prompt length and position), each `copies`
    times (a copy's rows are the same, so they share the keys)."""
    g = torch.Generator().manual_seed(0)
    lens = (torch.randperm(n, generator=g) * (gap + 1) + lo).tolist()
    prompts = [torch.randint(1000, 60000, (k,), generator=g).tolist() for k in lens]
    return [p for p in prompts for _ in range(copies)]


def _step_rows_max_tokens() -> list[int]:
    """Per request: 1 + the decode steps it takes, so that decode step s has
    STEP_ROWS[s - 1] rows."""
    out: list[int] = []
    for s, rows in reversed(list(enumerate(STEP_ROWS, 1))):
        out += [s + 1] * (rows - len(out))
    return out


def _run(vllm_runner, monkeypatch, mono: bool, graph: bool, phases: dict, path=None):
    """One engine: the decode-only phase (`generate`), then (eager) the mixed
    phase: MIXED_DECODERS requests decode, and once each has its first decode
    token a LATE_LEN-token prompt joins, so a step has its prefill next to their
    decode rows. path: capture every decode row's layer outputs there (eager).
    → (outputs per phase, the probe's state per rank)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    steps, mixed, late = phases["steps"], phases["mixed"], phases["late"]
    every = [*steps[0], *mixed, late]
    engine = (
        dict(
            enforce_eager=False,
            max_model_len=SHORT_LEN,
            compilation_config={
                "decode_cudagraph_capture_request_sizes": sorted(STEP_ROWS),
                "prefill_cudagraph_capture_token_sizes": [],
            },
        )
        if graph
        else dict(enforce_eager=True, max_model_len=LONG_LEN)
    )
    with vllm_runner(
        MODEL,
        tensor_parallel_size=ASM_TP,
        load_format=LOAD_FORMAT,
        hf_overrides=ASM_MODEL,
        worker_extension_cls=f"{__name__}.AsmProbe",
        moe_backend="aiter",
        monokernel=mono,
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
        engram_config={"cpu_offload": True},
        block_size=128,
        max_num_seqs=len(steps[0]),
        # every prompt of a phase prefills in one step
        max_num_batched_tokens=max(sum(map(len, steps[0])), sum(map(len, mixed))),
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        **engine,
    ) as runner:
        llm = runner.llm
        if path is not None:
            llm.collective_rpc("capture_decode_rows", args=([len(p) for p in every],))
        params = lambda t: SamplingParams(  # noqa: E731
            temperature=0.0, max_tokens=t, ignore_eos=True, logprobs=5
        )
        outs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in steps[0]],
            [params(t) for t in steps[1]],
        )
        out = {"steps": [o.outputs[0] for o in outs]}
        if not graph:
            eng, got = llm.llm_engine, {}
            for i, p in enumerate(mixed):
                eng.add_request(
                    f"m{i}", TokensPrompt(prompt_token_ids=p), params(MIXED_TOKENS)
                )
            while eng.has_unfinished_requests():
                for o in eng.step():
                    got[o.request_id] = o
                if (
                    "late" not in got
                    and len(got) == len(mixed)
                    and all(len(o.outputs[0].token_ids) >= 2 for o in got.values())
                ):
                    eng.add_request(
                        "late", TokensPrompt(prompt_token_ids=late), params(8)
                    )
                    got["late"] = None
            out["mixed"] = [got[f"m{i}"].outputs[0] for i in range(len(mixed))]
            out["mixed"].append(got["late"].outputs[0])
        if path is not None:
            llm.collective_rpc("save_capture", args=(path,))
        state = llm.collective_rpc("asm_state") if mono else None
    return out, state


# The bounds, the asm all-reduces summing in bf16 as vLLM's do: every layer's
# output is 6.6% off vLLM's in the median and 8.2% at most (K2's rounding path,
# as close to exact as vLLM's), the residual stream 0.09% at layer 0, 0.86% at
# layer 5; 20-25% of the tokens flip (ten runs).
X_TOL, RES_TOL = 0.10, 0.012


def _check_layers(prompts, on, off, tmp_path) -> None:
    """Each layer's outputs for the decode tokens both engines produced alike,
    asm against vLLM's path, token by token: x within X_TOL, the residual stream
    within RES_TOL. A token whose top-6 routing is a near
    tie on vLLM's path (6th / 7th scores within 5e-3 at layer 0, 2e-2 past it:
    the residual stream drifts there, and a flip at 0.0106 was seen, none past
    0.015 in 662 tokens) may pick another expert and land far off (48-77%
    seen); the later layers are exempt for it, and at most a third of the
    tokens may flip (20-25% seen over ten runs, as gfx942's mono layers
    above; a kernel that no longer follows the layer moves every token)."""
    errs: list[float] = []
    # A key (prompt length, position) counts where every request with that
    # prompt (the copies) has one token history in both engines: vLLM's path is
    # not batch invariant, so on random weights' near ties copies of a prompt
    # can part even within one engine, and the key's captured row is then any
    # of theirs.
    histories: dict[int, list] = {}
    for prompt, a, b in zip(prompts, on, off):
        histories.setdefault(len(prompt), []).extend([a.token_ids, b.token_ids])
    same: set[tuple[int, int]] = set()
    for n, seqs in histories.items():
        for step in range(min(map(len, seqs))):
            if any(t[: step + 1] != seqs[0][: step + 1] for t in seqs):
                break
            same.add((n, n + step))
    for rank in range(ASM_TP):
        got, _ = torch.load(f"{tmp_path}/on.rank{rank}")
        want, gaps = torch.load(f"{tmp_path}/off.rank{rank}")
        keys = sorted(same & {k[1:] for k in got} & {k[1:] for k in want})
        assert len(keys) >= len({len(p) for p in prompts}), keys
        flipped: set = set()
        for i in K2_LAYERS:
            for k in keys:
                if k in flipped:
                    continue
                (x, res), (x_ref, res_ref) = got[(i, *k)], want[(i, *k)]
                err = _rel(res, res_ref)
                assert err < RES_TOL, (rank, i, k, "residual", err)
                err = _rel(x, x_ref)
                errs.append(err)
                if err >= X_TOL:
                    tie = 5e-3 if i == 0 else 2e-2
                    assert gaps[(i, *k)] < tie, (rank, i, k, "x", err)
                    flipped.add(k)
        assert len(flipped) <= len(keys) // 3, (rank, sorted(flipped))
    errs.sort()
    med, top = errs[len(errs) // 2], errs[-1]
    print(
        f"layer outputs vs vLLM's: median {med:.4f}, max {top:.4f}; "
        f"{len(flipped)} of {len(keys)} tokens flip"
    )


def _check_replay(eager, graph) -> None:
    """Greedy sequences from graph replay against the eager run. The runs batch
    alike up to when requests reach the engine core, so a step's logprobs may
    differ a little (0.032 at most seen) and flip a near tie. Random weights give
    exact ties (several tokens at one logprob, broken either way), so where the
    tokens part the two picks need only be equally likely; check_logprobs_close's
    top-k membership would fail on a tie wider than k."""
    for i, (a, b) in enumerate(zip(eager, graph)):
        for j, (ta, tb) in enumerate(zip(a.token_ids, b.token_ids)):
            la, lb = a.logprobs[j][ta].logprob, b.logprobs[j][tb].logprob
            assert abs(la - lb) < 0.1, (i, j, ta, tb, la, lb)
            if ta != tb:
                break


def _check_state(state, ranges, kinds) -> None:
    """The asm kernels took every layer they cover and steps in each row range on
    every rank, the indexer top-k on our kernels (kinds) without unexpected
    fallbacks, and no all-reduce timed out."""
    for rank in state:
        assert rank["layers"] == {i: [i in K1_LAYERS, True] for i in K2_LAYERS}, rank
        for part in ("k1", "k2", "engram"):
            for lo, hi in ranges:
                assert any(lo <= m <= hi for m in rank[part]), (part, lo, hi, rank)
        assert all(s.startswith("asm") for s in rank["idx"]), rank["idx"]
        assert kinds <= {s.split(", ")[-1].rstrip(")") for s in rank["idx"]}, rank
        assert not rank["fallbacks"], rank["fallbacks"]
        assert not rank["timeouts"], rank["timeouts"]


def test_asm_decode_matches_the_model(vllm_runner, monkeypatch, tmp_path):
    _skip_unless_gfx950_tp4()
    max_tokens = _step_rows_max_tokens()
    # 32 prompts × 8 copies: short, so they all prefill in one step
    steps = _prompts(len(max_tokens) // 8, 128, len(STEP_ROWS) + 1, copies=8)
    # contexts past both ratios' 512 compressed entries; the late prompt's
    # positions stay below the others'
    mixed = _prompts(MIXED_DECODERS, 1100, MIXED_TOKENS + 1)
    late = _prompts(1, LATE_LEN, 0)[0]
    phases = {"steps": (steps, max_tokens), "mixed": mixed, "late": late}

    on, state = _run(vllm_runner, monkeypatch, True, False, phases, f"{tmp_path}/on")
    off, _ = _run(vllm_runner, monkeypatch, False, False, phases, f"{tmp_path}/off")
    _check_state(state, MODE_ROWS, LONG_KINDS)
    assert all(r["mixed"] for r in state), state
    _check_layers(
        [*steps, *mixed, late],
        [*on["steps"], *on["mixed"]],
        [*off["steps"], *off["mixed"]],
        tmp_path,
    )

    # graph replay of the decode-only phase
    replay, state = _run(vllm_runner, monkeypatch, True, True, phases)
    _check_state(state, MODE_ROWS, set())
    _check_replay(on["steps"], replay["steps"])
