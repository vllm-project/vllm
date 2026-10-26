# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode path through the real model: a five-layer
DeepSeek-V4.1-Flash at TP2 and TP4 on CDNA4, and at TP4 on gfx942, served with
``--monokernel`` on and off on the same random weights: every decoder layer's
outputs at every decode step must match. Unlike
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

    def capture_steps(self, lens: list[int]) -> None:
        """Keep each decoder layer's outputs (x, residual) for every token taken
        in a decode-only step (the steps the mono path takes), keyed by (layer,
        prompt length, position, its token; for a draft layer, its draft block's
        first position): a request's step and row can differ between engines
        (requests reach the engine core while it runs), its prompt length and
        position cannot. ``lens``: the prompt lengths, all distinct; a row's
        request is the longest prompt not past its position. With them vLLM's
        routing margin of each row: its last picked expert's score over the
        next one's."""
        from vllm.forward_context import get_forward_context

        self._captured: dict = {}
        self._gaps: dict = {}
        self._steps: dict[int, int] = {}
        self._keys: list | None = None
        lens = sorted(lens)

        def decode_only(layer) -> bool:
            md = get_forward_context().attn_metadata
            if not isinstance(md, dict):
                return False
            swa = md.get(layer.attn.swa_cache_layer.prefix)
            return swa is not None and swa.num_prefills == 0

        def row_keys(layer_id, args, kwargs):
            pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
            pos = pos.tolist()
            if layer_id in DRAFT:
                tag = list(pos)
                for row in range(1, len(pos)):
                    if pos[row] == pos[row - 1] + 1:
                        tag[row] = tag[row - 1]
            else:
                ids = kwargs.get("input_ids", args[2] if len(args) > 2 else None)
                tag = ids.tolist()
            return [
                (layer_id, max(n for n in lens if n <= p), p, a)
                for p, a in zip(pos, tag)
            ]

        def pre_hook(layer_id, layer):
            def keys(module, args, kwargs):
                self._keys = None
                if decode_only(layer):
                    self._keys = row_keys(layer_id, args, kwargs)
                    self._steps[layer_id] = self._steps.get(layer_id, 0) + 1

            return keys

        def hook(module, args, kwargs, out):
            for row, key in enumerate(self._keys or ()):
                kept = [t[row].float().cpu() for t in out[:2]]
                self._captured.setdefault(key, kept)

        def gap(ffn, args, kwargs):
            # vLLM's path only (the mono kernels route inside):
            # sqrt(softplus(logits)) + bias as vLLM routes
            if self._keys is not None:
                h = args[0] if args else kwargs["hidden_states"]
                logits = h.float() @ ffn.gate.weight.float().T
                score = torch.nn.functional.softplus(logits).sqrt()
                score = score + ffn.gate.e_score_correction_bias.float()
                k = ffn.n_activated_experts
                top = score.topk(k + 1, dim=-1).values
                margin = (top[:, k - 1] - top[:, k]).tolist()
                for row, key in enumerate(self._keys):
                    self._gaps.setdefault(key, margin[row])

        for i, layer in self._decoder_layers().items():
            layer.register_forward_pre_hook(pre_hook(i, layer), with_kwargs=True)
            layer.register_forward_hook(hook, with_kwargs=True)
            layer.ffn.register_forward_pre_hook(gap, with_kwargs=True)

    def save_capture(self, path: str) -> None:
        rank = self.rank  # type: ignore[attr-defined]
        torch.save((self._captured, self._gaps), f"{path}.rank{rank}")

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
    mono: bool,
    prompts,
    path: str,
    arch: str,
    spec: bool = False,
    graphs: bool = False,
):
    """Greedy decode of ``prompts`` with the mono path on or off, with DSpark
    (``spec``), in FULL CUDA graphs of 8 rows (``graphs``) or eager: (each
    prompt's output, the workers' mono layers before and after the decode);
    each decoder layer's outputs for the tokens of decode-only steps, and
    vLLM's routing margins, saved under ``path``."""
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
        monokernel=mono,
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
        runner.llm.collective_rpc("capture_steps", args=([len(p) for p in prompts],))
        before = runner.llm.collective_rpc("mono_layers")
        params = SamplingParams(temperature=0.0, max_tokens=STEPS + 1, logprobs=5)
        outs = runner.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params
        )
        runner.llm.collective_rpc("save_capture", args=(path,))
        probe = runner.llm.collective_rpc("mono_layers")
    return [o.outputs[0] for o in outs], before, probe


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _prompts(rows: int, gap: int = STEPS) -> list[list[int]]:
    """Prompts past both ratios' 512 compressed entries, so top-k selects.
    The captures key a row by its prompt length and position, so the lengths
    are more than ``gap`` apart: a request whose positions reached a longer
    prompt's length would share that prompt's keys. Skipping the lengths that
    are too close keeps the 8-row cases' lengths as they were."""
    g = torch.Generator().manual_seed(0)
    lens: list[int] = []
    for n in torch.randperm(1700, generator=g).add(300).tolist():
        if all(abs(n - m) > gap for m in lens):
            lens.append(n)
        if len(lens) == rows:
            break
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


def _matches(prompts, on, off, path: str, tp: int, layers) -> None:
    """Each layer's outputs for the rows whose inputs match between the runs,
    mono against vLLM's path, row by row: a target layer's rows whose greedy
    tokens up to their own match, a draft layer's rows whose block follows
    matching tokens."""
    arch = _arch()
    same: set[tuple[int, int]] = set()
    token: dict[tuple[int, int], int] = {}
    for prompt, a, b in zip(prompts, on, off):
        n = len(prompt)
        for step in range(STEPS):
            if a.token_ids[: step + 1] != b.token_ids[: step + 1]:
                break
            same.add((n, n + step))
            token[(n, n + step)] = a.token_ids[step]

    def comparable(key) -> bool:
        i, n, p, a = key
        if i in DRAFT:
            return (n, a) in same
        return (n, p) in same and token[(n, p)] == a

    # Layer 0 sees the same inputs on both paths, but its FFN input differs by
    # up to test_dsv41_mono_numerics' 1.2%, which random weights amplify
    # through the MoE: 3%. Each later layer takes the previous one's slightly
    # different output, which random weights amplify (2-4% by layer 4): 8%. A
    # token whose top-6 routing is a near-tie on vLLM's path may pick another
    # expert: a 0.4% input difference moves scores near 0.8 by about 3e-3, so
    # 6th / 7th scores within 5e-3 at layer 0 and, past it, where the input
    # carries the earlier layers' differences (0.4-0.7%), within 1e-2. Once it
    # has, its residual stream differs and the later layers are exempt for it.
    # The residual stream: 3% (0.5% seen). A layer the kernels no longer
    # follow is off by far more, on every token.
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
        got, _ = torch.load(f"{path}/on.rank{rank}")
        want, gaps = torch.load(f"{path}/off.rank{rank}")
        keys = sorted(k for k in got.keys() & want.keys() if comparable(k))
        rows = {(k[0] in DRAFT, *k[1:]) for k in keys}
        for group in {i in DRAFT for i in layers}:
            assert sum(r[0] == group for r in rows) >= len(prompts), keys
        flipped: set = set()
        for k in keys:
            i, row = k[0], (k[0] in DRAFT, *k[1:])
            if i not in layers or row in flipped:
                continue
            (x, res), (x_ref, res_ref) = got[k], want[k]
            err = _rel(res, res_ref)
            assert err < 0.03, (rank, k, "residual", err)
            bound = first_bound if i == 0 else 0.08
            err = _rel(x, x_ref)
            if err >= bound:
                tie = 5e-3 if i == 0 else later_tie
                assert gaps[k] < tie, (rank, k, "x", err)
                flipped.add(row)
        assert len(flipped) <= len(rows) // flip_divisor, (rank, sorted(flipped))


@pytest.mark.parametrize(("tp", "rows"), [(2, 8), (4, 8), (4, 42)])
def test_mono_decode_matches_the_model(vllm_runner, monkeypatch, tmp_path, tp, rows):
    arch = _arch()
    _skip_unless(arch, tp)
    if (tp, rows) not in CASES[arch]:
        pytest.skip(f"not a case of {arch}")
    prompts = _prompts(rows)
    on, before, probe_on = _decode(
        vllm_runner, monkeypatch, tp, True, prompts, f"{tmp_path}/on", arch
    )
    off, _, probe_off = _decode(
        vllm_runner, monkeypatch, tp, False, prompts, f"{tmp_path}/off", arch
    )
    # both layer kinds took the mono path on every rank, at every decode step,
    # and on gfx942 from the weight copies made at load
    layers = {i: "whole" if i in WHOLE else "ffn" for i in WHOLE + FFN}
    _engaged(before, probe_on, layers, STEPS)
    assert all(set(r["layers"].values()) == {None} for r in probe_off)
    _matches(prompts, on, off, str(tmp_path), tp, layers)


@pytest.mark.parametrize("tp", [4])
def test_mono_dspark_matches_the_model(vllm_runner, monkeypatch, tmp_path, tp):
    """DSpark with 5 drafts on 8 requests: the target's verify steps of 48
    rows on the target's runner, the three draft layers whole on the draft's
    runner (its 128 / 3 routing and non-causal SWA rows)."""
    arch = _arch()
    _skip_unless(arch, tp)
    if arch != "cdna4":
        pytest.skip("the DSpark draft layers run whole on CDNA4")
    # a draft block's rows reach 5 past a request's last position
    prompts = _prompts(8, gap=2 * STEPS + DSPARK_TOKENS)
    run = dict(arch=arch, spec=True)
    on, before, probe_on = _decode(
        vllm_runner, monkeypatch, tp, True, prompts, f"{tmp_path}/on", **run
    )
    off, _, probe_off = _decode(
        vllm_runner, monkeypatch, tp, False, prompts, f"{tmp_path}/off", **run
    )
    layers = {i: "whole" for i in WHOLE} | {i: "draft" for i in DRAFT}
    _engaged(before, probe_on, layers, 1)
    assert all(set(r["layers"].values()) == {None} for r in probe_off)
    _matches(prompts, on, off, str(tmp_path), tp, layers)


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
        vllm_runner, monkeypatch, tp, True, prompts, f"{tmp_path}/eager", arch
    )
    graphs, before_g, after_g = _decode(
        vllm_runner, monkeypatch, tp, True, prompts, f"{tmp_path}/graphs", arch,
        graphs=True,
    )  # fmt: skip
    layers = {i: "whole" for i in WHOLE}
    _engaged(before_e, after_e, layers, STEPS)
    _engaged(before_g, after_g, layers, STEPS, replayed=True)
    for rank in range(tp):
        assert torch.load(f"{tmp_path}/eager.rank{rank}")[0], rank
        assert not torch.load(f"{tmp_path}/graphs.rank{rank}")[0], rank
    check_logprobs_close(
        outputs_0_lst=[(o.token_ids, o.text, o.logprobs) for o in eager],
        outputs_1_lst=[(o.token_ids, o.text, o.logprobs) for o in graphs],
        name_0="eager",
        name_1="graphs",
    )
