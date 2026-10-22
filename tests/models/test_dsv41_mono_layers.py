# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 specialized decode against vLLM's path, through the real
model: at TP2 and TP4 on CDNA4, and at TP4 on gfx942, where the indexer is
vLLM's FP8 one (the MXFP4 indexer cache needs CDNA4) and the kernels read the
copies of the weights that the model makes when it loads them."""

import zlib

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
        runs), its prompt length and position cannot. ``lens``: the prompt
        lengths, all distinct; a row's request is the longest prompt not past
        its position."""
        from vllm.forward_context import get_forward_context

        self._captured: dict = {}
        lens = sorted(lens)

        def record(layer, layer_id, pos, out) -> None:
            md = get_forward_context().attn_metadata
            if not isinstance(md, dict):
                return
            swa = md.get(layer.attn.swa_cache_layer.prefix)
            if swa is None or swa.num_prefills != 0:
                return
            for row, p in enumerate(pos.tolist()):
                n = max(n for n in lens if n <= p)
                kept = [t[row].float().cpu() for t in out[:2]]
                self._captured.setdefault((layer_id, n, p), kept)

        def hook(layer_id):
            def keep(module, args, kwargs, out):
                pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
                record(module, layer_id, pos, out)

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

        # A whole-layer launch replaces the layer's forward, so no module hook
        # sees it: wrap the launch itself to keep the same outputs.
        decode = getattr(self.model_runner.model, "specialized_decode", None)  # type: ignore[attr-defined]
        if decode is None:
            return
        kernel = decode.kernel
        launch_whole = kernel.launch_whole

        def launch(plan, x, positions, *args, **kwargs):
            out = launch_whole(plan, x, positions, *args, **kwargs)
            if out is not None:
                record(plan.layer, plan.layer.attn.layer_id, positions, out)
            return out

        kernel.launch_whole = launch

    def save_capture(self, path: str) -> None:
        rank = self.rank  # type: ignore[attr-defined]
        torch.save((self._captured, self._gaps), f"{path}.rank{rank}")

    def mono_layers(self) -> dict:
        from vllm.models.deepseek_v41.amd.specialized_decode.runtime.weights942 import (
            ATTN_LINEARS,
        )

        model = self.model_runner.model  # type: ignore[attr-defined]
        decode = getattr(model, "specialized_decode", None)
        plans = {} if decode is None else decode.kernel.plans

        def copies(plan) -> bool:
            """Whether the layer holds the gfx942 copies of its weights, which
            the model makes when it loads them (``mono942``)."""
            ffn = plan.layer.ffn
            held = [
                ffn.experts.routed_experts,
                ffn.shared_experts.gate_up_proj,
                ffn.shared_experts.down_proj,
            ]
            if not plan.ffn_only:
                held += [getattr(plan.layer.attn, n) for n in ATTN_LINEARS]
            return all(hasattr(t, "mono942") for t in held)

        layers = {
            i: (
                [
                    "ffn" if plans[i].ffn_only else "whole",
                    plans[i].weights is not None,
                    copies(plans[i]),
                ]
                if i in plans
                else None
            )
            for i in self._decoder_layers()
        }
        runner = None if decode is None else decode.kernel.runner
        return {
            "layers": layers,
            "epoch": -1 if runner is None else int(runner.epoch[0]),
        }


def _decode(
    vllm_runner, monkeypatch, tp: int, mono: bool, prompts, path: str, arch: str
):
    """Greedy decode of ``prompts`` with the mono path on or off: (each
    prompt's output, the workers' mono layers); each decoder layer's outputs
    for the decode tokens of decode-only steps, and vLLM's routing margins,
    saved under ``path``."""
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
        specialized_decode="monokernel" if mono else None,
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


@pytest.mark.parametrize(("tp", "rows"), [(2, 8), (4, 8), (4, 42)])
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
