# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 mono decode path through the real model: a five-layer
DeepSeek-V4.1-Flash at TP2 and TP4 on CDNA4, served with ``VLLM_ROCM_MONO_DECODE`` on
and off on the same random weights: every decoder layer's outputs at every
decode step must match. Unlike ``test_dsv41_mono_numerics.py``, which
checks the kernels against vLLM's ops one stage at a time, this runs
``DeepseekV4DecoderLayer.forward`` as the engine does -- the model's hooks,
the per-step path choice, the weights taken off the layer, the metadata and
the indexer's top-k -- so a change to the decoder layer that the kernels do
not follow fails here.

The five layers hold both kinds the mono path takes:
- 0: sliding window only -> the FFN launch
- 1: ratio-2 KV and index source -> the FFN launch
- 2: ratio-2 consumer -> the whole layer (K1 + K2)
- 3: ratio-1 KV, index and candidate source -> the FFN launch
- 4: ratio-1 consumer -> the whole layer
"""

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
    "index_source_layer_ids": [1, 3],
    "candidate_source_layer_id": 3,
    "engram_layer_ids": [],
    "engram_num_embeddings": [],
    "dspark_target_layer_ids": [2, 3, 4],
}
WHOLE, FFN = (2, 4), (0, 1, 3)
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

        layers = {
            i: None
            if m.mono is None
            else ["ffn" if m.mono.ffn_only else "whole", m.mono._weights is not None]
            for i, m in self._decoder_layers().items()
        }
        runner = mono_decode._runner
        return {
            "layers": layers,
            "epoch": -1 if runner is None else int(runner.epoch[0]),
        }


def _decode(vllm_runner, monkeypatch, tp: int, mono: bool, prompts, path: str):
    """Greedy decode of ``prompts`` with the mono path on or off: (each
    prompt's output, the workers' mono layers); each decoder layer's outputs
    for the decode tokens of decode-only steps, and vLLM's routing margins,
    saved under ``path``."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_ROCM_MONO_DECODE", "1" if mono else "0")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    with vllm_runner(
        MODEL,
        tensor_parallel_size=tp,
        load_format=LOAD_FORMAT,
        hf_overrides=TRUNCATED,
        worker_extension_cls=f"{__name__}.MonoProbe",
        moe_backend="aiter",
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
        block_size=128,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=len(prompts),
        # a prompt prefills in one step, so its decode tokens never share one
        max_num_batched_tokens=16384,
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


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
@pytest.mark.parametrize("tp", [2, 4])
def test_mono_decode_matches_the_model(vllm_runner, monkeypatch, tmp_path, tp):
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")
    g = torch.Generator().manual_seed(0)
    # contexts past both ratios' 512 compressed entries, so top-k selects
    lens = torch.randperm(1700, generator=g)[:8].add(300).tolist()  # distinct
    prompts = [torch.randint(1000, 60000, (n,), generator=g).tolist() for n in lens]

    on, probe_on = _decode(
        vllm_runner, monkeypatch, tp, True, prompts, f"{tmp_path}/on"
    )
    off, probe_off = _decode(
        vllm_runner, monkeypatch, tp, False, prompts, f"{tmp_path}/off"
    )

    # both layer kinds took the mono path on every rank, at every decode step
    launches = len(WHOLE + FFN)  # one launch (pair) a layer a step
    for rank in probe_on:
        assert rank["layers"] == {
            i: ["whole" if i in WHOLE else "ffn", True] for i in WHOLE + FFN
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
    # token. Layer 0 sees the same inputs on both paths at a request's first
    # decode token: there the bound is test_dsv41_mono_numerics' (about 3x the
    # kernels' 0.4%); its later tokens attend over KV each path wrote, 3%. Each
    # later layer takes the previous one's slightly different output, which
    # random weights amplify (2-4% by layer 4): 8%. A token whose top-6 routing
    # is a near-tie on vLLM's path (6th / 7th score within 5e-3: a 0.4% input
    # difference moves scores near 0.8 by about 3e-3) may pick another expert;
    # once it has, its residual stream differs and the later layers are exempt
    # for it. The residual stream: 3% (0.3% seen). A layer the kernels no
    # longer follow is off by far more, on every token.
    for rank in range(tp):
        got, _ = torch.load(f"{tmp_path}/on.rank{rank}")
        want, gaps = torch.load(f"{tmp_path}/off.rank{rank}")
        keys: list[tuple[int, int]] = sorted(
            same & {k[1:] for k in got} & {k[1:] for k in want}
        )
        assert len(keys) >= len(prompts), keys
        # the bound that sees a 3% error: a request's first decode token
        assert any(p == n for n, p in keys), keys
        flipped: set = set()
        for i in sorted(WHOLE + FFN):
            for k in keys:
                if k in flipped:
                    continue
                (x, res), (x_ref, res_ref) = got[(i, *k)], want[(i, *k)]
                err = _rel(res, res_ref)
                assert err < 0.03, (rank, i, k, "residual", err)
                bound = (0.015 if k[1] == k[0] else 0.03) if i == 0 else 0.08
                err = _rel(x, x_ref)
                if err >= bound:
                    assert gaps[(i, *k)] < 5e-3, (rank, i, k, "x", err)
                    flipped.add(k)
        assert len(flipped) <= len(keys) // 4, (rank, sorted(flipped))
