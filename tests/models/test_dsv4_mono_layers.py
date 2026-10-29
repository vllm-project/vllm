# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4-Pro mono decode path through the real model: a five-layer
DeepSeek-V4-Pro at TP8 on CDNA4, served with ``--monokernel`` on and off on
the same random weights: every decoder layer's outputs at every decode step
must match. Unlike ``test_dsv4_mono_numerics.py``, which checks the kernels
against vLLM's ops one stage at a time, this runs
``DeepseekV4DecoderLayer.forward`` in the engine -- the model's hook, the
per-step path choice, the weights taken off the layer, the metadata, the
compressor and the indexer's top-k -- so a change to the decoder layer that
the kernels do not follow fails here. Scope: eager steps of one token a
request; speculative decoding's multi-token rows and graph replays are not
covered.

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
    elif name.endswith(("attn_hc.scale", "ffn_hc.scale")):
        t.copy_(randn(0.5).abs() + 0.5)
    elif name.endswith(("attn_hc.base", "ffn_hc.base", "sinks")):
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
            extract_layer_index(m.self_attn.prefix): m
            for m in model.modules()
            if isinstance(m, DeepseekV4DecoderLayer)
        }

    def capture_steps(self, lens: list[int]) -> None:
        """Keep each decoder layer's outputs (x, residual, post_mix, res_mix)
        for every decode token taken in a decode-only step (the steps the
        mono path takes), keyed by (layer, prompt length, position): a
        request's step and row can differ between engines, its prompt length
        and position cannot.
        ``lens``: the prompt lengths, more than ``STEPS`` apart; a row's
        request is the longest prompt not past its position."""
        from vllm.forward_context import get_forward_context

        self._captured: dict = {}
        self._gaps: dict = {}
        lens = sorted(lens)

        def decode_only(layer) -> bool:
            md = get_forward_context().attn_metadata
            if not isinstance(md, dict):
                return False
            swa = md.get(layer.self_attn.swa_cache_layer.prefix)
            return swa is not None and swa.num_prefills == 0

        def keys(layer_id, pos):
            for row, p in enumerate(pos.tolist()):
                yield row, (layer_id, max(n for n in lens if n <= p), p)

        def hook(layer_id):
            def keep(module, args, kwargs, out):
                if not decode_only(module):
                    return
                pos = kwargs.get("positions", args[1] if len(args) > 1 else None)
                for row, k in keys(layer_id, pos):
                    kept = [t[row].float().cpu() for t in out[:4]]
                    self._captured.setdefault(k, kept)

            return keep

        def gap_hook(layer_id, layer):
            def keep(ffn, args, kwargs):
                # vLLM's path of a bias-routed layer only (the mono kernels
                # route inside): each row's top-6 margin, sqrt(softplus(logits))
                # + bias as vLLM routes
                if ffn.gate.tid2eid is not None or not decode_only(layer):
                    return
                h = args[0] if args else kwargs["hidden_states"]
                logits = h.float() @ ffn.gate.weight.float().T
                score = torch.nn.functional.softplus(logits).sqrt()
                score = score + ffn.gate.e_score_correction_bias.float()
                top = score.topk(7, dim=-1).values
                gap = (top[:, 5] - top[:, 6]).tolist()
                for row, k in keys(layer_id, self._positions):
                    self._gaps.setdefault(k, gap[row])

            return keep

        def pos_hook(module, args, kwargs):
            self._positions = kwargs.get(
                "positions", args[1] if len(args) > 1 else None
            )

        for i, layer in self._decoder_layers().items():
            layer.register_forward_hook(hook(i), with_kwargs=True)
            layer.register_forward_pre_hook(pos_hook, with_kwargs=True)
            gap = gap_hook(i, layer)
            layer.mlp.register_forward_pre_hook(gap, with_kwargs=True)

    def save_capture(self, path: str) -> None:
        rank = self.rank  # type: ignore[attr-defined]
        torch.save((self._captured, self._gaps), f"{path}.rank{rank}")

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


def _decode(vllm_runner, monkeypatch, mono: bool, prompts, path: str):
    """Greedy decode of ``prompts`` with the mono path on or off: (each
    prompt's output, the workers' mono layers); each decoder layer's outputs
    for the decode tokens of decode-only steps, and vLLM's routing margins,
    saved under ``path``."""
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
        monokernel=mono,
        kv_cache_dtype="fp8",
        # the serving page (vllm_runner's default 16 holds no C128 row)
        block_size=256,
        tokenizer_mode="deepseek_v4",
        # eager: the hooks read every step (the kernels take eager steps as
        # they take FULL graph replays)
        enforce_eager=True,
        max_model_len=32768,
        max_num_seqs=len(prompts),
        # the steps mixing prefill and decode take vLLM's path and are not
        # captured (the runner turns chunked prefill off: a step's tokens
        # must cover the model length)
        max_num_batched_tokens=32768,
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
    assert a.shape == b.shape, (a.shape, b.shape)
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
def test_mono_decode_matches_the_model(vllm_runner, monkeypatch, tmp_path):
    if torch.accelerator.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    pytest.importorskip("flydsl")
    g = torch.Generator().manual_seed(0)
    # contexts past the CSA layers' 1024 top-k (4 tokens a compressed entry),
    # so the indexer selects; 512 apart, so a row's prompt is unambiguous
    jitter = torch.randint(256, (8,), generator=g).tolist()
    lens = [4400 + 512 * i + r for i, r in enumerate(jitter)]
    prompts = [torch.randint(1000, 60000, (n,), generator=g).tolist() for n in lens]

    on, probe_on = _decode(vllm_runner, monkeypatch, True, prompts, f"{tmp_path}/on")
    off, probe_off = _decode(
        vllm_runner, monkeypatch, False, prompts, f"{tmp_path}/off"
    )

    # every layer took the mono path on every rank, at every decode step
    for rank in probe_on:
        assert rank["layers"] == dict.fromkeys(LAYERS, True), rank
        assert rank["epoch"] >= len(LAYERS) * STEPS, rank
        assert rank["epoch"] % len(LAYERS) == 0, rank
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
    # token. Layer 0 sees the same inputs on both paths, but its FFN input and
    # output differ by up to test_dsv4_mono_numerics' 2% / 3% (MXFP8
    # activations against vLLM's FP8 a 128), which random weights amplify
    # through the MoE: 5%. Each later layer takes the previous one's slightly
    # different output: 10%. The hash layers' (0-2) experts are fixed by the
    # token; a bias-routed layer's token (3, 4: past the hash layers, its input
    # carries their differences) whose top-6 routing is a near-tie on vLLM's
    # path may pick another expert. The margin: the FFN input's own MXFP8
    # difference (2e-2), plus twice the token's residual difference (a
    # logit moves by its std, 1.7, times the input's relative difference, a
    # top score by a quarter of that, both scores either way). Once it has,
    # its residual stream differs and the later layers are exempt for it.
    # The residual stream, set before the FFN: 4% (2.2% the most measured,
    # layer 4). Its mixes for the next seam (post_mix in (0, 2), res_mix in
    # [0, 1]; the last layer's feed the model's hc_post) are sigmoids of it,
    # whose relative difference blows up where they saturate: 0.1 absolute
    # (0.055 measured). With random weights about a third of the tokens
    # reach a near-tie by layer 4 (every one within its margin); at most half
    # may. A layer the kernels no longer follow is off by far more, on every
    # token.
    for rank in range(TP):
        got, _ = torch.load(f"{tmp_path}/on.rank{rank}")
        want, gaps = torch.load(f"{tmp_path}/off.rank{rank}")
        keys: list[tuple[int, int]] = sorted(
            same & {k[1:] for k in got} & {k[1:] for k in want}
        )
        assert len(keys) >= len(prompts), keys
        flipped: set = set()
        for i in LAYERS:
            for k in keys:
                if k in flipped:
                    continue
                (x, *stream), (x_ref, *stream_ref) = got[(i, *k)], want[(i, *k)]
                res_err = _rel(stream[0], stream_ref[0])
                assert res_err < 0.04, (rank, i, k, "residual", res_err)
                for name, a, b in zip(
                    ("post_mix", "res_mix"), stream[1:], stream_ref[1:]
                ):
                    err = (a.float() - b.float()).abs().max().item()
                    assert err < 0.1, (rank, i, k, name, err)
                tie = 2e-2 + 2 * res_err
                err = _rel(x, x_ref)
                if err >= (0.05 if i == 0 else 0.10):
                    gap = gaps.get((i, *k), float("inf"))  # none: hash-routed
                    assert gap < tie, (rank, i, k, "x", err, gap, tie)
                    flipped.add(k)
        assert len(flipped) <= len(keys) // 2, (rank, sorted(flipped))
