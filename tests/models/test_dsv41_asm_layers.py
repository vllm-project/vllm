# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 asm decode layers (``vllm/models/deepseek_v41/amd/asm``)
through the real model: the five-layer DeepSeek-V4.1-Flash of
``test_dsv41_mono_layers.py`` at TP4 on gfx950, served with ``--monokernel`` on
and off on the same random weights. Steps of 17-256 rows run on the asm
kernels with ``--monokernel`` on: K2 on every layer, K1 on the index and
consumer layers (1-4), and the Engram side chain. Every decoder layer's outputs
must match vLLM's path:

- on decode-only steps of every row count where the kernels change mode
  (17, 64 / 65, 128 / 129, 256 rows; eager);
- on mixed prefill + decode steps (the decode rows; vLLM's attention, ours
  around it);
- and the same steps replayed from CUDA graphs must give the eager run's
  tokens and logprobs."""

import pytest
import torch

from vllm.platforms import current_platform

from .test_dsv41_mono_layers import (
    LOAD_FORMAT,
    MODEL,
    STEPS,
    TRUNCATED,
    MonoProbe,
    _rel,
)
from .utils import check_logprobs_close

TP = 4
K1_LAYERS, K2_LAYERS = (1, 2, 3, 4), (0, 1, 2, 3, 4)
# decode-only steps of these row counts, one after another: request i stops
# after its last step (requests that reach the engine core after its first
# step shift them a little)
STEP_ROWS = (256, 129, 128, 65, 64, 17)
# the kernels' modes change at 64 / 65 and 128 / 129 rows
MODE_ROWS = ((17, 64), (65, 128), (129, 255), (256, 256))
# mixed steps: prefill chunks next to the decode rows within the 256 rows
MIXED_PROMPTS, MIXED_BUDGET = 32, 256


class AsmProbe(MonoProbe):
    def capture_decode_rows(self, lens: list[int]) -> None:
        """capture_steps, plus the decode rows of mixed steps (the batch puts
        them first), with their routing gaps."""
        from vllm.forward_context import get_forward_context

        self.capture_steps(lens)
        lens = sorted(lens)

        def mixed_rows(layer) -> int:
            md = get_forward_context().attn_metadata
            swa = (
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

    def asm_layers(self) -> dict:
        from vllm.models.deepseek_v41.amd.asm import decode, engram

        layers = {
            i: None
            if m.asm is None
            else [m.asm.attn_fn is not None, m.asm.ffn_fn is not None]
            for i, m in self._decoder_layers().items()
        }
        return {
            "layers": layers,
            "k1": sorted(decode._k1["seen"]),
            "k2": sorted(decode._k2["seen"]),
            "engram": sorted(engram._eg["seen"]),
            "mixed": "mixed" in decode._k1["logged"],
        }


def _skip_unless_gfx950_tp4() -> None:
    if not current_platform.is_rocm():
        pytest.skip("the asm decode kernels target gfx950")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the asm decode kernels target gfx950")
    if torch.accelerator.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    pytest.importorskip("flydsl")
    from vllm.models.deepseek_v41.amd.asm.kernels.build import toolchain_missing

    if missing := toolchain_missing():
        pytest.skip(f"needs {missing} to build the kernels")


def _prompts(n: int, lo: int, gap: int, copies: int = 1) -> list[list[int]]:
    """N random prompts of distinct lengths from lo on, more than ``gap`` apart
    (the captures key a token by prompt length and position), each ``copies``
    times (a copy's rows are the same, so they share the keys)."""
    g = torch.Generator().manual_seed(0)
    lens = (torch.randperm(n, generator=g) * (gap + 1) + lo).tolist()
    prompts = [torch.randint(1000, 60000, (k,), generator=g).tolist() for k in lens]
    return [p for p in prompts for _ in range(copies)]


def _run(
    vllm_runner,
    monkeypatch,
    prompts,
    max_tokens: list[int],
    mono: bool,
    path: str | None,
    graph: bool = False,
    budget: int | None = None,
):
    """Greedy outputs (+ the probe on every rank); path: capture every decode
    row's layer outputs there (eager only)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    graphs = (
        dict(
            enforce_eager=False,
            compilation_config={
                "decode_cudagraph_capture_request_sizes": sorted(STEP_ROWS),
                "prefill_cudagraph_capture_token_sizes": [],
            },
        )
        if graph
        else dict(enforce_eager=True)
    )
    with vllm_runner(
        MODEL,
        tensor_parallel_size=TP,
        load_format=LOAD_FORMAT,
        hf_overrides=TRUNCATED,
        worker_extension_cls=f"{__name__}.AsmProbe",
        moe_backend="aiter",
        monokernel=mono,
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
        engram_config={"cpu_offload": True},
        block_size=128,
        max_model_len=4096,
        max_num_seqs=len(prompts),
        # budget None: every prompt prefills in one step, so that the decode
        # steps have all the rows
        max_num_batched_tokens=budget or sum(len(p) for p in prompts),
        enable_chunked_prefill=budget is not None,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        **graphs,
    ) as runner:
        if path is not None:
            runner.llm.collective_rpc(
                "capture_decode_rows", args=([len(p) for p in prompts],)
            )
        params = [
            SamplingParams(temperature=0.0, max_tokens=t, ignore_eos=True, logprobs=5)
            for t in max_tokens
        ]
        outs = runner.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params
        )
        if path is not None:
            runner.llm.collective_rpc("save_capture", args=(path,))
        probe = runner.llm.collective_rpc("asm_layers")
    return [o.outputs[0] for o in outs], probe


def _check_layers(prompts, on, off, tmp_path) -> None:
    """Each layer's outputs for the decode tokens both engines produced alike,
    asm against vLLM's path, token by token. The all-reduces carry MXFP8 here
    (bf16 on vLLM's path), which puts every layer's output 7% off vLLM's in the
    median and 8.4% at most: 12%. The residual stream follows at under 1% (0.8%
    seen): 2%. A token whose top-6 routing is a near tie on vLLM's path (6th /
    7th scores within 5e-3 at layer 0, 1e-2 past it, as in
    test_dsv41_mono_layers.py) may pick another expert and land far off (48-77%
    seen); the later layers are exempt for it, and at most a quarter of the
    tokens may flip (20% seen)."""
    same: set[tuple[int, int]] = set()
    for prompt, a, b in zip(prompts, on, off):
        for step in range(len(a.token_ids)):
            if a.token_ids[: step + 1] != b.token_ids[: step + 1]:
                break
            same.add((len(prompt), len(prompt) + step))
    for rank in range(TP):
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
                assert err < 0.02, (rank, i, k, "residual", err)
                err = _rel(x, x_ref)
                if err >= 0.12:
                    tie = 5e-3 if i == 0 else 1e-2
                    assert gaps[(i, *k)] < tie, (rank, i, k, "x", err)
                    flipped.add(k)
        assert len(flipped) <= len(keys) // 4, (rank, sorted(flipped))


def _check_probe(probe_on, probe_off, ranges) -> None:
    """The asm kernels took the layers and steps in each row range on every
    rank."""
    for rank in probe_on:
        assert rank["layers"] == {
            i: [i in K1_LAYERS, i in K2_LAYERS] for i in K2_LAYERS
        }, rank
        for part in ("k1", "k2", "engram"):
            for lo, hi in ranges:
                assert any(lo <= m <= hi for m in rank[part]), (part, lo, hi, rank)
    assert all(set(r["layers"].values()) == {None} for r in probe_off)


def _step_rows_max_tokens() -> list[int]:
    """Per request: 1 + the decode steps it takes, so that decode step s has
    STEP_ROWS[s - 1] rows."""
    out: list[int] = []
    for s, rows in reversed(list(enumerate(STEP_ROWS, 1))):
        out += [s + 1] * (rows - len(out))
    return out


def test_asm_decode_steps_of_every_mode(vllm_runner, monkeypatch, tmp_path):
    """Decode-only steps of 256, 129, 128, 65, 64 and 17 rows (eager), then
    the same requests from CUDA graphs: the eager run's tokens and logprobs."""
    _skip_unless_gfx950_tp4()
    max_tokens = _step_rows_max_tokens()
    # short prompts: all of them prefill in one step
    prompts = _prompts(len(max_tokens) // 8, 128, len(STEP_ROWS) + 1, copies=8)
    on, probe_on = _run(
        vllm_runner, monkeypatch, prompts, max_tokens, True, f"{tmp_path}/on"
    )
    off, probe_off = _run(
        vllm_runner, monkeypatch, prompts, max_tokens, False, f"{tmp_path}/off"
    )
    _check_probe(probe_on, probe_off, MODE_ROWS)
    _check_layers(prompts, on, off, tmp_path)

    # graph replay: the captured steps take the kernels as eagerly
    replay, probe_graph = _run(
        vllm_runner, monkeypatch, prompts, max_tokens, True, None, graph=True
    )
    _check_probe(probe_graph, probe_off, MODE_ROWS)
    check_logprobs_close(
        outputs_0_lst=[(o.token_ids, o.text, o.logprobs) for o in on],
        outputs_1_lst=[(o.token_ids, o.text, o.logprobs) for o in replay],
        name_0="eager",
        name_1="graph",
    )


def test_asm_decode_mixed_steps(vllm_runner, monkeypatch, tmp_path):
    """Prefill chunks next to decode rows in steps of at most 256 rows: the
    decode rows' layer outputs."""
    _skip_unless_gfx950_tp4()
    # contexts past both ratios' 512 compressed entries
    prompts = _prompts(MIXED_PROMPTS, 1100, STEPS + 1)
    max_tokens = [STEPS + 1] * len(prompts)
    run = dict(budget=MIXED_BUDGET)
    on, probe_on = _run(
        vllm_runner, monkeypatch, prompts, max_tokens, True, f"{tmp_path}/on", **run
    )
    off, probe_off = _run(
        vllm_runner, monkeypatch, prompts, max_tokens, False, f"{tmp_path}/off", **run
    )
    _check_probe(probe_on, probe_off, ((17, 256),))
    assert all(r["mixed"] for r in probe_on), probe_on
    _check_layers(prompts, on, off, tmp_path)
