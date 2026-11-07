# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DeepSeek-V4.1 asm decode layers (`vllm/models/deepseek_v41/amd/asm`)
through the real model: a six-layer DeepSeek-V4.1-Flash on the random weights of
`test_dsv41_mono_layers.py` at TP4 on gfx950, served with `--monokernel` on
and off. Its layers take every path the real model's do: a dense layer (0), an
Engram on a ratio-0 layer (1: our seam, vLLM's attention), an Engram on an index
source (2), a consumer (3), the candidate source (4) and a layer indexing over
its candidates (5). Steps of 17-256 rows run on the asm kernels with
`--monokernel` on, and every decoder layer's outputs must match vLLM's path:

- on decode-only steps of every row count where the kernels change mode
  (aimed at 256, 129, 128, 65, 64 and 17 rows), at a long --max-model-len (the
  indexer's wide, candidate-source and gather kernels);
- on mixed steps (a prompt's prefill next to 40 decode rows: the decode rows);
- and the decode-only steps replayed from CUDA graphs (the indexer's narrow
  kernels) give the eager run's tokens.

Three engines (on, off, graph replay), so the step stays short."""

import os
from typing import Any

import pytest
import torch

from tests.models.test_dsv41_mono_layers import (
    LOAD_FORMAT,
    MODEL,
    TRUNCATED,
    MonoProbe,
    _rel,
)
from vllm.platforms import current_platform

TP = 4
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
        runners = [decode._wob["wob"], decode._comb["car"]]
        for side in decode._eg["sides"]:
            runners += [side.ag_rows, side.ag_gate]
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
            "timeouts": sum(r.errors() for r in runners if r is not None),
        }


def _skip_unless_gfx950_tp4() -> None:
    """Skips where the asm kernels can't run; on CI's gfx950 jobs a missing
    dependency fails instead (no green job without the asm path)."""
    if not current_platform.is_rocm():
        pytest.skip("the asm decode kernels target gfx950")
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the asm decode kernels target gfx950")
    if torch.accelerator.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
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
    # vLLM's default, so the asm all-reduces sum in bf16 as vLLM's do
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "NONE")
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
        tensor_parallel_size=TP,
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
# layer 5; 17-21% of the tokens flip.
X_TOL, RES_TOL = 0.10, 0.012


def _check_layers(prompts, on, off, tmp_path) -> None:
    """Each layer's outputs for the decode tokens both engines produced alike,
    asm against vLLM's path, token by token: x within X_TOL, the residual stream
    within RES_TOL. A token whose top-6 routing is a near
    tie on vLLM's path (6th / 7th scores within 5e-3 at layer 0, 2e-2 past it:
    the residual stream drifts there, and a flip at 0.0106 was seen, none past
    0.015 in 662 tokens) may pick another expert and land far off (48-77%
    seen); the later layers are exempt for it, and at most a quarter of the
    tokens may flip (17-21% seen)."""
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
                assert err < RES_TOL, (rank, i, k, "residual", err)
                err = _rel(x, x_ref)
                errs.append(err)
                if err >= X_TOL:
                    tie = 5e-3 if i == 0 else 2e-2
                    assert gaps[(i, *k)] < tie, (rank, i, k, "x", err)
                    flipped.add(k)
        assert len(flipped) <= len(keys) // 4, (rank, sorted(flipped))
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
        assert rank["timeouts"] == 0, rank


def test_asm_decode_layers(vllm_runner, monkeypatch, tmp_path):
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
