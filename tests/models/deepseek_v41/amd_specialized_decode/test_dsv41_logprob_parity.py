# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced logprob parity of `--monokernel` on and off for the
five-layer DeepSeek-V4.1-Flash of `test_dsv41_mono_layers` (its random
weights and layer variants, every layer whole on the mono path) at TP4 on
CDNA4, with a 4096-token vocabulary and full-vocabulary logprobs
(`tests/models/mono_logprob_parity`): 8 requests of 1100-2000 prompt tokens
(both compress ratios past their 512 compressed entries, so the indexers'
top-k selects), 32 forced decode positions each.

Through `D`'s weights the layers add ~0.5% (attention) and ~3% (FFN) to the
residual, so the logits barely depend on them: on/off agree to the BF16 logit
grid even with the last layer's attention or routing broken on a rank. More
gain in every layer makes the stack chaotic (FP8 activations amplify any
difference: off/off KL 0.3-0.7 with every E8M0 code +1), so only the last
layer gets it, and the test probes that layer's mono path. Its router's top-k
margins are small (median 0.019), so the difference of any two runs' MoE
inputs, on or off, moves its routing at 13-26 of the 256 positions, and those
positions take KL up to 0.32. Not ties within the router's own rounding (of
the 24 in one off/on pair, none is within `D._route_ties`), these are
compared apart: the routing is recomputed in float64 from each arm's own MoE
input (`Probe.routes`), a position where it differs is dropped, and at most
ROUTE_MOVES may be (a bug upstream of the router moves more; a bug in it
moves none). The rest hold the bounds but for OUTLIERS positions, which also
cover their top-1 flips at clear margins.

Calibrated on 11 runs (4 off, 4 on, 3 on in FULL graphs, each relaunched), 55
pairs: at most 26 positions moved (ROUTE_MOVES twice that), at most 1 kept
position over the bounds and 4 kept top-1 flips (OUTLIERS twice that); KL <=
0.0050 and |dlogprob| <= 0.163 at the 9th worst kept position (the bounds 3x
that). Bugs in the probe layer's mono path, on every rank (`MUTANTS`),
against the 4 off runs: the 9th worst kept KL over KL router_off_by_one
118-124x, shared_scale_off_by_one 88-94x, w2_scale_off_by_one 32-34x;
attention_dropped only 1.5-1.7x, but it moves 80-86 positions. Mono is not
bit-exact run to run (vLLM's prefill writes the caches it reads), so FULL
graphs are held to the same bounds as eager."""

import dataclasses

import pytest
import torch

from tests.models import mono_logprob_parity as P
from tests.models.deepseek_v41.amd_specialized_decode import (
    test_dsv41_mono_layers as D,
)
from vllm.model_executor.model_loader import register_model_loader

LOAD_FORMAT = "dsv41_parity_random"
TP = 4
VOCAB = 4096
REQUESTS, STEPS = 8, 33
# the head's scale: off entropy 4.65 nats, mean top-1 margin 0.68
HEAD_SCALE = 2.0
ENTROPY = (3.0, 6.0)
# the last layer's E8M0 codes +2 (attention: x16 through wo_a and wo_b) and +1
# (FFN)
PROBE_LAYER = 4
PROBE_SHIFT = {".attn.": 2, ".ffn.": 1}
ROUTE_MOVES = 52  # of 256 positions
OUTLIERS = 8
KL, DLP = 0.015, 0.5


@register_model_loader(LOAD_FORMAT)
class ProbeLoader(D.RandomWeightLoader):
    """`D.RandomWeightLoader`'s weights, PROBE_LAYER's scales shifted by
    PROBE_SHIFT, the vocabulary shards of the embedding and the head drawn per
    rank (alike, every token would have TP - 1 twins of equal logits) and the
    head scaled by HEAD_SCALE."""

    def load_weights(self, model, model_config) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        super().load_weights(model, model_config)
        rank = get_tensor_model_parallel_rank()
        for name, t in model.state_dict().items():
            if f"layers.{PROBE_LAYER}." in name and t.dtype == torch.uint8:
                for part, shift in PROBE_SHIFT.items():
                    if part in name and "scale" in name:
                        t.add_(shift)
        for name, t in model.named_parameters():
            if name.endswith(("model.embed_tokens.weight", "lm_head.weight")):
                D._fill(f"rank{rank}.{name}", t.data)
            if name.endswith("lm_head.weight"):
                t.data.mul_(HEAD_SCALE)


def _overrides(config):
    config = D._truncate(config)
    config.vocab_size = VOCAB
    return config


def _attn(w, **kw):
    return dataclasses.replace(w, attn=dataclasses.replace(w.attn, **kw))


# Bugs in PROBE_LAYER's mono path (its own view of the weights, which vLLM's
# path does not read).
MUTANTS = {
    # every expert id one off
    "router_off_by_one": lambda w: dataclasses.replace(
        w, gate_w=w.gate_w.roll(1, 0).contiguous(), bias=w.bias.roll(1, 0)
    ),
    # the attention's output never reaches the residual
    "attention_dropped": lambda w: _attn(
        w, wo_b_scale=torch.zeros_like(w.attn.wo_b_scale)
    ),
    # an E8M0 exponent one off: the routed experts' down projection doubled
    "w2_scale_off_by_one": lambda w: dataclasses.replace(w, w2_s=w.w2_s + 1),
    # the shared expert's gate_up doubled
    "shared_scale_off_by_one": lambda w: dataclasses.replace(w, sgu_s=w.sgu_s + 1),
}


class Probe(D.MonoProbe):
    def mutate(self, name: str) -> None:
        """PROBE_LAYER's mono weights through `MUTANTS[name]`, on every rank.
        Eager only: FULL graphs keep the pointers they captured."""
        layer = self._decoder_layers()[PROBE_LAYER]
        layer.mono._weights = MUTANTS[name](layer.mono.weights(layer))

    def capture_router(self) -> None:
        """Records PROBE_LAYER's MoE input at every decode step: off at its
        FFN, on from the mono scratch once the step is done (the last layer's,
        so FULL graph replays leave it there too)."""
        from vllm.models.deepseek_v41.amd import mono_decode
        from vllm.models.deepseek_v41.amd.mono.layer import scratch_layout

        layer = self._decoder_layers()[PROBE_LAYER]
        self._router_in: list[torch.Tensor] = []
        r = mono_decode._runner
        epoch: list[int | None] = [None if r is None else int(r.epoch[0])]

        def ffn(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            if x.shape[0] == REQUESTS:
                self._router_in.append(x.float().cpu())

        def logits(module, args):
            assert r is not None  # registered only when the layer runs mono
            if int(r.epoch[0]) == epoch[0]:  # the prefill
                return
            epoch[0] = int(r.epoch[0])
            tuned = r.tuned(REQUESTS, layer.attn.compress_ratio)
            lay = scratch_layout(REQUESTS, r.tp, r.experts, r.topk, r.ug_groups, tuned)
            off = lay["normed"][0]
            x = r.scratch(REQUESTS, tuned)[off : off + REQUESTS * D.HIDDEN * 2]
            x = x.view(torch.bfloat16).view(REQUESTS, D.HIDDEN)
            self._router_in.append(x.float().cpu())

        if layer.mono is None:
            layer.ffn.register_forward_pre_hook(ffn, with_kwargs=True)
        else:
            model = self.model_runner.model  # type: ignore[attr-defined]
            model.language_model.logits_processor.register_forward_pre_hook(logits)

    def routes(self) -> list:
        """PROBE_LAYER's top-k experts of each captured step and row (the
        request), in float64 from the MoE input."""
        ffn = self._decoder_layers()[PROBE_LAYER].ffn
        x = torch.stack(self._router_in).double()
        w, b = ffn.gate.weight.double().cpu(), ffn.gate.e_score_correction_bias
        score = torch.nn.functional.softplus(x @ w.T).sqrt() + b.double().cpu()
        return score.topk(ffn.n_activated_experts).indices.sort(-1).values.tolist()


def _requests():
    return P.forced_requests(VOCAB, REQUESTS, STEPS, prompt_len=(1100, 2000))


def _run(
    vllm_runner,
    monkeypatch,
    mono: bool,
    graphs: bool = False,
    mutant: str | None = None,
) -> tuple[P.Forced, torch.Tensor]:
    """The forced decode of `_requests` with mono on or off, eager or in
    FULL graphs, and its `Probe.routes` (one step a decode position); on,
    every layer took the mono path at every decode step."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    steps: dict = dict(enforce_eager=True)
    if graphs:
        steps = dict(
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [REQUESTS],
            }
        )
    prompts, traces = _requests()
    with vllm_runner(
        D.MODEL,
        tensor_parallel_size=TP,
        load_format=LOAD_FORMAT,
        hf_overrides=_overrides,
        worker_extension_cls=f"{__name__}.Probe",
        moe_backend="aiter",
        monokernel=mono,
        language_model_only=True,
        tokenizer_mode="deepseek_v41",
        skip_tokenizer_init=True,
        attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": True},
        block_size=128,
        max_model_len=4096,
        max_num_seqs=REQUESTS,
        max_num_batched_tokens=16384,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        enable_trace_replay=True,
        max_logprobs=-1,
        **steps,
    ) as runner:
        llm = runner.llm
        if mutant is not None:
            llm.collective_rpc("mutate", args=(mutant,))
        llm.collective_rpc("capture_steps", args=(False,))
        llm.collective_rpc("capture_router")
        before = llm.collective_rpc("mono_layers")
        run = P.run_forced(llm, prompts, traces)
        after = llm.collective_rpc("mono_layers")
        routes = torch.tensor(llm.collective_rpc("routes")[0])
    if mono:
        D._engaged(
            before, after, {i: "whole" for i in D.WHOLE}, STEPS - 1, replayed=graphs
        )
    else:
        assert all(set(a["layers"].values()) == {None} for a in after), after
    assert routes.shape[0] == STEPS - 1, routes.shape
    return run, routes


@pytest.fixture(scope="module")
def arm(vllm_runner):
    """`_run` once a module for each setting."""
    runs: dict = {}

    def get(**kw) -> tuple[P.Forced, torch.Tensor]:
        arch = D._arch()
        D._skip_unless(arch, TP)
        if arch != "cdna4":
            pytest.skip("the MXFP4 indexer cache needs CDNA4")
        key = tuple(sorted(kw.items()))
        if key not in runs:
            with pytest.MonkeyPatch.context() as mp:
                runs[key] = _run(vllm_runner, mp, **kw)
        return runs[key]

    return get


def _compare(off, on) -> tuple[dict, int]:
    """`P.compare` of two `_run`s without the positions where their
    PROBE_LAYER routing differs, and how many those are."""
    (a, routes_a), (b, routes_b) = off, on
    m = P.compare(a, b)
    moved = (routes_a != routes_b).any(-1)
    keep = torch.tensor([not moved[p - 1, r] for r, p in m["index"]])
    pos = {k: v[keep] for k, v in m["pos"].items()}
    index = [i for i, k in zip(m["index"], keep.tolist()) if k]
    return dict(pos=pos, index=index, clear=m["clear"][keep]), int((~keep).sum())


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "full_graphs"])
def test_mono_logprobs_match_vllm(arm, graphs):
    m, moved = _compare(arm(mono=False), arm(mono=True, graphs=graphs))
    P.assert_informative(m, entropy=ENTROPY, min_margin=0.3)
    assert moved <= ROUTE_MOVES, (moved, P.summary(m))
    P.assert_parity(m, KL, DLP, OUTLIERS)


@pytest.mark.parametrize("mutant", ["router_off_by_one"])
def test_mono_bug_fails_parity(arm, mutant):
    """The bounds are not vacuous: a bug in the mono path exceeds them 10x."""
    m, moved = _compare(arm(mono=False), arm(mono=True, mutant=mutant))
    assert moved <= ROUTE_MOVES, (moved, P.summary(m))
    assert P.worst(m, "kl", OUTLIERS) >= 10 * KL, P.summary(m)
