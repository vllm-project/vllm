# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Pro `--monokernel` against vLLM's own path, end to end:
teacher-forced logprobs (`mono_logprob_parity`) of the five-layer random
DeepSeek-V4-Pro of `test_dsv4_mono_layers` at TP8 on CDNA4, every decode
step of every layer on the mono path in the on arm, eager and in FULL graphs.
Vocabulary 4096: full distributions, no tokenizer, no checkpoint. A quarter of
the prompts run past the CSA layers' 1024-entry top-k (4096 tokens), so the
indexer selects.

The weights are `test_dsv4_mono_layers`' (checkpoint ranges) but for:
- lm_head: `_fill` seeds by name, so every rank's vocabulary shard is alike
  and the top two logits always tie. A distinct shard a rank, std 0.04: the off
  arm's entropy 3.9 nats, mean top-1 margin 0.8.
- the bias-routed layers' (3, 4) bias, std 20: at `_fill`'s 0.02 their
  experts' near-ties flip under any change of rounding, off against off in
  reverse order too (KL over 1e-2 at 27 of 176 positions). At 20 the bias picks
  the experts by margins past rounding (none over 5e-3).

Scope: at checkpoint ranges the attention and the shared expert add about 1%
to the residual stream, so the logprobs see mostly the routed experts and the
router. Every wo_b E8M0 code one off (attention output x2) moves KL to 0.019
only; the shared expert's down scales one off, to the null's. Those stages are
held to derived bounds in `test_dsv4_mono_layers`. With the attention's
scales 4x, that wo_b bug reaches KL 1.4, but the two paths' own attention
roundings then reach 0.05 (2.5x off against off) and the expert roll below
is 7x that only.

Injected bugs (`Probe.mutate`, mono's own view of layer 3 on every rank),
their worst KL over `KL`: the down projections one expert off
(`expert_roll`) 74x, the busiest expert's down scales one off
(`expert_scale`) 10.3x, the top two experts' gate rows swapped
(`gate_swap`) 21x. `expert_roll` runs here."""

import dataclasses
import zlib

import pytest
import torch

from tests.models import mono_logprob_parity as P
from tests.models import test_dsv4_mono_layers as D
from vllm.model_executor.model_loader import register_model_loader

VOCAB = 4096
STEPS = 12
SHORT, LONG = 12, 4  # requests of 16-48 and of 4.4K-4.9K prompt tokens
LOAD_FORMAT = "dsv4_parity_random"
HEAD, BIAS = 0.04, 20.0
ENV = {
    "VLLM_ROCM_USE_AITER": "1",
    "VLLM_ROCM_USE_AITER_MOE": "1",
    "VLLM_ROCM_USE_AITER_LINEAR": "1",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS": "0",
    "VLLM_ROCM_FP8_PADDING": "1",
}
# Null (176 positions): off against off in reverse order, in FULL graphs,
# relaunched: max KL 4.3e-3, |dlogprob| 0.16; mono against mono alike: 4.9e-3,
# 0.16 (eager against FULL graphs is not bit-exact). On against off: 4.2e-3,
# 0.23. Bounds 3x the null; a top-1 flip counts where the off arm's margin is
# 2x the null's |dlogprob|.
KL, DLP, CLEAR = 1.5e-2, 0.45, 0.3


@register_model_loader(LOAD_FORMAT)
class RandomWeightLoader(D.RandomWeightLoader):
    def load_weights(self, model, model_config) -> None:
        from vllm.distributed import get_tensor_model_parallel_rank

        super().load_weights(model, model_config)
        for name, t in model.state_dict().items():
            if t.device.type == "meta":
                continue
            g = torch.Generator(device=t.device)
            if name == "lm_head.weight":
                g.manual_seed(get_tensor_model_parallel_rank())
                t.copy_(torch.randn(t.shape, generator=g, device=t.device) * HEAD)
            elif name.endswith("e_score_correction_bias"):
                g.manual_seed(zlib.crc32(name.encode()))
                t.copy_(torch.randn(t.shape, generator=g, device=t.device) * BIAS)


class Probe(D.MonoProbe):
    def mutate(self, kind: str, layer: int = 3) -> None:
        """A bug in the mono path's own view of one layer's weights."""
        lay = self._decoder_layers()[layer]
        w = lay.mono.weights(lay)
        k = w.w2.shape[1]
        if kind == "expert_roll":
            new = dict(
                w2=w.w2.view(torch.uint8).roll(1, 0).view(w.w2.dtype),
                w2_s=w.w2_s.roll(k, 0),
            )
        elif kind == "expert_scale":
            e = int(w.bias.argmax())
            new = dict(w2_s=w.w2_s.clone())
            new["w2_s"][e * k : (e + 1) * k] += 1
        elif kind == "gate_swap":
            top = w.bias.topk(2).indices
            new = dict(gate_w=w.gate_w.clone())
            new["gate_w"][top] = new["gate_w"][top.flip(0)]
        else:
            raise ValueError(kind)
        lay.mono._weights = dataclasses.replace(w, **new)


def _requests():
    short = P.forced_requests(VOCAB, SHORT, STEPS, seed=0)
    long = P.forced_requests(VOCAB, LONG, STEPS, seed=1, prompt_len=(4400, 4900))
    return short[0] + long[0], short[1] + long[1]


def _arm(vllm_runner, prompts, traces, mono, graphs=False, mutant=None):
    with vllm_runner(
        D.MODEL,
        tensor_parallel_size=D.TP,
        load_format=LOAD_FORMAT,
        hf_overrides={**D.TRUNCATED, "vocab_size": VOCAB},
        worker_extension_cls=f"{__name__}.Probe",
        moe_backend="aiter",
        monokernel=mono,
        kv_cache_dtype="fp8",
        block_size=256,
        skip_tokenizer_init=True,
        enforce_eager=not graphs,
        compilation_config=dict(
            cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[len(prompts)]
        ),
        max_model_len=8192,
        max_num_seqs=len(prompts),
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        enable_trace_replay=True,
        max_logprobs=-1,
    ) as runner:
        if mutant is not None:
            runner.llm.collective_rpc("mutate", args=(mutant,))
        before = runner.llm.collective_rpc("mono_layers")
        out = P.run_forced(runner.llm, prompts, traces)
        after = runner.llm.collective_rpc("mono_layers")
    # A layer launches at most once a step: eager, exactly one launch a layer a
    # decode-only step means none fell back (``layers`` alone does not: the
    # mutant's weights() call sets it); in FULL graphs at least that, as in
    # DSv4.1's ``_engaged``.
    launches = len(D.LAYERS) * (STEPS - 1)
    for b, a in zip(before, after):
        if not mono:
            assert set(a["layers"].values()) == {None}, a
            continue
        assert a["layers"] == dict.fromkeys(D.LAYERS, True), a
        moved = a["epoch"] - max(b["epoch"], 0)
        assert moved == launches or graphs and moved >= launches, (b, a)
        assert moved % len(D.LAYERS) == 0, (b, a)
    return out


@pytest.fixture(scope="module")
def forced(vllm_runner):
    """Each arm's teacher-forced run, once a module."""
    if not D._on_cdna4():
        pytest.skip("the mono kernels target CDNA4")
    if torch.accelerator.device_count() < D.TP:
        pytest.skip(f"needs {D.TP} GPUs")
    pytest.importorskip("flydsl")
    runs: dict = {}
    prompts, traces = _requests()
    with pytest.MonkeyPatch.context() as mp:
        for k, v in ENV.items():
            mp.setenv(k, v)

        def run(mono: bool, **kw):
            key = (mono, *sorted(kw.items()))
            if key not in runs:
                runs[key] = _arm(vllm_runner, prompts, traces, mono, **kw)
            return runs[key]

        yield run


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "full_graphs"])
def test_mono_matches_vllm(forced, graphs):
    """Mono's logprobs within the null's bounds of vLLM's eager ones."""
    m = P.compare(forced(False), forced(True, graphs=graphs), CLEAR)
    P.assert_informative(m, entropy=(3.0, 5.0), min_margin=0.3)
    P.assert_parity(m, KL, DLP)


@pytest.mark.parametrize("mutant", ["expert_roll"])
def test_injected_bug_fails(forced, mutant):
    """The bounds are not vacuous: a bug in the mono path's weights fails them
    by 10x (mutants in eager steps only: FULL graphs bake the pointers in)."""
    m = P.compare(forced(False), forced(True, mutant=mutant), CLEAR)
    assert P.worst(m, "kl") >= 10 * KL, P.summary(m)
