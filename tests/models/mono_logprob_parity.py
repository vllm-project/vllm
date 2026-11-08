# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Teacher-forced logprob parity of --monokernel on and off, on random weights.

Both arms decode the same prompts through the same forced tokens (trace
replay), so their contexts are identical at every position and a difference
cannot cascade: each position's distributions differ by that step's arithmetic
only. Position 0 of each request comes from the prefill, which runs stock in
both arms, and is dropped. Trace replay also forces speculative decoding's
verify steps (each draft accepted only if it is the trace's token), but their
widths change the stock ops' rounding: compare arms with the same speculative
config, never one against a run without it.

A test builds two engines alike but for the mono switch, runs ``run_forced``
on each, then ``assert_informative`` on ``compare(off, on)`` (random weights
must give logits of a sane entropy and top-1 margin, else the metrics are
vacuous) and ``assert_parity`` with bounds calibrated from off-vs-off runs and
checked against bugs injected into the mono path (``worst``). High-gain random
stacks have heavy-tailed run-to-run noise, so a model may need an outlier
allowance; it must be justified (measured or derived)."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class Forced:
    """Flat per decode position: the forced token, the returned logprobs
    ({token id: logprob}) and the (request, position) it came from."""

    tokens: list[int]
    logprobs: list[dict[int, float]]
    index: list[tuple[int, int]]


def forced_requests(
    vocab: int,
    n: int,
    steps: int,
    seed: int = 0,
    prompt_len: tuple[int, int] = (16, 48),
) -> tuple[list[list[int]], list[list[int]]]:
    """``n`` prompts of random lengths in ``prompt_len`` and ``n`` traces of
    ``steps`` tokens, uniform ids in [0, vocab), alike every run."""
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(*prompt_len, (n,), generator=g).tolist()
    prompts = [torch.randint(vocab, (m,), generator=g).tolist() for m in lens]
    traces = [torch.randint(vocab, (steps,), generator=g).tolist() for _ in lens]
    return prompts, traces


@contextmanager
def _all_in_first_step(llm):
    """The engine core in its own process schedules requests as they arrive,
    so the first steps' composition (and the stock ops' rounding) would vary
    run to run: hold its scheduler until ``generate`` has added them all."""
    client = llm.llm_engine.engine_core
    if not hasattr(client, "call_utility"):  # in process: added before a step
        yield
        return
    client.call_utility("pause_scheduler", "keep", False)
    run = llm._run_engine
    paused = True

    def resumed(*args, **kwargs):
        nonlocal paused
        client.call_utility("resume_scheduler")
        paused = False
        return run(*args, **kwargs)

    llm._run_engine = resumed
    try:
        yield
    finally:
        del llm._run_engine
        if paused:
            client.call_utility("resume_scheduler")


def run_forced(
    llm,
    prompts: list[list[int]],
    traces: list[list[int]],
    k: int = -1,
) -> Forced:
    """Decode every prompt through its trace with ``k`` logprobs a position
    (-1: the full vocabulary; needs ``max_logprobs=-1``) on an engine built
    with ``enable_trace_replay=True``."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = [
        SamplingParams(
            max_tokens=len(trace),
            temperature=0,
            logprobs=k,
            trace_decode_token_ids=trace,
            detokenize=False,
            ignore_eos=True,
        )
        for trace in traces
    ]
    with _all_in_first_step(llm):
        outs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            params,
            use_tqdm=False,
        )
    run = Forced([], [], [])
    for r, (out, trace) in enumerate(zip(outs, traces)):
        got = out.outputs[0]
        assert list(got.token_ids) == trace, (r, list(got.token_ids), trace)
        assert got.logprobs is not None and len(got.logprobs) == len(trace), r
        for pos in range(1, len(trace)):
            run.tokens.append(trace[pos])
            run.logprobs.append({t: lp.logprob for t, lp in got.logprobs[pos].items()})
            run.index.append((r, pos))
    return run


def _dense(a: dict[int, float], b: dict[int, float]):
    """Both distributions over the tokens both returned, each renormalized
    there (a no-op over the full vocabulary)."""
    ids = sorted(a.keys() & b.keys())
    la = torch.tensor([a[t] for t in ids], dtype=torch.float64)
    lb = torch.tensor([b[t] for t in ids], dtype=torch.float64)
    return ids, la.log_softmax(0), lb.log_softmax(0)


def compare(off: Forced, on: Forced, clear_margin: float = 0.1) -> dict:
    """Per position (``pos``): KL(off || on), |delta logprob| of the forced
    token, the off arm's entropy and top-1 margin (nats), whether the two
    top-1 tokens agree, and the support overlap (the share of the larger
    returned token set both arms return: 1 over the full vocabulary, less
    when a top-k set moves); ``clear``: the positions whose off margin exceeds
    ``clear_margin``."""
    assert off.tokens == on.tokens and off.index == on.index, "arms forced apart"
    kl, dlp, ent, margin, agree, overlap = [], [], [], [], [], []
    for tok, a, b in zip(off.tokens, off.logprobs, on.logprobs):
        ids, la, lb = _dense(a, b)
        assert len(ids) >= 2 and tok in a and tok in b, (tok, len(ids))
        overlap.append(len(ids) / max(len(a), len(b)))
        p = la.exp()
        kl.append(float((p * (la - lb)).sum()))
        dlp.append(abs(a[tok] - b[tok]))
        ent.append(float(-(p * la).sum()))
        top = la.topk(2)
        margin.append(float(top.values[0] - top.values[1]))
        agree.append(ids[int(top.indices[0])] == ids[int(lb.argmax())])
    pos = dict(
        kl=torch.tensor(kl).clamp(min=0),
        dlp=torch.tensor(dlp),
        entropy=torch.tensor(ent),
        margin=torch.tensor(margin),
        agree=torch.tensor(agree),
        overlap=torch.tensor(overlap),
    )
    return dict(pos=pos, index=off.index, clear=pos["margin"] > clear_margin)


def worst(m: dict, key: str, outliers: int = 0) -> float:
    """The ``outliers + 1``-th largest of a per-position metric ("kl",
    "dlp"): a mutant's, over the bound, tells how far it fails."""
    v = m["pos"][key].sort(descending=True).values
    return v[min(outliers, len(v) - 1)].item()


def flips(m: dict) -> int:
    return int((m["clear"] & ~m["pos"]["agree"]).sum())


def summary(m: dict) -> str:
    pos = m["pos"]
    kl, dlp = pos["kl"].sort(descending=True), pos["dlp"].sort(descending=True)
    top = [
        (m["index"][i], f"{kl.values[j]:.3g}", f"{pos['dlp'][i]:.3g}")
        for j, i in enumerate(kl.indices[:4].tolist())
    ]
    return (
        f"{len(pos['kl'])} positions: KL max {kl.values[0]:.3g} "
        f"p99 {pos['kl'].quantile(0.99):.3g}, |dlogprob| max {dlp.values[0]:.3g} "
        f"p99 {pos['dlp'].quantile(0.99):.3g}, top-1 flips {flips(m)} of "
        f"{int(m['clear'].sum())} clear; off entropy "
        f"{pos['entropy'].mean():.2f} nats, mean margin "
        f"{pos['margin'].mean():.3f}, support overlap min "
        f"{pos['overlap'].min():.3f}; worst (request, position), KL, "
        f"|dlogprob|: {top}"
    )


def assert_informative(
    m: dict, entropy: tuple[float, float], min_margin: float = 0.1
) -> None:
    """The off arm's logits are neither near-uniform nor near-one-hot: their
    mean entropy within ``entropy`` (nats), their mean top-1 margin at least
    ``min_margin`` (a mean: BF16 logits tie their top two often)."""
    lo, hi = entropy
    pos = m["pos"]
    ent, margin = pos["entropy"].mean().item(), pos["margin"].mean().item()
    assert lo <= ent <= hi and margin >= min_margin, summary(m)


def assert_parity(
    m: dict, kl: float, dlp: float, outliers: int = 0, min_overlap: float = 0.0
) -> None:
    """All positions but at most ``outliers`` (the model's justified
    allowance, each isolated by the teacher forcing) within ``kl`` and
    ``dlp`` and with a support overlap of at least ``min_overlap`` (top-k
    logprobs: KL over a shrunken intersection hides a moved top-k set), and
    as many top-1 flips at clear margins at most."""
    pos = m["pos"]
    bad = (pos["kl"] > kl) | (pos["dlp"] > dlp) | (pos["overlap"] < min_overlap)
    over = int(bad.sum())
    assert over <= outliers and flips(m) <= outliers, (
        f"{over} positions over KL {kl:.3g} or |dlogprob| {dlp:.3g} or under "
        f"overlap {min_overlap:.3g} ({outliers} allowed); {summary(m)}"
    )
