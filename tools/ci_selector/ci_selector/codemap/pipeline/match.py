# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Match a step definition to the job slugs Buildkite actually reported.

A step's key and label imply a set of spellings a status context can take, and
a mirror or a shard spells it differently again. This lives with the step model
because both halves need it, and keeping it in `validate/` made an import cycle.
"""

from __future__ import annotations

import regex as re

# Shortest prefix Buildkite truncates a context to. Steps can share one; a
# drift guard bounds how many do.
TRUNC_MIN = 44


def _slug(label: str, plus_word: bool) -> str:
    """Two spellings. Buildkite's job slug writes `+`, `/` and `.` as words;
    ci-infra's key collapses them to a dash."""
    s = label.lower()
    if plus_word:
        for char, word in (("+", " plus "), ("/", " slash "), (".", " dot ")):
            s = s.replace(char, word)
    else:
        s = s.replace("+", " ")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return re.sub(r"-n$", "", s)


def amd_label(label: str, device: str | None) -> str:
    """ci-infra `amd.get_amd_label`. Pinned by the snapshot suite."""
    return f"AMD: {label} ({device or ''})"


def step_slug_candidates(step) -> list[str]:
    """Every status-context spelling a step can take: its key, the label slug
    with '+' spelled out or dropped, and for a mirror whichever label the
    generator would have used.

    A mirror is published under its own yaml label, or the parent wrap if it
    declares none. Emitting both invents a spelling naming two vendors at once,
    which nothing posts. Older revisions replay through the wrap.

    Each spelling is slugged whole, not assembled from slugged parts: `_slug`
    strips a trailing `-n`, which swallowed the `%N` of a sharded mirror.
    """
    out = []
    if step.mirror_hw:
        if step.mirror_label:
            for pw in (True, False):
                out.append(_slug(step.mirror_label, pw))
        else:
            base_label = step.label.rsplit(" (", 1)[0]
            for pw in (True, False):
                out.append(_slug(amd_label(base_label, None), pw))
                if step.device:
                    out.append(_slug(amd_label(base_label, step.device), pw))
    else:
        if step.key:
            out.append(step.key)
        for pw in (True, False):
            out.append(_slug(step.label, pw))
    return out


def slug_matches(ran_slug: str, cands: list[str], *, exact: bool) -> bool:
    """exact: literal equality. Otherwise the reported context is a cut-off
    prefix of a candidate, or a numeric shard of it ('lora-1' for a sharded
    'lora' step). A plain prefix is not enough, or 'engine' would claim
    'engine-2-gpus', a different job."""
    if exact:
        return ran_slug in cands
    return any(
        (c.startswith(ran_slug) and len(ran_slug) >= TRUNC_MIN)
        or re.fullmatch(re.escape(c) + r"-\d+", ran_slug)
        for c in cands
    )


def slug_matches_any(ran_slug: str, cands: list[str]) -> bool:
    return slug_matches(ran_slug, cands, exact=True) or slug_matches(
        ran_slug, cands, exact=False
    )


def match_jobs(ran: dict[str, str], steps: list) -> tuple[set, dict, dict]:
    """Which jobs a step set accounts for: (matched, unmatched, matched_by_step).

    Exact pass over every step first, truncation pass second, so a short-slugged
    step cannot absorb a longer job's status. Steps are walked in step_id order
    so credit for a job two steps both match is reproducible. The analyzer and
    the today-replica share this, or the trade measurement would be comparing
    two differently scored things.
    """
    unmatched = dict(ran)
    matched: set[str] = set()
    by_step: dict[str, list[str]] = {}
    ordered = sorted(steps, key=lambda s: s.step_id)
    for exact in (True, False):
        for step in ordered:
            cands = step_slug_candidates(step)
            hit = {r for r in unmatched if slug_matches(r, cands, exact=exact)}
            matched |= hit
            if hit:
                by_step.setdefault(step.step_id, []).extend(sorted(hit))
            for h in hit:
                unmatched.pop(h)
    return matched, unmatched, by_step
