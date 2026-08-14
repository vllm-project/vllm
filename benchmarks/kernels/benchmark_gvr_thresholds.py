# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Analyze exact speculative GVR thresholds on captured model inputs."""

import argparse
from collections import defaultdict
from pathlib import Path

import regex as re
import torch

_TOPK = 2048
_CAPACITY = 6144
_QUANTILES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35)
_NAME_RE = re.compile(r"gvr_b(?P<batch>\d+)_call(?P<call>\d+)\.pt")


def _summary(values: torch.Tensor) -> str:
    values = values.float()
    return (
        f"mean={values.mean():.1f} p50={values.quantile(0.5):.0f} "
        f"p95={values.quantile(0.95):.0f} max={values.max():.0f}"
    )


def _candidate_counts(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    columns = torch.arange(logits.shape[1]).unsqueeze(0)
    valid = columns < seq_lens.unsqueeze(1)
    return ((logits >= thresholds.unsqueeze(1)) & valid).sum(1)


def _load(path: Path) -> dict[str, torch.Tensor]:
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    seq_lens = tensors["seq_lens"]
    columns = torch.arange(tensors["logits"].shape[1]).unsqueeze(0)
    valid = columns < seq_lens.unsqueeze(1)
    if not tensors["logits"][valid].isfinite().all():
        raise ValueError(f"non-finite valid logits in {path}")
    if seq_lens.min() < _TOPK:
        raise ValueError(f"sequence shorter than top-k in {path}")
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    args = parser.parse_args()

    captures: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    quantile_counts: dict[float, list[torch.Tensor]] = defaultdict(list)
    overlaps: list[torch.Tensor] = []

    for path in sorted(args.capture_dir.glob("gvr_b*_call*.pt")):
        match = _NAME_RE.fullmatch(path.name)
        if match is None:
            continue
        batch = int(match["batch"])
        call = int(match["call"])
        tensors = _load(path)
        captures[batch, call] = tensors

        logits = tensors["logits"]
        hints = tensors["hints"].long()
        outputs = tensors["output_indices"].long()
        hint_scores = logits.gather(1, hints)
        sorted_hint_scores = hint_scores.sort(1).values
        overlap = torch.stack(
            [torch.isin(output, hint).sum() for output, hint in zip(outputs, hints)]
        )
        overlaps.append(overlap)

        print(f"batch={batch} call={call} overlap: {_summary(overlap)}")
        for quantile in _QUANTILES:
            offset = round(quantile * (_TOPK - 1))
            thresholds = sorted_hint_scores[:, offset]
            counts = _candidate_counts(logits, tensors["seq_lens"], thresholds)
            quantile_counts[quantile].append(counts)
            admitted = ((counts >= _TOPK) & (counts <= _CAPACITY)).float().mean()
            print(
                f"  hint-q{quantile:0.2f}: {_summary(counts)} admitted={admitted:.1%}"
            )

    if not captures:
        raise ValueError(f"no captures found in {args.capture_dir}")

    print("aggregate")
    print(f"  temporal overlap: {_summary(torch.cat(overlaps))}")
    for quantile in _QUANTILES:
        counts = torch.cat(quantile_counts[quantile])
        admitted = ((counts >= _TOPK) & (counts <= _CAPACITY)).float().mean()
        underflow = (counts < _TOPK).float().mean()
        overflow = (counts > _CAPACITY).float().mean()
        print(
            f"  hint-q{quantile:0.2f}: {_summary(counts)} "
            f"admitted={admitted:.1%} under={underflow:.1%} "
            f"over={overflow:.1%}"
        )

    print("previous numeric cutoff")
    for batch in sorted({batch for batch, _ in captures}):
        for previous_call, current_call in ((0, 21), (10, 31), (20, 41)):
            previous = captures.get((batch, previous_call))
            current = captures.get((batch, current_call))
            if previous is None or current is None:
                continue
            previous_scores = previous["logits"].gather(
                1, previous["output_indices"].long()
            )
            thresholds = previous_scores.min(1).values
            counts = _candidate_counts(
                current["logits"], current["seq_lens"], thresholds
            )
            admitted = ((counts >= _TOPK) & (counts <= _CAPACITY)).float().mean()
            print(
                f"  batch={batch} calls={previous_call}->{current_call}: "
                f"{_summary(counts)} admitted={admitted:.1%}"
            )


if __name__ == "__main__":
    main()
