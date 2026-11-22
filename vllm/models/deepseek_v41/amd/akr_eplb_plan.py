# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert placement planner for `kernel_config.amd_kernels.expert_perm`
(DeepSeek-V4.1 MegaMoE experts on EP8). CPU only; not used at runtime.

Input: the routed-expert counts of a serving run, `expert_counts_dp<r>.npz` of
every DP rank (`VLLM_AKR_EXPERT_COUNTS=<dir>`, `akr_expert_counts`) or any
npz with the same `counts_decode`, `counts_prefill`, `layer_ids` (and
`draft_*`) arrays; several files are summed. Output: a plan file that
`akr_eplb.Plan` loads.

Per layer, with `V[k, e]` the normalised load of logical expert `e` under
objective `k` (`2d`: decode-only and prefill-carrying steps as two objectives,
`all`: their sum, `decode`: decode steps only):

1. Replicas: every expert gets one slot; each of the `ep * S - E` extra slots
   goes to the expert with the largest per-replica load `max_k V[k, e] /
   nrep[e]`, at most one replica per rank (ties: lower expert id).
2. Placement: replicas in descending per-replica load (longest processing time
   first), each to the rank with a free slot and no replica of that expert
   that minimises `max_k (load[rank, k] + V[k, e] / nrep[e])` (ties: smaller
   sum, lower rank).
3. Slots within a rank are sorted by logical id; slot `p` lives on rank
   `p // S`.

The planner assumes that an expert's load splits evenly over its replicas,
which the runtime's per-route hash does on average. The shipped
`configs/expert_plan_S49.npz` is
`--slots 49 --mode 2d --draft-mode all` (draft slots 17) on the counts of
`tests/models/deepseek_v41/amd/data/expert_counts_agentx_c1024.npz`.

Usage:
    python -m vllm.models.deepseek_v41.amd.akr_eplb_plan \\
        --counts <dir or npz> [...] --slots 49 --out plan.npz
"""

import argparse
import glob
import math
import os

import numpy as np

from vllm.models.deepseek_v41.amd.akr_eplb import (
    DRAFT_EXPERTS,
    TARGET_EXPERTS,
    LayerPerm,
    Plan,
)

RANKS = 8
MODES = ("2d", "all", "decode")
# the fused EP8 kernel's limit on experts per rank
FUSED_EP8_MAX_SLOTS = 63


def default_draft_slots(slots: int) -> int:
    """Draft slots that follow the backbone's: one more per 3 extra."""
    return DRAFT_EXPERTS // RANKS + math.ceil((slots - TARGET_EXPERTS // RANKS) / 3)


def load_counts(paths: list[str]) -> dict[str, np.ndarray]:
    """Sum the count arrays of the given files (directories: their
    `expert_counts_dp*.npz`)."""
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "expert_counts_dp*.npz")))
            if not found:
                raise FileNotFoundError(f"no expert_counts_dp*.npz in {p}")
            files += found
        else:
            files.append(p)
    out: dict[str, np.ndarray] = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            for pre in ("", "draft_"):
                if pre + "counts_decode" not in z:
                    continue
                ids = z[pre + "layer_ids"].astype(np.int64)
                if pre + "layer_ids" in out and not np.array_equal(
                    out[pre + "layer_ids"], ids
                ):
                    raise ValueError(f"{f}: other {pre}layer_ids than the first file")
                out[pre + "layer_ids"] = ids
                for k in ("counts_decode", "counts_prefill"):
                    v = z[pre + k].astype(np.int64)
                    out[pre + k] = out[pre + k] + v if pre + k in out else v
    if "counts_decode" not in out:
        raise ValueError("no backbone counts (counts_decode) in the input")
    return out


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    tot = v.sum(1, keepdims=True)
    # an empty objective counts as uniform
    return np.where(tot > 0, v / np.where(tot > 0, tot, 1), 1.0 / v.shape[1])


def objectives(decode: np.ndarray, prefill: np.ndarray, mode: str) -> np.ndarray:
    """One layer's normalised objectives `V [K, E]`."""
    if mode == "2d":
        return _normalise(np.stack([decode, prefill]))
    if mode == "all":
        return _normalise(decode + prefill)
    if mode == "decode":
        return _normalise(decode)
    raise ValueError(f"unknown mode {mode!r}")


def replicate(V: np.ndarray, n_slots: int) -> np.ndarray:
    """Replica count per expert for `n_slots` slots (step 1)."""
    n = V.shape[1]
    if not n <= n_slots <= RANKS * n:
        raise ValueError(f"{n_slots} slots for {n} experts on {RANKS} ranks")
    m = V.max(0)
    nrep = np.ones(n, dtype=np.int64)
    for _ in range(n_slots - n):
        nrep[int(np.argmax(np.where(nrep < RANKS, m / nrep, -1.0)))] += 1
    return nrep


def place(V: np.ndarray, nrep: np.ndarray, slots: int) -> np.ndarray:
    """`phys2log [RANKS * slots]` of the replicas (steps 2 and 3)."""
    k, n = V.shape
    m = V.max(0)
    order = sorted((-m[x] / nrep[x], x) for x in range(n) for _ in range(nrep[x]))
    load = np.zeros((RANKS, k))
    held: list[set[int]] = [set() for _ in range(RANKS)]
    for _, x in order:
        w = V[:, x] / nrep[x]
        free = [r for r in range(RANKS) if len(held[r]) < slots and x not in held[r]]
        r = min(free, key=lambda i: ((load[i] + w).max(), (load[i] + w).sum(), i))
        held[r].add(x)
        load[r] += w
    return np.array([x for h in held for x in sorted(h)], dtype=np.int32)


def plan_layers(
    decode: np.ndarray, prefill: np.ndarray, slots: int, mode: str
) -> np.ndarray:
    """`phys2log [L, RANKS * slots]` for the layers' counts `[L, E]`."""
    rows = []
    for d, p in zip(decode, prefill):
        V = objectives(d, p, mode)
        rows.append(place(V, replicate(V, RANKS * slots), slots))
    return np.stack(rows)


def rank_imbalance(load: np.ndarray, phys2log: np.ndarray) -> float:
    """Max over mean of the per-rank load when each expert's `load` splits
    evenly over its replicas."""
    load = np.asarray(load, dtype=np.float64)
    nrep = np.bincount(phys2log, minlength=len(load))
    per_rank = (load[phys2log] / nrep[phys2log]).reshape(RANKS, -1).sum(1)
    return float(per_rank.max() / per_rank.mean())


def step_imbalance(
    load: np.ndarray,
    phys2log: np.ndarray,
    layer: int,
    tokens: int,
    topk: int,
    steps: int,
    rng: np.random.Generator,
) -> float:
    """Mean per-step rank max/mean: `steps` batches of `tokens` tokens per
    rank drawn from the popularity `load`, each route sent to the replica the
    runtime's hash picks."""
    n = len(load)
    p = np.asarray(load, dtype=np.float64)
    perm = LayerPerm(phys2log, layer)
    ratios = []
    for _ in range(steps):
        ids = rng.choice(n, size=(tokens * RANKS, topk), p=p / p.sum())
        per_rank = np.bincount(
            perm.remap_reference(ids).ravel() // perm.slots_per_rank,
            minlength=RANKS,
        )
        ratios.append(per_rank.max() / per_rank.mean())
    return float(np.mean(ratios))


def report(
    counts: dict[str, np.ndarray],
    plan: dict[str, np.ndarray],
    steps: int,
    tokens: int,
    seed: int,
) -> list[str]:
    """Rank load max/mean of the identity placement and of `plan`, on the
    aggregate counts and (`steps` > 0) on simulated decode steps."""
    lines = []
    rng = np.random.default_rng(seed)
    for key, n, topk in (("", TARGET_EXPERTS, 6), ("draft_", DRAFT_EXPERTS, 3)):
        if key + "phys2log" not in plan:
            continue
        name = key.rstrip("_") or "backbone"
        p2l = plan[key + "phys2log"]
        ident = np.arange(n, dtype=np.int32)
        dec, pre = counts[key + "counts_decode"], counts[key + "counts_prefill"]
        for kind, c in (("all", dec + pre), ("decode", dec), ("prefill", pre)):
            a = [rank_imbalance(x, ident) for x in c if x.sum()]
            b = [rank_imbalance(x, r) for x, r in zip(c, p2l) if x.sum()]
            if a:
                lines.append(
                    f"{name} {kind} counts, rank load max/mean (mean / worst "
                    f"layer): identity {np.mean(a):.3f} / {max(a):.3f}, "
                    f"plan {np.mean(b):.3f} / {max(b):.3f}"
                )
        if steps:
            ids = counts[key + "layer_ids"]
            sims = [
                np.mean(
                    [
                        step_imbalance(x, r, int(i), tokens, topk, steps, rng)
                        for x, r, i in zip(dec, rows, ids)
                        if x.sum()
                    ]
                )
                for rows in (np.tile(ident, (len(dec), 1)), p2l)
            ]
            lines.append(
                f"{name} decode steps, {tokens} tokens per rank, per-step rank "
                f"load max/mean: identity {sims[0]:.3f}, plan {sims[1]:.3f}"
            )
    return lines


def make_plan(
    counts: dict[str, np.ndarray],
    slots: int,
    mode: str = "2d",
    draft_slots: int | None = None,
    draft_mode: str = "all",
) -> dict[str, np.ndarray]:
    """The plan arrays (format of `akr_eplb.Plan`) for summed counts."""
    if counts["counts_decode"].shape[1] != TARGET_EXPERTS:
        raise ValueError(f"backbone counts are not {TARGET_EXPERTS} experts wide")
    plan = {
        "phys2log": plan_layers(
            counts["counts_decode"], counts["counts_prefill"], slots, mode
        ),
        "layer_ids": counts["layer_ids"].astype(np.int64),
        "world_size": np.int64(RANKS),
    }
    if "draft_counts_decode" in counts and len(counts["draft_layer_ids"]):
        sd = default_draft_slots(slots) if draft_slots is None else draft_slots
        plan["draft_phys2log"] = plan_layers(
            counts["draft_counts_decode"],
            counts["draft_counts_prefill"],
            sd,
            draft_mode,
        )
        plan["draft_layer_ids"] = counts["draft_layer_ids"].astype(np.int64)
    return plan


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--counts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slots", type=int, default=49, help="experts per rank, S")
    ap.add_argument("--mode", choices=MODES, default="2d")
    ap.add_argument("--draft-slots", type=int, default=None)
    ap.add_argument("--draft-mode", choices=MODES, default="all")
    ap.add_argument(
        "--sim-steps",
        type=int,
        default=0,
        help="also simulate this many decode steps per layer",
    )
    ap.add_argument("--sim-tokens", type=int, default=324)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.slots > FUSED_EP8_MAX_SLOTS:
        print(f"note: the fused_ep8 route declines {args.slots} slots per rank")
    counts = load_counts(args.counts)
    plan = make_plan(counts, args.slots, args.mode, args.draft_slots, args.draft_mode)
    note = (
        f"akr_eplb_plan --slots {args.slots} --mode {args.mode} "
        f"--draft-mode {args.draft_mode}"
    )
    np.savez(args.out, note=np.str_(note), **plan)
    loaded = Plan(args.out)
    print(f"wrote {args.out} (sha256 {loaded.sha}), slots per rank {loaded.slots}")
    for line in report(counts, plan, args.sim_steps, args.sim_tokens, args.seed):
        print(line)


if __name__ == "__main__":
    main()
