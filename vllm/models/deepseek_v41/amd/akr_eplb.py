# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static routed-expert placement with replicas for the MegaMoE experts of
DeepSeek-V4.1-Flash on EP8 (EPLB without the runtime rebalancer).

`kernel_config.amd_kernels.expert_perm` names a plan `.npz` (a bare file name
that `configs/` next to this module holds is that shipped plan, anything else
a path); unset, nothing here is imported or run. Per MoE layer the plan holds
`phys2log[8 * S]` (one S per expert family: a 2-D array), the logical
expert of each physical slot; slot `p` lives on EP rank `p // S`, the rank-major
physical id that MegaMoE (`dest = id // experts_per_rank`), the fused EP8
kernel and vLLM's EPLB use. A logical expert may hold several slots
(replicas). Plan keys:

    phys2log [L, 8 * S] int, layer_ids [L]: the backbone layers (384 experts)
    draft_phys2log [D, 8 * S_d], draft_layer_ids [D]: the DSpark draft layers
        (128 experts), optional (absent: identity placement)
    world_size: the EP size the plan was made for (default 8)

Weights: `DeepseekV4AiterMegaMoEExperts.weight_loader` copies a checkpoint
expert into every local slot that holds it (each rank allocates S experts per
layer). Routing: the routers still choose logical ids; right before the
experts run, `LayerPerm.remap` turns each id `e >= 0` into the physical id
`l2p[e, hash(token * K + k, layer, rank) % count[e]]` (one Triton launch,
graph safe, deterministic, uniform over the replicas); ids < 0 stay.
"""

import dataclasses
import functools
import hashlib
import os

import numpy as np
import regex as re
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
TARGET_EXPERTS = 384
DRAFT_EXPERTS = 128
_BLOCK = 1024
_M32 = 0xFFFFFFFF

# AITER's tuned MegaMoE profile for this model on EP8 is keyed on 48 experts
# per rank; with one replica slot per rank (S = 49) the same tuning applies.
_AITER_PROFILE = "v41_flash_ep8"
_AITER_PROFILE_EPR = (49,)


def plan_path(vllm_config) -> str:
    """The plan file of `kernel_config.amd_kernels.expert_perm`, '' if unset."""
    try:
        path = str(vllm_config.kernel_config.amd_kernels.expert_perm or "")
    except AttributeError:
        return ""
    if path and os.path.basename(path) == path:
        # a bare file name is a shipped plan; a path names the user's file
        shipped = os.path.join(CONFIG_DIR, path)
        if os.path.exists(shipped):
            return shipped
    return path


def layer_index(prefix: str) -> int:
    """The decoder layer id of a module prefix such as
    `model.layers.7.ffn.experts`."""
    found = re.findall(r"(?:^|\.)layers\.(\d+)(?:\.|$)", prefix)
    if len(found) != 1:
        raise ValueError(f"cannot find the layer index in {prefix!r}")
    return int(found[0])


def check_row(row: np.ndarray, n_experts: int, world: int, what: str) -> np.ndarray:
    """One layer's phys2log: integers, `world * S` slots with
    `S >= n_experts / world`, every logical id present."""
    row = np.asarray(row)
    if row.ndim != 1 or not np.issubdtype(row.dtype, np.integer):
        raise ValueError(
            f"expert plan {what}: expected a 1-D integer phys2log, "
            f"got {row.dtype}{row.shape}"
        )
    if len(row) % world or len(row) < n_experts or n_experts % world:
        raise ValueError(
            f"expert plan {what}: {len(row)} slots is not world {world} x S "
            f"with S >= {n_experts}/{world}"
        )
    if row.min() < 0 or row.max() >= n_experts:
        raise ValueError(f"expert plan {what}: ids outside 0..{n_experts - 1}")
    if np.bincount(row, minlength=n_experts).min() < 1:
        raise ValueError(f"expert plan {what}: a logical expert has no slot")
    return row.astype(np.int64)


class Plan:
    """A validated plan file: `layers[n_experts][layer_id] = phys2log`."""

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "rb") as f:
            self.sha = hashlib.sha256(f.read()).hexdigest()
        z = np.load(path, allow_pickle=False)
        self.world = int(z["world_size"]) if "world_size" in z else 8
        self.layers: dict[int, dict[int, np.ndarray]] = {}
        for pre, n in (("", TARGET_EXPERTS), ("draft_", DRAFT_EXPERTS)):
            if pre + "phys2log" not in z:
                if not pre:
                    raise ValueError(f"expert plan {path}: no 'phys2log' array")
                continue
            p2l, ids = z[pre + "phys2log"], z[pre + "layer_ids"]
            if (
                p2l.ndim != 2
                or ids.shape != (p2l.shape[0],)
                or len(set(ids.tolist())) != len(ids)
            ):
                raise ValueError(
                    f"expert plan {path}: bad {pre}phys2log {p2l.shape} / "
                    f"layer_ids {ids.shape}"
                )
            self.layers[n] = {
                int(i): check_row(r, n, self.world, f"{pre}layer {int(i)}")
                for i, r in zip(ids, p2l)
            }
        self.slots = {
            n: next(iter(rows.values())).shape[0] // self.world
            for n, rows in self.layers.items()
            if rows
        }
        logger.info(
            "amd_kernels expert placement %s sha256=%s: world %d, layers %s, "
            "slots per rank %s",
            path,
            self.sha[:16],
            self.world,
            {n: len(rows) for n, rows in self.layers.items()},
            self.slots,
        )


@functools.cache
def load_plan(path: str) -> Plan:
    return Plan(path)


def hash32(idx, salt: int):
    """Host reference of the kernel's replica hash (idx = token * K + k)."""
    h = (np.asarray(idx, dtype=np.int64) * 2654435761 + salt) & _M32
    h = h ^ (h >> 15)
    h = (h * 2246822519) & _M32
    return h ^ (h >> 13)


class LayerPerm:
    """One layer's placement on `world` EP ranks: host tables and their
    device copies."""

    def __init__(
        self, phys2log: np.ndarray, layer: int = 0, rank: int = 0, world: int = 8
    ) -> None:
        p2l = np.asarray(phys2log, dtype=np.int64)
        self.phys2log = p2l
        self.n_slots = len(p2l)
        self.world, self.layer, self.rank = world, layer, rank
        self.slots_per_rank = self.n_slots // world
        self.n = int(p2l.max()) + 1
        self.count = np.bincount(p2l, minlength=self.n)
        self.max_rep = int(self.count.max())
        # padded logical -> physical table [E, R]; entries past count[e] repeat
        # the first replica and are never read
        order = np.argsort(p2l, kind="stable")
        starts = np.concatenate([[0], np.cumsum(self.count)[:-1]])
        self.l2p = np.empty((self.n, self.max_rep), dtype=np.int64)
        for r in range(self.max_rep):
            self.l2p[:, r] = order[starts + np.minimum(r, self.count - 1)]
        self.salt = (layer * 0x9E3779B1 + rank * 0x85EBCA77 + 0x27D4EB2F) & _M32
        self._tables: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def is_identity(self) -> bool:
        return self.max_rep == 1 and np.array_equal(self.phys2log, np.arange(self.n))

    def local_slots(self, logical: int, start: int, n_local: int) -> list[int]:
        """The local slots (of the rank holding slots `[start, start +
        n_local)`) that hold `logical`."""
        slots = self.l2p[logical, : self.count[logical]]
        return [int(p) - start for p in slots if start <= p < start + n_local]

    def tables(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """(l2p int32 [E * R], count int32 [E]) on `device`; allocate before
        graph capture."""
        key = torch.device(device)
        t = self._tables.get(key)
        if t is None:
            t = (
                torch.from_numpy(self.l2p.astype(np.int32).reshape(-1)).to(key),
                torch.from_numpy(self.count.astype(np.int32)).to(key),
            )
            self._tables[key] = t
        return t

    def remap(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Logical -> physical ids (a new tensor of the same shape and dtype);
        ids < 0 (padding) stay."""
        l2p, count = self.tables(topk_ids.device)
        ids = topk_ids.contiguous()
        out = torch.empty_like(ids)
        n = ids.numel()
        if n:
            _remap_kernel[(triton.cdiv(n, _BLOCK),)](
                ids,
                l2p,
                count,
                out,
                n,
                self.salt,
                count.numel(),
                R=self.max_rep,
                BLOCK=_BLOCK,
            )
        return out

    def remap_reference(self, topk_ids: np.ndarray) -> np.ndarray:
        """Host reference of `remap`."""
        ids = np.asarray(topk_ids)
        flat = ids.reshape(-1).astype(np.int64)
        ok = (flat >= 0) & (flat < self.n)
        e = np.where(ok, flat, 0)
        r = 0
        if self.max_rep > 1:
            r = (hash32(np.arange(flat.size), self.salt) >> 8) % self.count[e]
        out = np.where(ok, self.l2p[e, r], flat)
        return out.reshape(ids.shape).astype(ids.dtype)


@triton.jit(do_not_specialize=["n", "salt", "E"])
def _remap_kernel(
    ids_ptr,
    tbl_ptr,
    cnt_ptr,
    out_ptr,
    n,
    salt,
    E,
    R: tl.constexpr,
    BLOCK: tl.constexpr,
):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    e = tl.load(ids_ptr + o, mask=m, other=0)
    ok = m & (e >= 0) & (e < E)
    ee = tl.where(ok, e, 0).to(tl.int64)
    if R == 1:
        slot = ee
    else:
        c = tl.load(cnt_ptr + ee, mask=ok, other=1).to(tl.int64)
        h = (o.to(tl.int64) * 2654435761 + salt) & 0xFFFFFFFF
        h = h ^ (h >> 15)
        h = (h * 2246822519) & 0xFFFFFFFF
        h = h ^ (h >> 13)
        slot = ee * R + ((h >> 8) % c)
    p = tl.load(tbl_ptr + slot, mask=ok, other=0).to(e.dtype)
    tl.store(out_ptr + o, tl.where(ok, p, e), mask=m)


def layer_perm(
    vllm_config, prefix: str, n_experts: int, ep_rank: int, ep_size: int
) -> LayerPerm | None:
    """The placement of the MoE layer named by `prefix`, or None (no plan, an
    identity row, a draft layer the plan leaves out). A plan that does not fit
    the model (EP size, expert count, a missing backbone layer) raises."""
    path = plan_path(vllm_config)
    if not path:
        return None
    plan = load_plan(path)
    if plan.world != ep_size:
        raise ValueError(
            f"expert plan {path}: made for EP size {plan.world}, "
            f"running EP size {ep_size}"
        )
    layer = layer_index(prefix)
    rows = plan.layers.get(n_experts)
    draft = n_experts == DRAFT_EXPERTS
    if rows is None:
        if draft:
            return None
        raise ValueError(
            f"expert plan {path}: no layers with {n_experts} experts "
            f"(has {sorted(plan.layers)})"
        )
    row = rows.get(layer)
    if row is None:
        if not draft:
            raise ValueError(
                f"expert plan {path}: layer {layer} ({prefix}) missing "
                f"(plan layers {sorted(rows)})"
            )
        logger.warning_once("expert plan: draft layer %d not in the plan", layer)
        return None
    perm = LayerPerm(row, layer, ep_rank, ep_size)
    return None if perm.is_identity else perm


def register_aiter_profile(experts_per_rank: int, hidden: int, inter: int) -> bool:
    """Give MegaMoE's replica-slot geometry (S = 49) AITER's tuned EP8 profile
    of this model, which AITER keys on 48 experts per rank only. Must run
    before the MegaMoE runtimes are built; a no-op for other S, other shapes,
    or an AITER that already matches S (ROCm/aiter#8453 adds S = 49 to the
    profile; drop this once the pinned AITER has it)."""
    if experts_per_rank not in _AITER_PROFILE_EPR:
        return False
    from aiter.ops.flydsl.kernels.mega_moe import mega_moe_config as mc

    if not hasattr(mc, "_match_network") or not hasattr(mc, "_NETWORKS"):
        logger.warning(
            "MegaMoE: this AITER has no profile table to extend; %d experts per "
            "rank run AITER's untuned MegaMoE configs",
            experts_per_rank,
        )
        return False
    if mc._match_network(experts_per_rank, hidden, inter) is not None:
        return False
    base = next((p for p in mc._NETWORKS if p.name == _AITER_PROFILE), None)
    if base is None or (base.model_dim, base.inter_dim) != (hidden, inter):
        logger.warning(
            "MegaMoE: AITER has no %s profile for hidden %d / intermediate %d; "
            "%d experts per rank run AITER's untuned MegaMoE configs",
            _AITER_PROFILE,
            hidden,
            inter,
            experts_per_rank,
        )
        return False
    mc._NETWORKS = mc._NETWORKS + (
        dataclasses.replace(
            base,
            name=f"{_AITER_PROFILE}_e{experts_per_rank}",
            experts_per_rank=experts_per_rank,
            family_experts_per_rank=(),
        ),
    )
    mc.build_mega_moe_bundle_plan.cache_clear()
    logger.info(
        "MegaMoE: AITER profile %s registered for %d experts per rank",
        _AITER_PROFILE,
        experts_per_rank,
    )
    return True
