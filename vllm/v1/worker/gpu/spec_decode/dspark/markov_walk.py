# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch DSpark Markov walk with an int8 Markov projection kept on chip.

Host side of ``csrc/dspark_markov_walk.cu`` (layouts adapted from Xinwei
Qiang's int8 DSpark Markov walk kernels, MIT License). The eager
walk reads the whole ``V x 256`` projection W2 from HBM once per draft step.
The kernel quantizes W2 to int8 with one fp32 scale per row, keeps most of it
in shared memory across all SMs and streams the rest from L2, and runs all
steps in one cooperative launch. The target model verifies every draft token,
so the only effect of the quantization is on acceptance.
"""

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

MARKOV_RANK = 256
MAX_STEPS = 16
MAX_REQS = 64
TILE_ROWS = 64
W1_ROW_BYTES = 528
STATE_NUMEL = 16 + 2 * MAX_STEPS * MAX_REQS * 16
# Tiles per CTA built into the kernel; each keeps RES tiles in SMEM and
# streams the others from L2 through a RING-slot ring.
TILE_BUILDS = (16, 18, 20, 24, 28, 32)
RES, RING = 11, 2
# The timed stream mask of the 18-tile layout (V = 151936 on 132 SMs).
MASK_18 = sum(1 << t for t in (1, 2, 4, 7, 9, 11, 14))

# Tile row permutation: accumulator column n of a 64-row tile holds tile row
# rho(n), so every thread owns two aligned 8-row runs.
_RHO = [
    (8 * ((n % 8) // 2) + 2 * (n // 8) + n % 2)
    if n // 8 < 4
    else (32 + 8 * ((n % 8) // 2) + 2 * (n // 8 - 4) + n % 2)
    for n in range(64)
]


def unsupported_reason(
    w1: torch.Tensor, w2: torch.Tensor, num_steps: int
) -> str | None:
    """Why the kernel cannot walk this Markov head, or None if it can."""
    if not current_platform.is_cuda() or not current_platform.is_device_capability(90):
        return "it needs an sm_90 GPU"
    if not hasattr(torch.ops._C, "dspark_markov_walk"):
        return "this build has no dspark_markov_walk kernel"
    if w2.dim() != 2 or w2.shape[1] != MARKOV_RANK:
        return f"it needs a rank-{MARKOV_RANK} Markov head"
    if w1.shape != w2.shape:
        return "it needs a full-vocabulary draft (no draft-to-target map)"
    if not w2.dtype.is_floating_point or not w1.dtype.is_floating_point:
        return "it needs an unquantized Markov projection"
    if w2.shape[0] % 8 != 0:
        return "it needs a vocabulary size divisible by 8"
    if not 1 <= num_steps <= MAX_STEPS:
        return f"it supports at most {MAX_STEPS} draft tokens"
    if pick_tiles(w2.shape[0], _num_sms(w2.device)) is None:
        return "the vocabulary is too large for this GPU"
    return None


def _num_sms(device: torch.device) -> int:
    return current_platform.num_compute_units(device.index or 0)


def pick_tiles(vocab_size: int, num_sms: int) -> int | None:
    need = -(-vocab_size // (num_sms * TILE_ROWS))
    return next((t for t in TILE_BUILDS if t >= need), None)


def stream_mask(tiles: int) -> int:
    """Which tiles stream from L2 (bit t), spread over the four warpgroups.

    With one 32-request block, warpgroup p walks tiles p, p + 4, ...; residents
    are shared in proportion to each warpgroup's tile count, the first RING
    warpgroups of (1, 2, 3, 0) open on a streamed tile (they consume the ring
    prefill) and the others on a resident one.
    """
    if tiles == 18:
        return MASK_18
    seqs = [list(range(p, tiles, 4)) for p in range(4)]
    quota = [RES * len(sq) / tiles for sq in seqs]
    res = [int(q) for q in quota]
    for _ in range(RES - sum(res)):
        parity = [res[0] + res[2], res[1] + res[3]]
        p = max(
            (p for p in range(4) if res[p] < len(seqs[p])),
            key=lambda p: (round(quota[p] - res[p], 9), -parity[p % 2], -p),
        )
        res[p] += 1
    starts_streamed = {(1, 2, 3, 0)[i] for i in range(min(RING, 4))}
    mask = 0
    for p, sq in enumerate(seqs):
        n, nr = len(sq), res[p]
        ns = n - nr
        if p in starts_streamed:
            pos = {i * n // ns for i in range(ns)} if ns else set()
        else:
            pos = set(range(n)) - ({i * n // nr for i in range(nr)} if nr else set())
        for j in pos:
            mask |= 1 << sq[j]
    assert bin(mask).count("1") == tiles - RES and mask < (1 << tiles)
    return mask


def quantize_w2(w2: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 W2 with one fp32 scale per row; ``scale`` (the logit scale) folds in."""
    w = w2.float()
    row_scale = (w.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-30)
    q = (w / row_scale).round().clamp(-127, 127).to(torch.int8)
    return q, (row_scale.view(-1) * scale).contiguous()


def quantize_w1(w1: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """U ~= s_hi * q_hi + s_lo * q_lo with the fixed ratio s_lo = s_hi / 256."""
    u = w1.float()
    s_hi = (u.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-30)
    q_hi = (u / s_hi).round().clamp(-127, 127)
    s_lo = s_hi / 256.0
    q_lo = ((u - s_hi * q_hi) / s_lo).round().clamp(-127, 127)
    return q_hi.to(torch.int8), q_lo.to(torch.int8), s_hi.view(-1), s_lo.view(-1)


def build_w2_tiles(
    q_w2: torch.Tensor, rows_pad: int, tiles: int, mask: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per CTA, 64-row tiles as K-major wgmma B operands with permuted rows,
    split into the resident and the streamed tiles (each [grid, n, 16384])."""
    qp = torch.zeros(rows_pad, MARKOV_RANK, dtype=torch.int8, device=q_w2.device)
    qp[: q_w2.shape[0]] = q_w2
    num_tiles = rows_pad // TILE_ROWS
    t = qp.view(num_tiles, TILE_ROWS, MARKOV_RANK)[:, _RHO]
    # [tile][k32 chunk][8-column group][k16 core][column][16 B]
    v = t.view(num_tiles, 8, 8, 8, 2, 16).permute(0, 3, 1, 4, 2, 5)
    v = v.contiguous().view(num_tiles // tiles, tiles, TILE_ROWS * MARKOV_RANK)
    streamed = torch.tensor(
        [(mask >> i) & 1 for i in range(tiles)], dtype=torch.bool, device=qp.device
    )
    return v[:, ~streamed].contiguous(), v[:, streamed].contiguous()


def build_w1_rows(q_hi, q_lo, s_hi, s_lo) -> torch.Tensor:
    """W1 rows in A-fragment order: per k32 chunk j and lane tig,
    q_hi[32j + 4tig ..] | q_lo[..] | q_hi[32j + 16 + 4tig ..] | q_lo[..],
    then s_hi and s_lo as fp32 (528 B per row)."""
    vocab = q_hi.shape[0]
    hi = q_hi.view(vocab, 8, 2, 4, 4)
    lo = q_lo.view(vocab, 8, 2, 4, 4)
    frag = torch.stack([hi[:, :, 0], lo[:, :, 0], hi[:, :, 1], lo[:, :, 1]], dim=3)
    rows = torch.zeros(vocab, W1_ROW_BYTES, dtype=torch.uint8, device=q_hi.device)
    rows[:, :512] = frag.reshape(vocab, 512).view(torch.uint8)
    rows[:, 512:520] = torch.stack([s_hi, s_lo], 1).contiguous().view(torch.uint8)
    return rows


class DSparkMarkovWalker:
    """int8 layouts of one Markov head and the kernel's persistent state."""

    def __init__(
        self,
        markov_w1: torch.Tensor,
        markov_w2: torch.Tensor,
        num_steps: int,
        logit_scale: float = 1.0,
    ) -> None:
        reason = unsupported_reason(markov_w1, markov_w2, num_steps)
        if reason is not None:
            raise ValueError(f"dspark_markov_quantization='int8': {reason}.")
        device = markov_w2.device
        self.vocab_size = markov_w2.shape[0]
        self.num_steps = num_steps
        num_sms = _num_sms(device)
        tiles = pick_tiles(self.vocab_size, num_sms)
        assert tiles is not None
        self.tiles = tiles
        self.mask = stream_mask(tiles)
        rows_cta = tiles * TILE_ROWS
        rows_pad = -(-self.vocab_size // rows_cta) * rows_cta
        with torch.no_grad():
            q_w2, row_scale = quantize_w2(markov_w2, logit_scale)
            self.w2_res, self.w2_str = build_w2_tiles(q_w2, rows_pad, tiles, self.mask)
            del q_w2
            self.row_scale = torch.full(
                (rows_pad,), 1e-30, dtype=torch.float32, device=device
            )
            self.row_scale[: self.vocab_size] = row_scale
            self.w1_rows = build_w1_rows(*quantize_w1(markov_w1))
        # Round counter and exchange slots; zeroed once, never reset.
        self.state = torch.zeros(STATE_NUMEL, dtype=torch.int64, device=device)
        assert self.state.data_ptr() % 128 == 0
        logger.info_once(
            "DSpark Markov walk: int8 W2 on chip (%d tiles per CTA, %d CTAs, "
            "%.1f MiB of layouts).",
            tiles,
            rows_pad // rows_cta,
            self.memory_bytes() / 2**20,
        )

    def memory_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.w2_res, self.w2_str, self.row_scale, self.w1_rows)
        )

    def supports(self, num_reqs: int, base_logits: torch.Tensor) -> bool:
        return (
            1 <= num_reqs <= MAX_REQS
            and base_logits.dtype == torch.bfloat16
            and base_logits.shape[-1] == self.vocab_size
            and base_logits.is_contiguous()
        )

    def walk(
        self,
        base_logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        anchor_stride: int,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor | None,
        seeds: torch.Tensor,
        pos: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> None:
        """Walk ``base_logits`` [num_reqs, num_steps, V] into ``draft_tokens``.

        No allocation and no host sync: safe under CUDA graph capture.
        """
        ops.dspark_markov_walk(
            self.w2_res,
            self.w2_str,
            self.row_scale,
            self.w1_rows,
            base_logits,
            anchor_ids,
            anchor_stride,
            idx_mapping,
            temperature,
            seeds,
            pos,
            draft_tokens,
            draft_logits,
            self.state,
            self.num_steps,
            self.tiles,
            self.mask,
        )

    def warmup(self) -> None:
        """Load the kernel before CUDA graph capture.

        The first launch of a cooperative kernel loads its module; doing that
        while another cooperative grid is spinning can deadlock.
        """
        device = self.state.device
        n = self.num_steps
        base = torch.zeros(1, n, self.vocab_size, dtype=torch.bfloat16, device=device)
        anchor = torch.zeros(1, dtype=torch.int32, device=device)
        idx = torch.full((n,), -1, dtype=torch.int64, device=device)
        seeds = torch.zeros(1, dtype=torch.int64, device=device)
        pos = torch.zeros(n, dtype=torch.int64, device=device)
        tokens = torch.zeros(1, n, dtype=torch.int64, device=device)
        self.walk(base, anchor, 1, idx, None, seeds, pos, tokens, None)
        torch.accelerator.synchronize(device)
