# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager DASH reference implementation for Unlimited-OCR-2.

The optional prompt length restricts remote reads to the input prefix, while
generated tokens are visible only in the local output window.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class RouterNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        y = y.clamp(-1e4, 1e4) * (1 + self.weight.float())
        return y.clamp(-1e4, 1e4).to(x.dtype)


class DashRouter(nn.Module):
    def __init__(self, hidden_size: int, index_dim: int, eps: float):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, index_dim, bias=False)
        self.q_norm = RouterNorm(index_dim, eps)
        self.k_norm = RouterNorm(index_dim, eps)
        self.head_cls = nn.Parameter(torch.zeros(index_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.q_norm(self.q_proj(x)), self.k_norm(self.k_proj(x))


def page_coordinates(
    roles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token/page RoPE coordinates and image/query page IDs.

    Roles encode text=0, image=1 and <PAGE>=2. The entire prompt must be
    available: response markers restart numbering after the last prompt image.
    """
    image = roles == 1
    marker = roles == 2
    ar = torch.arange(roles.numel(), device=roles.device)
    last_image = int(torch.where(image, ar, -1).max().item())
    cum = marker.long().cumsum(0)
    if last_image < 0 or int(cum[last_image]) == 0:
        raise ValueError("Unlimited-OCR-2 requires <PAGE><image> for every page")
    num_pages = cum[last_image]
    prefill = ar <= last_image
    raw_page = torch.where(prefill, cum, cum - num_pages).clamp(min=0)
    # Match the reference decode state machine after the final page.
    page = raw_page.clamp(max=num_pages)
    prompt_counts = torch.zeros(
        int(num_pages) + 1, device=roles.device, dtype=torch.long
    )
    prompt_counts.scatter_add_(0, page, prefill.long())
    starts = torch.cat(
        (torch.ones_like(page[:1], dtype=torch.bool), page[1:] != page[:-1])
    )
    run_start = torch.where(starts, ar, 0).cummax(0).values
    tok = ar - run_start + torch.where(prefill, 0, prompt_counts[page])
    # A further response marker restarts the final page at its prompt offset.
    last_marker = torch.where(marker & ~prefill, ar, 0).cummax(0).values
    tok = torch.where(raw_page > num_pages, prompt_counts[page] + ar - last_marker, tok)
    image_page = torch.where(image, cum - 1, -1)
    query_page = torch.where(prefill, cum - 1, cum - num_pages - 1)
    return tok, page, image_page, query_page


def prompt_page_positions(input_tokens: list[int], sp: dict) -> torch.Tensor:
    """Full-prompt token/page positions, page count and negative page lengths.

    The last two channels carry DASH cache metadata through the existing
    multi-dimensional position interface. Negative lengths distinguish prompt
    positions from the runner's ordinary, nonnegative decode positions.
    """
    ids = torch.tensor(input_tokens)
    roles = (ids == sp["image_token_id"]).long()
    roles += 2 * (ids == sp["page_token_id"]).long()
    tok, page, _, _ = page_coordinates(roles)
    last_image = torch.where(roles == 1)[0][-1]
    prefill = torch.arange(len(ids)) <= last_image
    pages = int(page.max())
    counts = torch.zeros(pages + 1, dtype=torch.long)
    counts.scatter_add_(0, page, prefill.long())
    return torch.stack((tok, page, torch.full_like(tok, pages), -counts[page] - 1))


def page_rope(
    x: torch.Tensor, tok: torch.Tensor, page: torch.Tensor, sp: dict, *, main: bool
) -> torch.Tensor:
    """Apply non-interleaved RoPE with token/page frequency-pair sections."""
    dim = x.shape[-1]
    pairs = int(sp.get("sparse_rope_page_pairs", 1))
    ntok = dim // 2 - pairs
    base = float(sp.get("sparse_rope_tok_theta", 10000.0))
    page_base = float(sp.get("sparse_rope_page_theta", 256.0))
    mode_key = "sparse_main_rope_tok_freq_mode" if main else "sparse_rope_tok_freq_mode"
    exponent = torch.arange(ntok, device=x.device).float()
    exponent = exponent / (dim / 2 if sp.get(mode_key) == "truncate" else ntok)
    inv_tok = 1.0 / (base**exponent)
    inv_page = 1.0 / (
        page_base ** (torch.arange(pairs, device=x.device).float() / pairs)
    )
    mod = int(sp.get("sparse_rope_page_mod", 0))
    if mod > 0:
        page = page % mod
    angles = torch.cat(
        (tok.float()[:, None] * inv_tok, page.float()[:, None] * inv_page), -1
    )
    phase = torch.cat((angles, angles), -1)
    cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def entmax15(x: torch.Tensor) -> torch.Tensor:
    x = (x - x.amax(-1, keepdim=True)) / 2
    ordered = x.sort(-1, descending=True).values
    rho = torch.arange(1, x.shape[-1] + 1, device=x.device).float()
    mean = ordered.cumsum(-1) / rho
    mean_sq = ordered.square().cumsum(-1) / rho
    delta = ((1 - rho * (mean_sq - mean.square())) / rho).clamp(min=0)
    tau = mean - delta.sqrt()
    support = (tau <= ordered).sum(-1, keepdim=True).clamp(min=1)
    threshold = tau.gather(-1, support - 1)
    return (x - threshold).clamp(min=0).square()


def route_blocks(
    scores: torch.Tensor, candidate: torch.Tensor, sp: dict, layer_idx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    remote = scores > -5e8
    candidate = remote & candidate
    candidate = torch.where(candidate.any(-1, keepdim=True), candidate, remote)
    freq = sp["sparse_attention_freq"]
    slot = sum(bool(f) for f in freq[:layer_idx])

    def layer_value(key, fallback):
        values = sp.get(key)
        return float(values[slot]) if values else float(fallback)

    scale = layer_value("dash_scale_per_layer", sp.get("dash_scaling_factor", 128))
    floor = layer_value("dash_floor_per_layer", sp.get("dash_prior_floor", 1e-4))
    x = torch.where(
        candidate, scores * (scale / math.sqrt(sp["sparse_index_dim"])), -1e4
    )
    x = torch.nan_to_num(x, nan=-1e4, posinf=-1e4, neginf=-1e4)
    # The published inference implementation uses top-M=16, then hard top-K.
    m = min(16, x.shape[-1])
    values, indices = x.topk(m, -1)
    prior = entmax15(values)
    k = min(int(sp.get("dash_topk_blocks", 12)), m)
    indices, prior = indices[:, :k], prior[:, :k]
    valid = candidate.gather(-1, indices)
    logp = prior.clamp(min=floor).log()
    mean = torch.where(valid, logp, 0).sum(-1, keepdim=True)
    mean = mean / valid.sum(-1, keepdim=True).clamp(min=1)
    bias = torch.where(valid, (logp - mean) / float(sp.get("dash_sigma", 1)), 0)
    return indices, bias, valid


def dash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    iq: torch.Tensor,
    ik: torch.Tensor,
    roles: torch.Tensor,
    head_cls: torch.Tensor,
    sp: dict,
    layer_idx: int,
    prompt_length: int | None = None,
    backend: str = "TRITON_ATTN",
) -> torch.Tensor:
    """Attend query suffix [Q,H,D] against cached keys [S,Hkv,D]."""
    length, dim = ik.shape
    nq, nh, hd = q.shape
    bs = int(sp["sparse_block_size"])
    local = int(sp["sparse_local_block"]) * bs
    nblk = (length + bs - 1) // bs
    tok, page, image_page, query_page = page_coordinates(roles)
    q = page_rope(q, tok[-nq:], page[-nq:], sp, main=True)
    k = page_rope(k, tok, page, sp, main=True)
    iq = page_rope(iq, tok[-nq:], page[-nq:], sp, main=False).float()
    ik = page_rope(ik, tok, page, sp, main=False).float()
    iq = torch.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0)
    ik = torch.nan_to_num(ik, nan=0.0, posinf=0.0, neginf=0.0)
    padded = nblk * bs
    kk = F.pad(ik, (0, 0, 0, padded - length)).view(nblk, bs, dim)
    real = torch.arange(padded, device=q.device).view(nblk, bs) < length
    if prompt_length is not None:
        real &= torch.arange(padded, device=q.device).view(nblk, bs) < prompt_length
    inv_sqrt = 1.0 / math.sqrt(dim)
    logits = ((kk * head_cls.float()).sum(-1) * inv_sqrt).clamp(-1e4, 1e4)
    row_max = logits.masked_fill(~real, torch.finfo(logits.dtype).min).amax(
        -1, keepdim=True
    )
    exp = (logits - row_max).exp() * real
    weights = exp / exp.sum(-1, keepdim=True).clamp(min=1e-6)
    summary = torch.nan_to_num(
        (weights[:, :, None] * kk).sum(1), nan=0.0, posinf=0.0, neginf=0.0
    )
    image_blocks = F.pad(image_page, (0, padded - length), value=-1).view(nblk, bs)
    num_pages = int(image_page.max()) + 1
    membership = torch.zeros((nblk, num_pages + 1), dtype=torch.bool, device=q.device)
    membership.scatter_(1, image_blocks + 1, True)
    output = torch.empty_like(q)
    block_ids = torch.arange(nblk, device=q.device)
    offsets = torch.arange(bs, device=q.device)
    local_offsets = torch.arange(local, device=q.device)
    # Bound the gathered [queries, selected tokens, heads, dim] temporary.
    chunk_limit = 16 if backend == "TRITON_ATTN" else 256
    chunk = min(int(sp.get("dash_query_chunk", 2048)), chunk_limit)
    kv_group = nh // k.shape[1]
    for start in range(0, nq, chunk):
        end = min(start + chunk, nq)
        pos = torch.arange(length - nq + start, length - nq + end, device=q.device)
        scores = (iq[start:end] @ summary.T * inv_sqrt).clamp(-1e4, 1e4)
        remote = block_ids[None] <= pos[:, None] // bs - int(sp["sparse_local_block"])
        if prompt_length is not None:
            remote &= block_ids[None] * bs < prompt_length
        scores = scores.masked_fill(~remote, -1e9)
        qp = query_page[-nq:][start:end]
        candidate = membership.T[(qp + 1).clamp(0, num_pages)]
        candidate &= (qp[:, None] >= 0) & (qp[:, None] < num_pages)
        selected, bias, valid = route_blocks(scores, candidate, sp, layer_idx)
        remote_pos = (selected[:, :, None] * bs + offsets).flatten(1)
        rbias = bias.repeat_interleave(bs, -1)
        rvalid = valid.repeat_interleave(bs, -1)
        # Local tokens win on overlap with a selected block (no double count).
        rvalid &= (remote_pos <= pos[:, None] - local) & (remote_pos < length)
        if prompt_length is not None:
            rvalid &= remote_pos < prompt_length
        lpos = pos[:, None] - local + 1 + local_offsets
        indices = torch.cat((remote_pos, lpos), -1)
        keep = torch.cat((rvalid, lpos >= 0), -1)
        additive = torch.cat((rbias, torch.zeros_like(lpos, dtype=torch.float32)), -1)
        indices = indices.clamp(0, length - 1)
        if backend != "TRITON_ATTN":
            from vllm.v1.attention.ops.unlimited_ocr2 import flash_dash_attention

            # Both Flash APIs consume segments with a valid prefix. Shift the
            # early local window so padding is at the end rather than the front.
            local_pos = (pos - local + 1).clamp(min=0)[:, None] + local_offsets
            gather = torch.cat((remote_pos, local_pos), -1).clamp(0, length - 1)
            used = (
                torch.cat(
                    (
                        rvalid.view(end - start, -1, bs).sum(-1),
                        (pos + 1).clamp(max=local)[:, None],
                    ),
                    -1,
                )
                .to(torch.int32)
                .flatten()
            )
            output[start:end] = flash_dash_attention(
                q[start:end],
                k[gather].flatten(0, 1).contiguous(),
                v[gather].flatten(0, 1).contiguous(),
                used,
                bias,
                backend,
                bs,
                local,
            )
            continue
        keys = k[indices].repeat_interleave(kv_group, 2).float()
        values = v[indices].repeat_interleave(kv_group, 2).float()
        logits = torch.einsum("qhd,qshd->qhs", q[start:end].float(), keys)
        logits = logits * (1.0 / math.sqrt(hd)) + additive[:, None]
        probs = logits.masked_fill(~keep[:, None], -torch.inf).softmax(-1)
        output[start:end] = torch.einsum("qhs,qshd->qhd", probs, values).to(q.dtype)
    return output
