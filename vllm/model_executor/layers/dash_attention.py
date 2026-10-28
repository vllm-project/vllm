# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DASH attention with page-aware RoPE and paged KV execution.

The optional prompt length restricts remote reads to the input prefix, while
generated tokens are visible only in the local output window.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionMetadata,
)


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
            from vllm.v1.attention.ops.dash_attention import flash_dash_attention

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


TOKEN_ROLES_KEY = "unlimited_ocr2_token_roles"
PAGE_POSITIONS_KEY = "unlimited_ocr2_page_positions"


class DASHAttention(nn.Module):
    def __init__(
        self,
        vllm_config,
        config,
        hidden_size,
        num_heads,
        cache_config=None,
        quant_config=None,
        reduce_results=True,
        prefix="",
        **kwargs,
    ):
        # Keep main KV and one shared router/metadata head per TP rank.
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp
        self.num_kv_heads = max(1, config.num_key_value_heads // tp)
        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dash_backend = config.unlimited_ocr2_attention_backend
        self.sp = config.sparse_attention_config
        self.layer_idx = int(prefix.split(".")[-2])
        index_dim = int(self.sp["sparse_index_dim"])
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
        )
        self.dash_router = DashRouter(hidden_size, index_dim, config.rms_norm_eps)
        window = vllm_config.model_config.rswa_window
        attention_cls = RSWAAttention if window is not None else Attention
        attention_kwargs = {"rswa_window": window} if window is not None else {}
        self.attn = attention_cls(
            self.num_kv_heads + 1,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads + 1,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
            attn_backend=TritonAttentionBackend,
            **attention_kwargs,
        )

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], -1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view_as(k)
        iq, ik = self.dash_router(hidden_states)
        kwargs = get_forward_context().additional_kwargs
        roles = kwargs.get(TOKEN_ROLES_KEY)
        positions = kwargs[PAGE_POSITIONS_KEY]
        result = torch.empty_like(q)
        self._attention(q, k, v, iq, ik, roles, positions, result)
        return self.o_proj(result.flatten(1))[0]

    @eager_break_during_capture
    def _attention(self, q, k, v, iq, ik, roles, positions, result):
        from vllm.v1.attention.ops.dash_attention import (
            paged_dash_decode,
            paged_dash_prefill,
        )

        context = get_forward_context()
        all_metadata = context.attn_metadata
        if all_metadata is None:
            result.zero_()
            return
        assert isinstance(all_metadata, dict)
        metadata = all_metadata[self.attn.layer_name]
        assert isinstance(metadata, TritonAttentionMetadata)
        assert roles is not None
        assert isinstance(context.slot_mapping, dict)
        slots = context.slot_mapping[self.attn.layer_name]
        cache = self.attn.kv_cache
        if metadata.max_query_len == 1:
            result.copy_(
                paged_dash_decode(
                    q,
                    k,
                    v,
                    iq,
                    ik,
                    roles,
                    cache,
                    metadata.block_table,
                    metadata.seq_lens,
                    slots,
                    self.dash_router.head_cls,
                    self.sp,
                    self.layer_idx,
                    self.max_model_len,
                    metadata.rswa_prefix_lens,
                    backend=self.dash_backend,
                    cached_summaries=True,
                    positions=positions,
                )
            )
        else:
            starts = metadata.query_start_loc.tolist()
            first_slots = slots[
                metadata.query_start_loc[:-1].clamp_max(slots.shape[0] - 1)
            ]
            lengths, first_slots_cpu = torch.stack(
                (metadata.seq_lens, first_slots)
            ).tolist()
            result.zero_()
            for request, length in enumerate(lengths):
                start, end = starts[request : request + 2]
                # Padding and capture rows have no assigned KV slots.
                if start == end or first_slots_cpu[request] < 0:
                    continue
                if end - start == 1:
                    result[start:end] = paged_dash_decode(
                        q[start:end],
                        k[start:end],
                        v[start:end],
                        iq[start:end],
                        ik[start:end],
                        roles[start:end],
                        cache,
                        metadata.block_table[request : request + 1],
                        metadata.seq_lens[request : request + 1],
                        slots[start:end],
                        self.dash_router.head_cls,
                        self.sp,
                        self.layer_idx,
                        self.max_model_len,
                        metadata.rswa_prefix_lens[request : request + 1]
                        if metadata.rswa_prefix_lens is not None
                        else None,
                        backend=self.dash_backend,
                        cached_summaries=True,
                        positions=positions[:, start:end],
                    )
                else:
                    result[start:end] = paged_dash_prefill(
                        q[start:end],
                        k[start:end],
                        v[start:end],
                        iq[start:end],
                        ik[start:end],
                        roles[start:end],
                        positions[:, start:end],
                        cache,
                        metadata.block_table[request : request + 1],
                        length,
                        slots[start:end],
                        self.dash_router.head_cls,
                        self.sp,
                        self.layer_idx,
                        self.dash_backend,
                    )
