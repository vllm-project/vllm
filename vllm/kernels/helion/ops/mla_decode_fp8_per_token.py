# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import helion
import helion.language as hl
import torch
from helion._utils import cdiv

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "MLA Decode Helion kernels requires helion to be installed. "
        "Install it with: pip install helion"
    )

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_mla_decode_kv_split_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    batch_sizes = [1, 4, 16, 128, 256]
    seq_lens = [512, 2048, 8192, 32768]
    head_counts = [16, 32]

    HEADS_PER_BLOCK = 4
    NUM_KV_SPLITS = 4
    PAGE_SIZE = 16
    latent_dim = 512
    rope_dim = 64
    logit_cap = None

    inputs: dict[CaseKey, tuple[Any, ...]] = {}

    for batch, seq_len, heads in product(batch_sizes, seq_lens, head_counts):
        num_pages = cdiv(seq_len, PAGE_SIZE)

        q_absorbed = torch.randn(
            batch,
            heads,
            latent_dim + rope_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

        latent_kv = torch.randn(
            num_pages,
            PAGE_SIZE,
            1,
            latent_dim + rope_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)

        req_to_tokens = (
            torch.arange(num_pages, device="cuda")
            .unsqueeze(0)
            .expand(batch, num_pages)
            .contiguous()
        )

        b_seq_len = torch.full(
            (batch,),
            seq_len,
            device="cuda",
            dtype=torch.int32,
        )

        attn_out = torch.empty(
            (
                batch,
                heads,
                NUM_KV_SPLITS,
                latent_dim + 1,
            ),
            device="cuda",
            dtype=torch.float32,
        )

        kv_dequant = torch.tensor(
            [1.0],
            device="cuda",
            dtype=torch.float32,
        )

        sm_scale = torch.tensor(
            [1.0 / (latent_dim**0.5)],
            device="cuda",
            dtype=torch.float32,
        )

        key = CaseKey({"batch": batch, "seqlen": seq_len, "headspb": heads})

        inputs[key] = (
            q_absorbed,
            latent_kv,
            attn_out,
            kv_dequant,
            sm_scale,
            b_seq_len,
            req_to_tokens,
            HEADS_PER_BLOCK,
            NUM_KV_SPLITS,
            PAGE_SIZE,
            latent_dim,
            rope_dim,
            logit_cap,
        )

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_mla_decode_kv_split_config(
    args: tuple[Any, ...], config_keys: list[CaseKey]
) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest batch size among available configs
         (exact match preferred).
      2. Among the seq_len values tuned for that batch size, pick
         the smallest seq_len >= the input's seq_len. If the input is
         larger than all available seq_lens, fall back to the largest.
      3. Among the headspb values tuned for that (batch, seqlen), pick
         the smallest headspb >= the input's headspb. If the input is
         larger than all available headspb, fall back to the largest.
    """
    if not config_keys:
        return None
    # print("PICKER")
    q_absorbed = args[0]
    b_seq_len = args[5]

    batch = int(q_absorbed.shape[0])
    heads = int(q_absorbed.shape[1])
    seq_len = int(b_seq_len.max().item())

    cache_key = (batch, seq_len, heads)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    by_batch: dict[int, dict[int, list[int]]] = {}
    for k in config_keys:
        if k.is_default():
            continue
        by_batch.setdefault(k["batch"], {}).setdefault(k["seqlen"], []).append(
            k["headspb"]
        )

    if not by_batch:
        return None

    best_batch = min(by_batch, key=lambda b: abs(b - batch))
    by_seqlen = by_batch[best_batch]
    available_seqlens = sorted(by_seqlen)
    best_seqlen = next(
        (s for s in available_seqlens if s >= seq_len), available_seqlens[-1]
    )
    available_heads = sorted(by_seqlen[best_seqlen])
    best_heads = next((h for h in available_heads if h >= heads), available_heads[-1])

    result = CaseKey(
        {"batch": best_batch, "seqlen": best_seqlen, "headspb": best_heads}
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    q_absorbed: torch.Tensor,  # query (batch, heads, d_model)
    latent_kv: torch.Tensor,  # latent kv vector (num_pages, page_size, latent+rope)
    attn_out: torch.Tensor,  # output (batch, heads, d_model)
    kv_dequant: torch.Tensor,  # scale to dequant fp8 latent kv cache
    sm_scale: torch.Tensor,  # normalising scale for attention product
    B_seq_len: torch.Tensor,  # seq_len of each request (batch,)
    Req_to_Tokens: torch.Tensor,  # page table mapping
    HEADS_PER_BLOCK: int,  # heads per thread block
    NUM_KV_SPLITS: int,  # no. of splits of kv_cache; determines -1 dim of launch grid
    PAGE_SIZE: int,  # page size: 16
    latent_dim: int,  # 512
    rope_dim: int,  # 64
    logit_cap: float | None = None,
) -> None:
    return


def baseline(
    q_absorbed: torch.Tensor,
    latent_kv: torch.Tensor,
    attn_out: torch.Tensor,
    kv_dequant: torch.Tensor,
    sm_scale: torch.Tensor,
    B_seq_len: torch.Tensor,
    Req_to_Tokens: torch.Tensor,
    HEADS_PER_BLOCK: int,
    NUM_KV_SPLITS: int,
    PAGE_SIZE: int,
    latent_dim: int,
    rope_dim: hl.constexpr,
    logit_cap: hl.constexpr = None,
) -> torch.Tensor:
    """Baseline wrapper that allocates the output buffer and forwards all

    arguments to the internal Triton configuration function.
    """
    sm_scale = float(sm_scale.item())
    logit_cap = float(logit_cap) if logit_cap is not None else 0.0

    from vllm.v1.attention.ops.triton_decode_attention import _decode_grouped_att_m_fwd

    _decode_grouped_att_m_fwd(
        q=q_absorbed,
        k_buffer=latent_kv,
        v_buffer=latent_kv[..., :latent_dim],
        att_out=attn_out,
        Req_to_tokens=Req_to_Tokens,
        B_Seqlen=B_seq_len,
        num_kv_splits=NUM_KV_SPLITS,
        sm_scale=sm_scale,
        page_size=PAGE_SIZE,
        logit_cap=logit_cap,
        k_scale=kv_dequant,
        v_scale=kv_dequant,
        is_mla=True,
    )


@register_kernel(
    mutates_args=["attn_out"],
    config_picker=pick_mla_decode_kv_split_config,
    input_generator=generate_mla_decode_kv_split_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
    ),
)
def mla_decode_kv_split(
    q_absorbed: torch.Tensor,  # query (batch, heads, d_model)
    latent_kv: torch.Tensor,  # latent kv vector (num_pages, page_size, latent+rope)
    attn_out: torch.Tensor,  # output (batch, heads, d_model)
    kv_dequant: torch.Tensor,  # scale to dequant fp8 latent kv cache
    sm_scale: torch.Tensor,  # normalising scale for attention product
    B_seq_len: torch.Tensor,  # seq_len of each request (batch,)
    Req_to_Tokens: torch.Tensor,  # page table mapping
    HEADS_PER_BLOCK: int,  # heads per thread block
    NUM_KV_SPLITS: int,  # no. of splits of kv_cache; determines -1 dim of launch grid
    PAGE_SIZE: int,  # page size: 16
    latent_dim: int,  # 512
    rope_dim: int,  # 64
    logit_cap: float | None = None,
) -> None:
    assert q_absorbed.ndim == 3
    assert latent_kv.ndim == 4 and latent_kv.dtype == torch.float8_e4m3fn
    batch = q_absorbed.shape[0]
    heads = q_absorbed.shape[1]
    heads_group = cdiv(heads, HEADS_PER_BLOCK)

    latent_dim = hl.specialize(latent_dim)
    rope_dim = hl.specialize(rope_dim)
    HEADS_PER_BLOCK = hl.specialize(HEADS_PER_BLOCK)

    num_pages = latent_kv.shape[0]
    latent_kv = latent_kv.reshape(num_pages * PAGE_SIZE, heads, -1)

    grid = (batch, heads_group, NUM_KV_SPLITS)

    for seq, head_group, split in hl.tile(grid, block_size=[1, 1, 1]):
        q_heads = head_group.begin * HEADS_PER_BLOCK + hl.arange(HEADS_PER_BLOCK)
        mask_h = q_heads < heads

        if rope_dim > 0:
            q_latent = hl.load(
                q_absorbed,
                [seq.begin, q_heads, hl.arange(latent_dim)],
                extra_mask=mask_h[:, None],
            )
            q_rope = hl.load(
                q_absorbed,
                [seq.begin, q_heads, hl.arange(latent_dim, latent_dim + rope_dim)],
                extra_mask=mask_h[:, None],
            )
        else:
            q_latent = hl.load(
                q_absorbed,
                [seq.begin, q_heads, hl.arange(latent_dim)],
                extra_mask=mask_h[:, None],
            )

        cur_batch_seq_len = B_seq_len[seq.begin]
        kv_len_per_split = cdiv(cur_batch_seq_len, NUM_KV_SPLITS)

        e_max = hl.zeros([HEADS_PER_BLOCK], dtype=torch.float32) - float("inf")
        e_sum = hl.zeros([HEADS_PER_BLOCK], dtype=torch.float32)
        acc = hl.zeros([HEADS_PER_BLOCK, latent_dim], dtype=torch.float32)

        split_kv_start = kv_len_per_split * split.begin
        split_kv_end = torch.min(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            for kv_block in hl.tile(split_kv_start, split_kv_end, block_size=None):
                offs_n = kv_block.begin + hl.arange(0, kv_block.block_size)
                log_page_offs = offs_n // PAGE_SIZE

                kv_page_number = hl.load(
                    Req_to_Tokens,
                    [seq.begin, log_page_offs],
                    extra_mask=offs_n < split_kv_end,
                )

                page_offs = offs_n % PAGE_SIZE
                kv_loc = kv_page_number * PAGE_SIZE + page_offs
                dequant = hl.load(kv_dequant, [0])
                if rope_dim > 0:
                    k_latent = hl.load(
                        latent_kv,
                        [kv_loc, 0, hl.arange(latent_dim)],
                        extra_mask=offs_n[:, None] < split_kv_end,
                    )
                    k_latent = (k_latent.to(torch.float32) * dequant).to(q_latent.dtype)
                    qk = hl.dot(q_latent, torch.transpose(k_latent, 0, 1))
                    k_rope = hl.load(
                        latent_kv,
                        [kv_loc, 0, hl.arange(latent_dim, latent_dim + rope_dim)],
                        extra_mask=offs_n[:, None] < split_kv_end,
                    )
                    k_rope = (k_rope.to(torch.float32) * dequant).to(q_rope.dtype)
                    qk = hl.dot(q_rope, torch.transpose(k_rope, 0, 1), qk)
                else:
                    k_latent = hl.load(
                        latent_kv,
                        [kv_loc, 0, hl.arange(latent_dim)],
                        extra_mask=offs_n[:, None] < split_kv_end,
                    )
                    k_latent = (k_latent.to(torch.float32) * dequant).to(q_latent.dtype)
                    qk = hl.dot(q_latent, torch.transpose(k_latent, 0, 1))

                scale = hl.load(sm_scale, [0])
                qk *= scale

                if logit_cap is not None and logit_cap > 1:
                    qk = logit_cap * torch.tanh(qk / logit_cap)
                token_mask = offs_n < split_kv_end
                qk = torch.where(
                    mask_h[:, None] & token_mask[None, :], qk, float("-inf")
                )

                v = k_latent

                tile_max = torch.amax(qk, dim=-1)
                n_e_max = torch.maximum(tile_max, e_max)
                re_scale = torch.exp(e_max - n_e_max)
                p = torch.exp(qk - n_e_max.unsqueeze(1))
                acc *= re_scale[:, None]
                tmp = hl.dot(p.to(v.dtype), v)
                acc += tmp

                e_sum = e_sum * re_scale + torch.sum(p, dim=-1)
                e_max = n_e_max
        hl.store(
            attn_out,
            [seq.begin, q_heads, split.begin, hl.arange(latent_dim)],
            acc / e_sum[:, None],
            extra_mask=mask_h[:, None],
        )
        hl.store(
            attn_out,
            [seq.begin, q_heads, split.begin, latent_dim],
            e_max + torch.log(e_sum),
            extra_mask=mask_h,
        )


def generate_softmax_reduce_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    batch_sizes = [1, 4, 16]
    seq_lens = [512]
    head_counts = [16]

    NUM_KV_SPLITS = 4
    latent_dim = 512

    inputs: dict[CaseKey, tuple[Any, ...]] = {}

    for batch, seq_len, heads in product(batch_sizes, seq_lens, head_counts):
        # partial_attn: output of mla_decode_kv_split
        # shape: (batch, heads, NUM_KV_SPLITS, latent_dim + 1)
        # the +1 slot stores the log-sum-exp scalar per split
        partial_attn = torch.randn(
            batch,
            heads,
            NUM_KV_SPLITS,
            latent_dim + 1,
            device="cuda",
            dtype=torch.float32,
        )

        attn_out = torch.empty(
            batch,
            heads,
            latent_dim,
            device="cuda",
            dtype=torch.float32,
        )

        lse_out = torch.empty(
            batch,
            heads,
            device="cuda",
            dtype=torch.float32,
        )

        b_seq_len = torch.full(
            (batch,),
            seq_len,
            device="cuda",
            dtype=torch.int32,
        )

        key = CaseKey({"batch": batch, "heads": heads})

        inputs[key] = (
            partial_attn,
            attn_out,
            lse_out,
            b_seq_len,
            NUM_KV_SPLITS,
            latent_dim,
        )

    return inputs


# Note: separate cache from the kv_split one — different arg layout
_softmax_reduce_pick_cache: dict[tuple[int, int], CaseKey | None] = {}


def pick_softmax_reduce_config(
    args: tuple[Any, ...], config_keys: list[CaseKey]
) -> CaseKey | None:
    """Pick the best pre-tuned config for the softmax-reduce kernel.

    1. Closest batch size (exact preferred).
    2. Smallest heads >= input heads; fall back to largest.
    """
    if not config_keys:
        return None

    partial_attn = args[0]  # (batch, heads, NUM_KV_SPLITS, latent_dim+1)

    batch = int(partial_attn.shape[0])
    heads = int(partial_attn.shape[1])

    cache_key = (batch, heads)
    cached = _softmax_reduce_pick_cache.get(cache_key)
    if cached is not None:
        return cached

    by_batch: dict[int, list[int]] = {}
    for k in config_keys:
        if k.is_default():
            continue
        by_batch.setdefault(k["batch"], []).append(k["heads"])

    if not by_batch:
        return None

    # 1. Find closest batch size
    best_batch = min(by_batch, key=lambda b: abs(b - batch))

    # 2. Find smallest heads >= input heads; fall back to largest
    available_heads = sorted(by_batch[best_batch])
    best_heads = next((h for h in available_heads if h >= heads), available_heads[-1])

    result = CaseKey({"batch": best_batch, "heads": best_heads})
    _softmax_reduce_pick_cache[cache_key] = result
    return result


def fake_impl_softmax_reduce(
    partial_attn: torch.Tensor,  # (batch, heads, NUM_KV_SPLITS, latent_dim + 1)
    attn_out: torch.Tensor,  # (batch, heads, latent_dim)
    lse_out: torch.Tensor,  # (batch, heads)
    B_seq_len: torch.Tensor,  # (batch,)
    NUM_KV_SPLITS: int,
    latent_dim: int,
) -> None:
    """No-op fake implementation for torch.compile / export tracing."""
    return


def baseline_softmax_reduce(
    partial_attn: torch.Tensor,
    attn_out: torch.Tensor,
    lse_out: torch.Tensor,
    B_seq_len: torch.Tensor,
    NUM_KV_SPLITS: int,
    latent_dim: int,
):
    from vllm.v1.attention.ops.triton_decode_attention import (
        _decode_softmax_reducev_fwd,
    )

    _decode_softmax_reducev_fwd(
        partial_attn,
        partial_attn,
        attn_out,
        lse_out,
        partial_attn[..., :latent_dim],
        B_seq_len,
        NUM_KV_SPLITS,
    )


@register_kernel(
    mutates_args=["attn_out", "lse_out"],
    config_picker=pick_softmax_reduce_config,
    input_generator=generate_softmax_reduce_inputs,
    fake_impl=fake_impl_softmax_reduce,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline_softmax_reduce,
    ),
)
def mla_decode_softmax_reduce(
    partial_attn: torch.Tensor,
    attn_out: torch.Tensor,
    lse_out: torch.Tensor,
    B_seq_len: torch.Tensor,
    NUM_KV_SPLITS: int,
    latent_dim: int,
) -> None:
    batches = partial_attn.shape[0]
    heads = partial_attn.shape[1]

    latent_dim = hl.specialize(latent_dim)
    NUM_KV_SPLITS = hl.specialize(NUM_KV_SPLITS)
    grid = (batches, heads)

    for seq, head in hl.tile(grid, block_size=[1, 1]):
        seq_len = B_seq_len[seq.begin]

        e_sum = hl.zeros([])
        e_max = hl.full([], -float("inf"))
        acc = hl.zeros([latent_dim], dtype=torch.float32)

        for kv_split in hl.tile(NUM_KV_SPLITS, block_size=1):
            kv_len_per_split = cdiv(seq_len, NUM_KV_SPLITS)
            split_start = kv_split.begin * kv_len_per_split
            split_end = torch.min(split_start + kv_len_per_split, seq_len)

            if split_end > split_start:
                tv = hl.load(
                    partial_attn,
                    [seq.begin, head.begin, kv_split.begin, hl.arange(latent_dim)],
                )
                tlogic = hl.load(
                    partial_attn, [seq.begin, head.begin, kv_split.begin, latent_dim]
                )

                n_e_max = torch.maximum(tlogic, e_max)
                old_scale = torch.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = torch.exp(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

        result = acc / e_sum
        lse_val = e_max + torch.log(e_sum)

        hl.store(attn_out, [seq.begin, head.begin, hl.arange(latent_dim)], result)
        hl.store(lse_out, [seq.begin, head.begin], lse_val)
