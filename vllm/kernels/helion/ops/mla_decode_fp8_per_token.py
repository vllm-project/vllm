# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import helion
import helion.language as hl
import regex as re
import torch
from helion._utils import cdiv

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "MLA Decode Helion kernels requires helion to be installed. "
        "Install it with: pip install helion"
    )


logger = init_logger(__name__)


def generate_mla_decode_kv_split_inputs() -> dict[str, tuple[Any, ...]]:
    intermediate_sizes = [2048, 2880, 4096, 8192, 11008, 14336]

    # Use the same num_tokens values as vLLM's default cudagraph capture sizes.
    # See vllm/config/vllm.py _set_cudagraph_sizes() for the canonical formula.
    num_tokens_list = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))

    inputs = {}
    for num_tokens in num_tokens_list:
        for intermediate_size in intermediate_sizes:
            input_tensor = torch.randn(
                num_tokens,
                2 * intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

            config_key = f"intermediate_{intermediate_size}_numtokens_{num_tokens}"
            inputs[config_key] = (input_tensor, scale)

    return inputs


def pick_mla_decode_kv_split_config(
    args: tuple[Any, ...], config_keys: list[str]
) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that intermediate_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "intermediate_{int}_numtokens_{int}".
    """
    if not config_keys:
        return None

    input_tensor, _scale = args
    intermediate_size = input_tensor.shape[-1] // 2
    num_tokens = input_tensor.view(-1, input_tensor.shape[-1]).shape[0]
    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"intermediate_(\d+)_numtokens_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'intermediate_{{int}}_numtokens_{{int}}'"
            )
        isize_str, ntokens_str = match.groups()
        configs.setdefault(int(isize_str), []).append(int(ntokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_isize = min(configs, key=lambda s: abs(s - intermediate_size))
    available_ntokens = sorted(configs[best_isize])
    best_ntokens = next(
        (n for n in available_ntokens if n >= num_tokens), available_ntokens[-1]
    )

    return f"intermediate_{best_isize}_numtokens_{best_ntokens}"

    # we want each thread block to handle a token's single head with a part of kv cache
    """
    the ides is to load a query's head per thread block
    ie kernel launch grid = (batch, head, kv_split)
    each thread block computes partial attention over a query's single head
    over a piece of latent kv cache
    the query is loaded once in thread block,
    then latent vector will be loaded in a loop.
    """


@helion.kernel
def mla_decode_kv_split(
    q_absorbed: torch.Tensor,  # query (batch, heads, d_model)
    latent_kv: torch.Tensor,  # latent kv vector (num_pages, page_size, latent+rope)
    attn_out: torch.Tensor,  # output (batch, heads, d_model)
    kv_dequant: torch.Tensor,  # scale to dequant fp8 latent kv cache
    sm_scale: torch.Tensor,  # normalising scale for attention product
    B_seq_len: torch.Tensor,  # seq_len of each request (batch,)
    Req_to_Tokens: torch.Tensor,  # page table mapping
    NUM_KV_SPLITS: int,  # no. of splits of kv_cache; determines -1 dim of launch grid
    PAGE_SIZE: int,  # page size: 16
    BLOCK_KV: int,  # kv cache to be processed in one iteration
    latent_dim: int,  # 512
    rope_dim: int,  # 64
    logit_cap: float,
):
    assert q_absorbed.ndim == 3
    assert latent_kv.ndim == 3 and latent_kv.dtype == torch.float8_e4m3fn
    batch = q_absorbed.shape[0]
    heads = q_absorbed.shape[1]

    grid = (batch, heads, NUM_KV_SPLITS)

    for seq, head, kv_split in hl.tile(grid, block_size=[1, 1, None]):
        q = q_absorbed[seq, head, :]
        if rope_dim > 0:
            q_latent = q[:, :, :latent_dim]
            q_rope = q[:, :, latent_dim:rope_dim]
        else:
            q_latent = q

        cur_batch_seq_len = B_seq_len[seq]
        kv_len_per_split = cdiv(cur_batch_seq_len, NUM_KV_SPLITS)

        split_kv_start = kv_len_per_split * kv_split
        split_kv_end = torch.min(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        e_max = hl.zeros([], dtype=torch.float32) - float("inf")
        e_sum = hl.zeros([], dtype=torch.float32)
        acc = hl.zeros([latent_dim], dtype=torch.float32)

        if split_kv_end > split_kv_start:
            for start in range(split_kv_start, split_kv_end, BLOCK_KV):
                offs_n = start + hl.arange(0, BLOCK_KV)
                log_page_offs = offs_n // PAGE_SIZE
                kv_page_number = hl.load(
                    Req_to_Tokens,
                    [batch, log_page_offs],
                    extra_mask=offs_n < split_kv_end,
                )

                page_offs = offs_n % PAGE_SIZE
                latent_slice = slice(None, latent_dim)

                k_latent = hl.load(
                    latent_kv,
                    [kv_page_number, page_offs, latent_slice],
                    extra_mask=offs_n < split_kv_end,
                )

                qk = hl.dot(q_latent, k_latent)

                if rope_dim > 0:
                    pos_slice = slice(latent_dim, rope_dim)
                    k_rope = hl.load(
                        latent_kv,
                        [
                            kv_page_number,
                            page_offs,
                            pos_slice,
                        ],
                        extra_mask=offs_n < split_kv_end,
                    )

                    if k_rope.dtype == torch.float8_e4m3fn:
                        k_rope = (k_rope.to(torch.float32) * kv_dequant).to(q.dtype)

                    qk = hl.dot(q_rope, k_rope, qk)

                qk *= sm_scale
                if logit_cap > 0:
                    qk = logit_cap * torch.tanh(qk / logit_cap)
                qk = torch.where(offs_n < split_kv_end, qk, float("-inf"))

                v = torch.transpose(k_latent)

                n_e_max = torch.max(torch.max(qk), e_max)
                re_scale = torch.exp(e_max - n_e_max)
                p = torch.exp(qk - n_e_max)
                acc *= re_scale
                acc += hl.dot(p.to(v.dtype), v)

                e_sum = e_sum * re_scale + torch.sum(p)
                e_max = n_e_max
