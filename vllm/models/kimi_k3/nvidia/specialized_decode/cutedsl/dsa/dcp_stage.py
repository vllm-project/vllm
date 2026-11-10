# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP stage of a K3.1 DSA layer (DCP = TP; each rank's cache holds 1 / DCP of every
context):

  publish          the q mailbox the query stage publishes this rank's heads into
                   (every rank's heads land in each rank's mailbox)
  local slots      the global top-k ids filtered to this rank's slots and compacted
                   to a prefix whose length bounds the sparse decode's work (about
                   1 / DCP of the top-k)
  DcpValue         pushes every rank's partial rows (BF16 latent rows, or
                   "proj": BF16 value rows through W_UV) and LSE of the gathered
                   heads to their owners (only the LSE of a row whose local
                   valid count is 0), merges this rank's heads, W_UV and the
                   gate. Under "proj" the value rows come from one batched GEMM
                   over the gathered heads (the stock W_UV bmm) ahead of
                   DcpValue, whose projecting CTAs only push them: in-kernel
                   they ran W_UV on mma.sync, one CTA per gathered head walking
                   the batch (GB300 DCP8, CUPTI 47-51 us at 320 rows)."""

from functools import cache

import cutlass
import torch

from ..common.host import stream, to_cute
from ..mla.dcp import PROJECTED, compile_value


@cache
def projected_rows(device, rows, gathered):
    """The "proj" value rows [DCP * H * rows * 128] BF16 of the W_UV bmm, shared by
    every DSA layer of a device (allocated before the KV cache is sized)."""
    return torch.empty(gathered * rows * 128, dtype=torch.bfloat16, device=device)


def compile_dcp_value(
    heads,
    dcp,
    batch,
    uv,
    pitch,
    device,
    sms,
    q,
    q_heads,
    fp8,
    lse_base_e,
    partials,
    cap=64,
    pushed=False,
):
    """DcpValue of one batch (DSA layout over the `q_heads`-padded mailbox of
    `cap` rows; `partials`: the DcpBox's; `pushed`: the sparse decode pushed
    the rows, attn_sparse)."""
    return compile_value(
        heads,
        dcp,
        batch,
        cap,
        uv,
        pitch,
        device,
        sms,
        q,
        q_heads,
        dsa_bf16=not fp8,
        lse_base_e=lse_base_e,
        partials=partials,
        preprojected=partials == PROJECTED,
        pushed=pushed,
    )


def publish_args(box, parity):
    """The query's publish arguments: this parity's q mailbox, its multicast address
    and this rank."""
    mailbox = to_cute(box.q[parity].view(torch.uint8))
    return mailbox, box.mc(parity), cutlass.Int32(box.rank)


def local_slots(metadata, topk, dcp, rank, interleave, stride_rows):
    """This rank's flat rows of the global top-k ids `topk` [B, K] (compacted to
    a prefix) and their counts."""
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )

    batch = topk.shape[0]
    return triton_filter_and_convert_dcp_index(
        metadata.req_id_per_token[:batch],
        metadata.block_table,
        topk,
        dcp_size=dcp,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        BLOCK_SIZE=metadata.block_size,
        BLOCK_STRIDE_ROWS=stride_rows,
        NUM_TOPK_TOKENS=topk.shape[1],
        return_valid_counts=True,
    )


def run_value(
    kernel, box, parity, publish, heads, core, lengths, uv, projected, out, w_uv=None
):
    """DcpValue over the sparse decode's `core` output: its padded-head latent
    rows, LSE and per-row valid counts (`lengths`: the requests' global lengths)
    -> `out` [B, H * 128]. Under "proj" `w_uv` is the gathered W_UV [DCP * H,
    512, 128] the bmm projects the rows through first."""
    latent, lse, valid = core
    batch = projected.shape[0]
    lens, counts = to_cute(lengths, 4), to_cute(valid[:batch], 4)
    mail = box.mailbox(batch)
    boxes = to_cute(mail.partial[parity]), to_cute(mail.lse[parity], 4)
    peers = to_cute(mail.peers, 8), to_cute(mail.lse_peers, 8)
    gate = to_cute(projected[:, 2112:])
    # DcpValue's [B, 1, DCP * H] views of the sparse decode's padded-head rows and LSE.
    gathered, padded = heads * box.size, box.q_heads
    latent = latent.as_strided((batch, 1, gathered, 512), (padded * 512, 0, 512, 1))
    lse = lse.as_strided((batch, 1, gathered), (padded, 0, 1))
    mailbox, _, rank = publish
    if mail.format == PROJECTED:
        assert w_uv is not None
        rows = projected_rows(out.device, box.cap, gathered)
        values = rows[: gathered * batch * 128].view(gathered, batch, 128)
        torch.bmm(latent[:, 0].transpose(0, 1), w_uv, out=values)
        uv = to_cute(values)
    v_args = (to_cute(latent), to_cute(lse, 4), counts, *boxes, *peers, mailbox, uv)
    s = stream(out.device)
    kernel(*v_args, gate, lens, to_cute(out), cutlass.Int32(parity), rank, s)
    return out
