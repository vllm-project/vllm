# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV-balanced MLA core ("balanced") of the specialized decode: the device-planned
split-KV FP8 MLA decode of `nvidia/kv_balanced_mla` in place of tokenspeed.

tokenspeed splits every request's KV into the same number of chunks, fixed per batch
shape, so the longest request of a mixed batch sets the layer time. The balanced core
plans each request's split count from this rank's KV lengths on the device (one small
kernel per decode step, launched by the step's first MLA layer and reused by the
others) and one wave of persistent clusters walks the planned work list. Under DCP it
reads the local KV shard, masks causally in global positions and returns the log2 LSE
that `DcpValue` merges, as tokenspeed does.

Precision is tokenspeed_mla_decode's (the stock TOKENSPEED_MLA backend): FP32 split-KV
partials and its P handling (row max kept while it grows by at most 2^8, P stored as
E4M3 unscaled), so only the split points (a summation order) differ from stock.

One `BalancedMla` per model: the kernels per query length q, their plan buffers and
one split-KV scratch, all allocated while the providers are built (before the KV
cache is sized)."""

import torch

# The most splits of one request (one 2-CTA cluster each) per (DCP, requests up to),
# measured on GB300 at q = 5. At TP8 a request is one query tile (12 heads x q rows),
# so a few long requests need more splits than the kernel's default of 8 to fill the
# GPU, and a lone one more still (R1 128K: 33 us at 32, 27 us at 64; from R2 64 loses
# 1-3 us to the larger reducer). Under DCP the 96 gathered heads give 4 tiles per
# request at q = 5 (R1 128K: 18.6 us at 8, 15.3 us at 16). Past 12 requests (q = 1)
# 8 splits fill the GPU and keep the FP32 split-KV scratch at 128 MiB for 64.
CAPACITY = {False: ((1, 64), (12, 32), (64, 8)), True: ((1, 16), (64, 8))}
BALANCED_MAX_REQUESTS = 64  # the most requests CAPACITY plans
# The shortest split, a quarter of the kernel's default: short decode batches still
# spread over the SMs (TP8 R1 8K: 13.3 us at 1024 tokens, 10.5 us at 256).
MIN_SPLIT_TOKENS = 256


def reducer_d_tiles(rows: int, device: torch.device) -> int:
    """D512 bands per row of the split-KV reduction of `rows` query rows:
    tokenspeed_mla_decode's choice, 2 when the rows fill at most half the SMs (TP8
    R1 at q = 5: 60 rows; -1 us at 8K-128K, GB300), else 1."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return 2 if -(-rows * 2 // sms) < 2 * -(-rows // sms) else 1


def supported(heads: int, q: int, device: torch.device) -> bool:
    """Whether the balanced core runs `heads` x `q` query rows (M128 tiles)."""
    from vllm.models.kimi_k3.nvidia.kv_balanced_mla.core import supports

    return supports(heads, q, device)


class BalancedMla:
    """The balanced cores of `heads` decode heads (under DCP: every rank's)."""

    def __init__(self, heads: int, device: torch.device, dcp: bool):
        self.heads, self.device, self.dcp = heads, device, dcp
        # Per (q, capacity): the core and its plan buffers (counts, work list).
        self.cores: dict = {}
        self.plans: dict = {}
        self.workspace = torch.empty(0, dtype=torch.int8, device=device)

    def key(self, q: int, requests: int) -> tuple[int, int]:
        """(q, capacity) of a step of `requests`: the core that decodes it."""
        return q, next(c for most, c in CAPACITY[self.dcp] if requests <= most)

    def require(self, q: int, requests: int):
        """Prepare the core of `q` rows per request for `requests`."""
        from vllm.models.kimi_k3.nvidia.kv_balanced_mla.core import (
            TOKENSPEED_P_SCALE,
            KvBalancedMla,
        )

        key = self.key(q, requests)
        core = self.cores.get(key)
        if core is not None and core.max_requests >= requests:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the balanced MLA core must be prepared before capture")
        heads, device, capacity = self.heads, self.device, key[1]
        core = KvBalancedMla(
            heads,
            q,
            requests,
            True,
            device,
            capacity,
            MIN_SPLIT_TOKENS,
            partial_fp16=False,
            p_scale=TOKENSPEED_P_SCALE,
            reducer_d_tiles=reducer_d_tiles(heads * q * requests, device),
        )
        i32 = dict(dtype=torch.int32, device=device)
        counts = torch.zeros(requests, **i32)
        work = torch.full((core.list_len,), -1, **i32)
        self.cores[key], self.plans[key] = core, (counts, work)
        if self.workspace.numel() < core.workspace_bytes:  # stream ordered: shared
            self.workspace = torch.empty(
                core.workspace_bytes, dtype=torch.int8, device=device
            )

    def plan(self, q: int, lengths: torch.Tensor):
        """Plan this step's split counts from `lengths` (this rank's KV lengths, one
        per request): once per step, before the decodes of its MLA layers."""
        key = self.key(q, lengths.numel())
        self.cores[key].plan(lengths, *self.plans[key])

    def decode(self, query, cache, block_table, lengths, scales, out, lse=None, **dcp):
        """BF16 `out` [B, q, heads, 512] (and the log2 `lse` [B, q, heads]) of the
        FP8 `query` [B, q, heads, 576] over the FP8 paged `cache` with the current
        plan, PDL chained. Under DCP `dcp` holds `cp` (world, rank, interleave)
        and `totals`, the requests' global lengths for the causal mask."""
        q, (softmax, output_scale) = query.shape[1], scales
        key = self.key(q, query.shape[0])
        self.cores[key].decode(
            query.reshape(-1, self.heads, 576),
            cache,
            block_table,
            lengths,
            softmax,
            output_scale,
            *self.plans[key],
            self.workspace,
            lse=lse,
            causal_lens=dcp.get("totals"),
            cp=dcp.get("cp", (1, 0, 1)),
            out=out,
            pdl=True,
        )
