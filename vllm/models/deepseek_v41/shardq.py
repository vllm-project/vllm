# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ShardQ support for DeepSeek-V4.1 prefill.

ShardQ (`vllm.v1.attention.ops.shardq`) gives every TP rank a zigzag slice of
a prefill step's tokens.  V4.1's attention runs its prefill as one chunk per
request, so this module re-expresses the slice as a `CommonAttentionMetadata`
whose "requests" are the rank's contiguous token runs: a rank owns at most two
runs per request (the early and late zigzag chunk), and each run is a chunk
whose context ends at its last token.  The existing V4.1 SWA / FlashMLA
metadata builders then build the sharded metadata unchanged.

The compact top-k layout is the one the sparse indexer already writes under
ShardQ: the step's decode tokens in place, then this rank's tokens ascending.
The runs are emitted in that same ascending token order, so the attention reads
the indexer's output without a remap.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import CommonAttentionMetadata, max_decode_query_len
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.shardq import (
    ShardQMetadata,
    ShardQSplit,
    build_shardq_split,
)

logger = init_logger(__name__)


@dataclass
class DSV41ShardQ:
    """One step's ShardQ view for V4.1.

    `sharded_cm` describes this rank's prefill runs as requests; the builders
    turn it into the SWA and FlashMLA metadata `forward_mqa` reads.  The
    exchange tensors are shared by every builder of the step; each builder
    makes its own `ShardQMetadata` so the per-builder sharded metadata does
    not overwrite another builder's.
    """

    split: ShardQSplit
    sharded_cm: CommonAttentionMetadata
    run_req: torch.Tensor
    token_ids: torch.Tensor
    rank_offsets: torch.Tensor
    exchange: ShardQMetadata = field(init=False)

    def __post_init__(self) -> None:
        self.exchange = ShardQMetadata(
            num_decode_tokens=self.split.num_decode_tokens,
            num_tokens_per_rank=self.split.num_tokens_per_rank.tolist(),
            token_ids=self.token_ids,
            rank_offsets=self.rank_offsets,
        )

    def new_exchange(self) -> ShardQMetadata:
        """A fresh exchange metadata, for a builder to attach its metadata to."""
        return ShardQMetadata(
            num_decode_tokens=self.exchange.num_decode_tokens,
            num_tokens_per_rank=self.exchange.num_tokens_per_rank,
            token_ids=self.exchange.token_ids,
            rank_offsets=self.exchange.rank_offsets,
        )

    def shard_replay_start(
        self, replay_start: torch.Tensor | None
    ) -> torch.Tensor | None:
        if replay_start is None:
            return None
        return replay_start.index_select(0, self.run_req)


# H20 crossover for DSV4.1: a single-request step is 7.5% slower at 1K, a wash
# at 2K-2.5K, and wins from 3K (1.5%, growing with context).  The same holds
# per request: a batch of short requests does not amortize the exchanges even
# when the step's total is large, so the main path also requires every request
# to contribute SHARDQ_MIN_REQUEST_TOKENS.  A short step whose own long-context
# requests reach SHARDQ_LONG_CONTEXT_MIN_TOKENS (a tail chunk) still pays,
# since the indexer's cost per row grows with the context length.
SHARDQ_MIN_TOKENS = 3072
SHARDQ_MIN_REQUEST_TOKENS = 2048
SHARDQ_LONG_CONTEXT_MIN_SEQ = 32768
SHARDQ_LONG_CONTEXT_MIN_TOKENS = 1024


def dsv41_shardq_eligible(
    cm: CommonAttentionMetadata,
    vllm_config: "VllmConfig",  # noqa: F821
    split: ShardQSplit,
) -> bool:
    """Whether this step should run its prefill under ShardQ.

    A run is one zigzag chunk of one request, so its width is that request's
    chunk width, floor(query_len / (2 * tp)).  The metadata builders read a
    run no wider than `max_decode_query_len` as a decode.  Both inputs are
    the step's shared query lengths, so every TP rank reaches the same
    decision and no rank can stay behind while its peers exchange.

    Short steps stay head-sharded: below `SHARDQ_MIN_TOKENS` prefill tokens,
    or when a request in the step is shorter than
    `SHARDQ_MIN_REQUEST_TOKENS`, the exchanges cost more than the sharded
    indexer saves, unless the step's long-context requests themselves
    contribute enough tokens to be a tail chunk.  Counting only those
    requests keeps a long context that this step barely extends from dragging
    a batch of short prefills into the split.
    """
    max_decode = max_decode_query_len(vllm_config)
    num_chunks = 2 * int(split.num_tokens_per_rank.size)
    spec_config = vllm_config.speculative_config
    num_spec = spec_config.num_speculative_tokens if spec_config else 0
    num_decodes, _, _, num_prefill_tokens = split_decodes_and_prefills(
        cm, decode_threshold=1 + num_spec
    )
    prefill_lens = cm.query_start_loc_cpu.diff().numpy()[num_decodes:]
    if prefill_lens.size == 0:
        return False
    seq_lens = cm.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    prefill_seq_lens = seq_lens[num_decodes : cm.num_reqs].numpy()
    long_context_tokens = int(
        prefill_lens[prefill_seq_lens >= SHARDQ_LONG_CONTEXT_MIN_SEQ].sum()
    )
    if long_context_tokens < SHARDQ_LONG_CONTEXT_MIN_TOKENS:
        if num_prefill_tokens < SHARDQ_MIN_TOKENS:
            return False
        if int(prefill_lens.min()) < SHARDQ_MIN_REQUEST_TOKENS:
            return False
    return int(prefill_lens.min()) >= (max_decode + 1) * num_chunks


def build_dsv41_shardq(
    cm: CommonAttentionMetadata,
    vllm_config: "VllmConfig",  # noqa: F821
    device: torch.device,
) -> DSV41ShardQ | None:
    """This step's ShardQ, or None when the step stays head-sharded.

    The result is memoized on `cm`: every layer group's builder asks for the
    same step's split, and the CPU scan and H2D copies are not cheap enough to
    repeat per layer.
    """
    cached = getattr(cm, "_dsv41_shardq", False)
    if cached is not False:
        return cached
    result = _build_dsv41_shardq(cm, vllm_config, device)
    cm._dsv41_shardq = result  # type: ignore[attr-defined]
    return result


def _build_dsv41_shardq(
    cm: CommonAttentionMetadata,
    vllm_config: "VllmConfig",  # noqa: F821
    device: torch.device,
) -> DSV41ShardQ | None:
    split = build_shardq_split(cm, vllm_config, eligibility=dsv41_shardq_eligible)
    if split is None:
        return None
    num_decode_tokens = split.num_decode_tokens
    local = split.local_token_ids + num_decode_tokens
    if local.size == 0:
        return None

    qsl_cpu = cm.query_start_loc_cpu.numpy()
    seq_cpu = cm.seq_lens_cpu_upper_bound
    assert seq_cpu is not None
    seq_cpu = seq_cpu.numpy()

    # Request index of each local token.  The batch is request-major, so
    # searchsorted over the cumulative query starts is exact.
    req = np.searchsorted(qsl_cpu[1:], local, side="right")

    # Split the ascending token list into contiguous runs of one request.  A
    # zigzag slice yields at most two runs per request.
    boundary = np.ones(local.size, dtype=bool)
    boundary[1:] = (local[1:] != local[:-1] + 1) | (req[1:] != req[:-1])
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:], local.size]
    run_qlen = (ends - starts).astype(np.int32)
    run_req = req[starts]
    run_last = local[ends - 1]

    # ``seq_lens`` is the uncompressed context length at each run's last token,
    # which the builders compress themselves.
    run_seq = (seq_cpu[run_req] - qsl_cpu[run_req + 1] + run_last + 1).astype(np.int32)

    run_qsl_cpu = np.zeros(starts.size + 1, dtype=np.int32)
    np.cumsum(run_qlen, out=run_qsl_cpu[1:])
    run_req_gpu = async_tensor_h2d(run_req.astype(np.int32), device=device)
    token_gpu = async_tensor_h2d(local.astype(np.int64), device=device)

    sharded_cm = CommonAttentionMetadata(
        query_start_loc=async_tensor_h2d(run_qsl_cpu, device=device),
        query_start_loc_cpu=torch.from_numpy(run_qsl_cpu),
        seq_lens=async_tensor_h2d(run_seq, device=device),
        seq_lens_cpu_upper_bound=torch.from_numpy(run_seq),
        num_reqs=int(starts.size),
        num_actual_tokens=int(local.size),
        max_query_len=int(run_qlen.max()),
        max_seq_len=int(run_seq.max()),
        block_table_tensor=cm.block_table_tensor.index_select(0, run_req_gpu),
        slot_mapping=cm.slot_mapping.index_select(0, token_gpu),
        positions=(
            cm.positions.index_select(0, token_gpu)
            if cm.positions is not None
            else None
        ),
        is_prefilling=torch.ones(starts.size, dtype=torch.bool, device=device),
        req_idx=run_req.astype(np.int32),
    )
    logger.info_once(
        "DeepSeek-V4.1 ShardQ prefill: %d requests, %d local tokens on this rank.",
        cm.num_reqs,
        local.size,
    )
    return DSV41ShardQ(
        split=split,
        sharded_cm=sharded_cm,
        run_req=run_req_gpu,
        token_ids=async_tensor_h2d(split.token_ids.astype(np.int32), device=device),
        rank_offsets=async_tensor_h2d(
            np.r_[0, np.cumsum(split.num_tokens_per_rank)].astype(np.int32),
            device=device,
        ),
    )
