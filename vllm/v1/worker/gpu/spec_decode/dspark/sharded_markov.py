# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark Markov sampling with the Markov head sharded by vocabulary.

After the DSpark backbone produces the base logits of its ``N`` draft
positions, the draft tokens are drawn one position at a time: the bias
``markov_head @ markov_embed[previous token]`` (``[V, r]`` by ``[r]``; 66 MB of
BF16 for DeepSeek-V4.1) is added to the position's logits and the token is
sampled from the result. The fused path (``fused_sample.py``) runs a replicated
bias GEMM and one sampling kernel per position, so every tensor-parallel rank
streams the whole head ``N`` times.

Here one cooperative kernel per rank (``sharded_markov_cute.py``, CuTe DSL)
runs the whole chain. Rank ``k`` of ``world`` owns rows
``[k V / world, (k + 1) V / world)`` of the head and keeps them in registers
for the launch, so the head is read once; the ranks agree on each position's
winner by exchanging one 64-bit ``(score, index)`` key per rank, request row
and position through a mailbox in symmetric memory. A call samples up to 8
request rows (4 with 8 ranks), the columns of the tensor-core MMA. Padding rows
(negative request-state index) are inert, as in the fused path.

Per call::

    Triton noise kernel   gumbel[i, v] of this rank's vocabulary shard (the
          |               fused path's hash), drawn on a side stream while the
          v               draft LM head runs
    chain kernel, step i  logit = cache_round(base + bf16(head . embed[p]))
                          local cache and every peer's staging <- logit
                          key = max over the shard of (logit / T + gumbel, index)
                          mailbox exchange -> winner p = draft_tokens[i]
    chain kernel, tail    cache <- the peers' shards from the local staging

The draws reproduce the fused path (same noise, rounding, temperature division
and ``torch.argmax`` tie-breaking) except for the bias reduction order before
its BF16 rounding, so a bias element can differ from cuBLAS's in its last bit.

The exchange is a cross-rank spin wait inside the draft step (and its FULL
CUDA graph): every rank must launch the kernel the same number of times.
Installation depends on configuration only and is agreed across the group, and
the per-call choice depends only on the padded batch shape, which tensor
parallelism keeps identical across ranks; a rank that diverges hangs.
"""

from functools import cache

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    _DRAFT_NOISE_SALT,
    _log1p_neg_stable,
    murmur3_uniform32,
    spec_step_seed,
)

logger = init_logger(__name__)

RANK_DIM = 256
WARPS = 16  # per CTA (sharded_markov_cute.py)
MAX_STEPS = 8
MAX_RANKS = 8
MAX_ROWS = 8  # MMA columns
MAX_VOCAB = (1 << 17) - 1  # the candidate key's index field
STAGING_OFFSET = 16384  # bytes; the mailbox records come first
CONTROL_WORDS = 256  # int32 control words (sharded_markov_cute.py)
NOISE_BLOCK = 1024
DTYPES = (torch.bfloat16, torch.float32)  # base logits and cache


@triton.jit
def _shard_noise_kernel(
    noise_ptr,
    idx_mapping_ptr,
    sample_pos_ptr,
    seeds_ptr,
    vocab_offset,
    vocab_local,
    num_steps,
    BLOCK: tl.constexpr,
):
    """Draft Gumbel noise of this rank's vocabulary shard for one (request row,
    position), as ``gumbel_noised_argmax`` draws it (fp32, drafting salt) with
    the draft step mixed into the seed. Samples are ordered (row, step)."""
    sample = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab_local
    req_state_idx = tl.load(idx_mapping_ptr + sample).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
    seed = spec_step_seed(seed, sample % num_steps)
    pos = tl.load(sample_pos_ptr + sample) - 1 + _DRAFT_NOISE_SALT
    u = murmur3_uniform32(seed, pos, vocab_offset + cols)
    gumbel_noise = -tl.log(-_log1p_neg_stable(u))
    tl.store(noise_ptr + sample * vocab_local + cols, gumbel_noise, mask=mask)


@cache
def _kernel(
    device_index: int, steps: int, base_dtype: torch.dtype, cache_dtype: torch.dtype
):
    """The CuTe DSL chain kernel compiled for a device, ``steps`` draft
    positions and the base-logits and cache dtypes. Its 512-thread CTA fits one
    per SM, and a call never launches more CTAs than the device has SMs, so the
    cooperative grid is always co-resident."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream

    from .sharded_markov_cute import ShardedMarkovSample

    types = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}
    address = Int64(0)
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            ShardedMarkovSample(steps, types[base_dtype], types[cache_dtype]),
            *[address] * 7,
            Int32(1),
            *[address] * 10,
            *[Int32(1)] * 5,
            address,
            Int32(1),
            make_fake_stream(),
        )


def _all_ranks_true(group: ProcessGroup | None, value: bool) -> bool:
    """Whether ``value`` holds on every rank of ``group`` (CPU collective)."""
    if group is None:
        return value
    flag = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def unsupported_reason(
    world: int,
    vocab_size: int,
    num_steps: int,
    embed_weight: torch.Tensor,
    head_weight: torch.Tensor,
    cache_dtype: torch.dtype,
) -> str | None:
    """Why the sharded sampler cannot serve this configuration, or None. It
    depends on configuration only, so every rank reaches the same answer."""
    if world not in (1, 2, 4, MAX_RANKS):
        return f"a tensor-parallel size of {world}"
    if torch.cuda.get_device_capability(embed_weight.device) < (10, 0):
        return "a GPU older than SM100"
    if not 1 <= num_steps <= MAX_STEPS:
        return f"{num_steps} draft positions"
    if vocab_size > MAX_VOCAB or vocab_size % (16 * world) != 0:
        return f"a vocabulary of {vocab_size} for {world} ranks"
    if cache_dtype not in DTYPES:
        return f"a {cache_dtype} draft-logits cache"
    if vocab_size // world * cache_dtype.itemsize % 16 != 0:
        return "a vocabulary shard that is not 16-byte aligned"
    for name, weight in (("embedding", embed_weight), ("head", head_weight)):
        if (
            weight.dim() != 2
            or weight.shape[0] < vocab_size
            or weight.shape[1] != RANK_DIM
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            return f"a Markov {name} that is not contiguous BF16 [V, {RANK_DIM}]"
    return None


class DSparkShardedMarkovSampler:
    """Persistent buffers, peer mappings and launchers of the sharded chain.

    ``embed_weight`` ``[V, 256]`` and the replicated ``head_weight``
    ``[V, 256]`` are BF16; this rank keeps a view of its head rows. ``group``
    is the CPU group of the tensor-parallel ranks (None for one rank).
    ``timeout_ms`` bounds every cross-rank wait with a trap (0: no bound, a
    debugging aid). Batches of up to ``max_rows`` request rows, CUDA-graph
    padding included, take the sharded chain.
    """

    def __init__(
        self,
        embed_weight: torch.Tensor,
        head_weight: torch.Tensor,
        vocab_size: int,
        num_steps: int,
        cache_dtype: torch.dtype,
        group: ProcessGroup | None,
        max_num_reqs: int,
        timeout_ms: int = 0,
    ):
        device = embed_weight.device
        self.group = group
        self.rank = dist.get_rank(group) if group is not None else 0
        self.world = dist.get_world_size(group) if group is not None else 1
        reason = unsupported_reason(
            self.world, vocab_size, num_steps, embed_weight, head_weight, cache_dtype
        )
        if reason is not None:
            raise ValueError(f"sharded Markov sampling does not support {reason}")
        self.device = device
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.cache_dtype = cache_dtype
        self.timeout_ns = int(timeout_ms) * 1_000_000
        self.vocab_local = vocab_size // self.world
        self.vocab_offset = self.rank * self.vocab_local
        self.embed_weight = embed_weight
        self.head_shard = head_weight[
            self.vocab_offset : self.vocab_offset + self.vocab_local
        ]
        self.ctas = min(
            torch.cuda.get_device_properties(device).multi_processor_count,
            triton.cdiv(self.vocab_local // 16, WARPS),
        )
        # The exchange gives each (request row, rank) pair one lane of a warp.
        self.max_rows = max(1, min(MAX_ROWS, 32 // self.world, max_num_reqs))
        self.noise = torch.empty(
            self.max_rows * num_steps,
            self.vocab_local,
            dtype=torch.float32,
            device=device,
        )
        self.control = torch.zeros(CONTROL_WORDS, dtype=torch.int32, device=device)
        # Compile before the collective set-up, so that a build failure cannot
        # leave peers waiting, and before any CUDA graph capture.
        index = device.index
        self.device_index = (
            torch.accelerator.current_device_index() if index is None else index
        )
        failure = None
        try:
            for base_dtype in DTYPES:
                _kernel(self.device_index, num_steps, base_dtype, cache_dtype)
        except Exception as error:  # any compile failure is agreed on below
            failure = error
        if not _all_ranks_true(group, failure is None):
            raise RuntimeError(f"kernel set-up failed: {failure or 'on another rank'}")
        # Mailbox records, then the staging logits of two launch parities.
        size = STAGING_OFFSET + (
            2 * self.max_rows * num_steps * vocab_size * cache_dtype.itemsize
        )
        self._peer_buffer, peer_pointers = self._allocate_peer_buffer(size)
        self.peers = torch.tensor(peer_pointers, dtype=torch.int64, device=device)
        self._noise_stream = torch.cuda.Stream(device)

    def _allocate_peer_buffer(self, size: int) -> tuple[torch.Tensor, list[int]]:
        """The zeroed mailbox and staging buffer and every rank's mapping of it
        (symmetric memory for more than one rank)."""
        if self.group is None:
            buffer = torch.zeros(size, dtype=torch.uint8, device=self.device)
            return buffer, [buffer.data_ptr()]
        import torch.distributed._symmetric_memory as symm_mem

        # The rendezvous is collective: agree on the allocation first, so that
        # a rank that failed it cannot leave its peers waiting there.
        buffer = None
        try:
            buffer = symm_mem.empty(size, dtype=torch.uint8, device=self.device)
        except RuntimeError as error:
            logger.warning("Symmetric memory allocation failed: %s", error)
        if not _all_ranks_true(self.group, buffer is not None):
            raise RuntimeError("symmetric memory allocation failed on a rank")
        assert buffer is not None
        self._handle = symm_mem.rendezvous(buffer, self.group.group_name)
        pointers = [
            self._handle.get_buffer(peer, (size,), torch.uint8, 0).data_ptr()
            for peer in range(self.world)
        ]
        buffer.zero_()
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=self.group)
        return buffer, pointers

    def supports_batch(self, num_reqs: int) -> bool:
        """Whether a batch of ``num_reqs`` request rows (padding included) can
        take the sharded chain; the same on every rank."""
        return 1 <= num_reqs <= self.max_rows

    def supports(self, num_reqs: int, base_logits: torch.Tensor) -> bool:
        """Whether a call with these ``[num_reqs, num_steps, V]`` base logits,
        or this rank's ``[num_reqs, num_steps, V / world]`` shard of them,
        takes the sharded chain. Depends on shapes and dtypes only."""
        return (
            self.supports_batch(num_reqs)
            and base_logits.dim() == 3
            and tuple(base_logits.shape[:2]) == (num_reqs, self.num_steps)
            and base_logits.shape[2] in (self.vocab_size, self.vocab_local)
            and base_logits.dtype in DTYPES
            and base_logits.stride(2) == 1
        )

    def accepts_shard(self, shard: tuple[int, int] | None) -> bool:
        """Whether, on every rank, ``shard`` (the first column and width of
        the draft LM head's local logits) is this rank's vocabulary shard. A
        CPU collective: a rank that skipped the LM head's all-gather would
        leave its peers waiting in it."""
        return _all_ranks_true(
            self.group, shard == (self.vocab_offset, self.vocab_local)
        )

    def fork_noise(
        self, idx_mapping: torch.Tensor, sample_pos: torch.Tensor, seeds: torch.Tensor
    ) -> None:
        """Draw this call's noise on a side stream after the current stream's
        work so far; ``join_noise`` must follow before ``sample``. The noise
        depends on the seeds and positions only, not on the chain."""
        num_sample = idx_mapping.numel()
        assert num_sample <= self.noise.shape[0]
        self._noise_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._noise_stream):
            grid = (num_sample, triton.cdiv(self.vocab_local, NOISE_BLOCK))
            _shard_noise_kernel[grid](
                self.noise,
                idx_mapping,
                sample_pos,
                seeds,
                self.vocab_offset,
                self.vocab_local,
                self.num_steps,
                BLOCK=NOISE_BLOCK,
            )

    def join_noise(self) -> None:
        torch.cuda.current_stream(self.device).wait_stream(self._noise_stream)

    def sample(
        self,
        base_logits: torch.Tensor,
        input_ids: torch.Tensor,
        query_per_row: int,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        logits_cache: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        """Draw every request row's draft tokens into ``draft_tokens[row]`` and
        the full-vocabulary logits of rows with a request state into
        ``logits_cache[idx_mapping]``, with the noise of ``fork_noise``.

        ``base_logits`` is ``[rows, num_steps, V]`` or this rank's
        ``[rows, num_steps, V / world]`` shard; row ``r``'s anchor token is
        ``input_ids[r * query_per_row]``; ``idx_mapping`` is the
        ``[rows * num_steps]`` request-state row (negative for padding).
        """
        rows = base_logits.shape[0]
        assert self.supports(rows, base_logits)
        assert idx_mapping.numel() == rows * self.num_steps
        assert idx_mapping.dtype == torch.int64
        assert input_ids.dtype == torch.int32 and temperature.dtype == torch.float32
        assert draft_tokens.dtype == torch.int64 and draft_tokens.stride(1) == 1
        assert logits_cache.dtype == self.cache_dtype
        assert logits_cache.size(2) >= self.vocab_size and logits_cache.stride(2) == 1
        element = logits_cache.element_size()
        assert logits_cache.data_ptr() % 16 == 0
        assert logits_cache.stride(0) * element % 16 == 0
        assert logits_cache.stride(1) * element % 16 == 0
        import cuda.bindings.driver as cuda
        from cutlass import Int32, Int64

        # The column of this rank's first vocabulary entry in base_logits.
        full_vocab = base_logits.shape[2] == self.vocab_size
        base_column = self.vocab_offset if full_vocab else 0
        kernel = _kernel(
            self.device_index, self.num_steps, base_logits.dtype, self.cache_dtype
        )
        kernel(
            Int64(base_logits.data_ptr()),
            Int64(base_logits.stride(0)),
            Int64(base_logits.stride(1)),
            Int64(base_column),
            Int64(self.embed_weight.data_ptr()),
            Int64(self.head_shard.data_ptr()),
            Int64(input_ids.data_ptr()),
            Int32(query_per_row),
            Int64(idx_mapping.data_ptr()),
            Int64(temperature.data_ptr()),
            Int64(self.noise.data_ptr()),
            Int64(logits_cache.data_ptr()),
            Int64(logits_cache.stride(0)),
            Int64(logits_cache.stride(1)),
            Int64(draft_tokens.data_ptr()),
            Int64(draft_tokens.stride(0)),
            Int64(self.peers.data_ptr()),
            Int64(self.control.data_ptr()),
            Int32(self.rank),
            Int32(self.world),
            Int32(self.vocab_local),
            Int32(rows),
            Int32(self.max_rows),
            Int64(self.timeout_ns),
            Int32(self.ctas),
            cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream),
        )


def make_sharded_sampler(
    embed_weight: torch.Tensor,
    head_weight: torch.Tensor | None,
    vocab_size: int,
    num_steps: int,
    cache_dtype: torch.dtype,
    group: ProcessGroup | None,
    max_num_reqs: int,
) -> DSparkShardedMarkovSampler | None:
    """The sampler on every rank of ``group``, or None on every rank (logged).

    Every rank must call this: the decision and the set-up are agreed with CPU
    collectives. ``head_weight`` None means this rank cannot install it.
    """
    world = dist.get_world_size(group) if group is not None else 1
    reason: str | None = "a Markov head the model does not share"
    if head_weight is not None:
        reason = unsupported_reason(
            world, vocab_size, num_steps, embed_weight, head_weight, cache_dtype
        )
    if not _all_ranks_true(group, reason is None):
        logger.info(
            "Sharded DSpark Markov sampling is off: %s.",
            reason or "another tensor-parallel rank cannot use it",
        )
        return None
    assert head_weight is not None
    sampler = None
    try:
        sampler = DSparkShardedMarkovSampler(
            embed_weight,
            head_weight,
            vocab_size,
            num_steps,
            cache_dtype,
            group,
            max_num_reqs,
        )
    except RuntimeError as error:
        logger.warning("Sharded DSpark Markov sampling set-up failed: %s", error)
    if not _all_ranks_true(group, sampler is not None):
        return None
    assert sampler is not None
    logger.info(
        "Sharded DSpark Markov sampling on %d ranks for batches of up to %d "
        "request rows.",
        sampler.world,
        sampler.max_rows,
    )
    return sampler
