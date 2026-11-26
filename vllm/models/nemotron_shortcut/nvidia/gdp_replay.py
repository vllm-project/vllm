# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bookkeeping of FlashInfer's GDP replay decode (`gdp_conv_replay`), enabled
with `--use-replayssm` on a Gated DeltaProduct model.

The op keeps, per mamba block, a checkpoint plus K / U / g history rings and
never advances its own cursors. The model keeps them in one
`GdpReplayTrackers`, shared by every GDP layer:

- `base`: conv ring cursor; wraps at the ring's row count.
- `hist_len`: tokens replayed from the checkpoint; flushes at `FLUSH_MIN`.
- `ring_bank`: which half of the banked K / U / g rings holds the history.
- `prev_query_len`: tokens the block ran in its last replay step, 0 after a
  reset (new request or prefill), which marks "nothing to commit".

All KV-cache groups draw their blocks from one pool, so block ids never
collide across groups and one tracker set serves them all. Each step, a
group's first GDP layer to run commits the group's previous step and plans
this step's op inputs in one kernel (`GdpReplayStep`); the group's other GDP
layers reuse the plan. The commit sits in the forward, where the attention
metadata carries the previous step's accepted counts, so the bookkeeping
stays inside a captured CUDA graph.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

# The op's default flush threshold: a step with hist_len >= FLUSH_MIN folds
# the history into the checkpoint.
FLUSH_MIN = 9

# Set on a step's attention metadata once a layer has committed its group,
# and the step plan that layer computed for the group's other layers.
_COMMITTED = "_gdp_replay_committed"
_STEP = "_gdp_replay_step"


@dataclass
class GdpReplayStep:
    """One KV-cache group's `gdp_conv_replay` inputs for one step, `padded`
    rows each (live rows first, then padding rows the op skips)."""

    slots: torch.Tensor  # int32 block ids; NULL_BLOCK_ID on padding rows
    valid_len: torch.Tensor  # int32 tokens per row; 0 on padding rows
    base: torch.Tensor  # int32 cursors after the commit
    hist_len: torch.Tensor
    ring_bank: torch.Tensor
    batch: int  # live rows
    tokens: int  # window width per row
    padded: int
    dst: torch.Tensor | None = None
    """Where each ragged-window token goes in the `batch * tokens`
    rectangle; None when every window is full."""


# One compile for every batch size: Triton would otherwise specialize the row
# counts on == 1 and on divisibility by 16, and an eager step whose counts fall
# in a class warmup's captured sizes missed would compile mid-serving.
@triton.jit(do_not_specialize=["slot_stride", "batch", "padded"])
def _commit_step_kernel(
    slots_ptr,
    slot_stride,
    loc_ptr,
    accepted_ptr,
    base_ptr,
    hist_ptr,
    bank_ptr,
    prev_ptr,
    out_slots_ptr,
    out_valid_ptr,
    out_base_ptr,
    out_hist_ptr,
    out_bank_ptr,
    rng_ptr,
    seed_ptr,
    batch,
    padded,
    modulus,
    FLUSH_MIN: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """`GdpReplayTrackers.commit` for live rows, the op inputs for every row,
    and the step's rounding seed, in one launch. `slots` may be a strided
    block-table column."""
    if tl.program_id(0) == 0:
        # A fresh seed each step: Philox of a per-process key and a step count
        # the kernel advances (rng = [key, count]), also under graph replay.
        key = tl.load(rng_ptr)
        count = tl.load(rng_ptr + 1)
        tl.store(rng_ptr + 1, count + 1)
        tl.store(seed_ptr, tl.randint(key, count).to(tl.int64) & 0xFFFFFFFF)
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = rows < batch
    inside = rows < padded
    slot = tl.load(slots_ptr + rows * slot_stride, mask=live, other=NULL_BLOCK_ID)
    slot = slot.to(tl.int64)
    start = tl.load(loc_ptr + rows, mask=live, other=0)
    valid = tl.load(loc_ptr + rows + 1, mask=live, other=0) - start
    accepted = tl.load(accepted_ptr + rows, mask=live, other=0)
    base = tl.load(base_ptr + slot, mask=live, other=0)
    hist = tl.load(hist_ptr + slot, mask=live, other=0)
    bank = tl.load(bank_ptr + slot, mask=live, other=0)
    committed = tl.load(prev_ptr + slot, mask=live, other=0) > 0
    flush = committed & (hist >= FLUSH_MIN)
    base = tl.where(flush, (base + hist) % modulus, base)
    bank = tl.where(flush, bank ^ 1, bank)
    hist = tl.where(committed, tl.where(flush, 0, hist) + accepted, hist)
    tl.store(base_ptr + slot, base, mask=live)
    tl.store(hist_ptr + slot, hist, mask=live)
    tl.store(bank_ptr + slot, bank, mask=live)
    tl.store(prev_ptr + slot, valid, mask=live)
    tl.store(out_slots_ptr + rows, slot.to(tl.int32), mask=inside)
    tl.store(out_valid_ptr + rows, valid, mask=inside)
    tl.store(out_base_ptr + rows, base, mask=inside)
    tl.store(out_hist_ptr + rows, hist, mask=inside)
    tl.store(out_bank_ptr + rows, bank, mask=inside)


class GdpReplayTrackers:
    """The replay cursors of every mamba block, sized when the cache binds."""

    def __init__(self) -> None:
        self.trackers = torch.empty(4, 0, dtype=torch.int32)
        """base, hist_len, ring_bank and prev_query_len, one row each, so a
        reset is one kernel."""
        self.base, self.hist_len, self.ring_bank, self.prev_query_len = self.trackers
        self.seed = torch.zeros(1, dtype=torch.int64)
        """The stochastic-rounding seed of the checkpoint fold, drawn fresh
        every step (also under CUDA graph replay)."""
        self.rng = torch.zeros(2, dtype=torch.int64)
        """The plan kernel's seed source: a per-process Philox key and the
        number of seeds drawn."""
        self.base_modulus = 0
        """The banked layout wraps `base` at the conv ring's row count."""

    def bind(self, conv_state: torch.Tensor) -> None:
        """Size the trackers for a GDP layer's conv ring `[blocks, rows,
        channels]`. Every layer binds at start-up, and again when an extensible
        cache remaps after CUDA graph capture. The first bind allocates; later
        binds keep the addresses, which captured graphs and live requests hold,
        and so must fit."""
        num_blocks, rows = conv_state.shape[:2]
        if self.base.numel() > 0:
            if (
                self.base.numel() < num_blocks
                or self.base.device != conv_state.device
                or self.base_modulus != rows
            ):
                raise RuntimeError(
                    f"GDP replay trackers hold {self.base.numel()} blocks of "
                    f"{self.base_modulus} conv rows; a rebind to {num_blocks} "
                    f"blocks of {rows} rows on {conv_state.device} would drop "
                    "them"
                )
            return

        self.trackers = torch.zeros(
            4, num_blocks, dtype=torch.int32, device=conv_state.device
        )
        self.base, self.hist_len, self.ring_bank, self.prev_query_len = self.trackers
        self.seed = torch.zeros(1, dtype=torch.int64, device=conv_state.device)
        key = int(torch.randint(0, 2**31, (1,)))
        self.rng = torch.tensor([key, 0], dtype=torch.int64, device=conv_state.device)
        self.base_modulus = rows

    @staticmethod
    def claim(metadata: object) -> bool:
        """True for the first GDP layer of a KV-cache group to run this step.

        Every layer of a group gets the same attention metadata object, and
        the runner builds a new one each step, so the first layer to see it
        commits the group and the rest only read."""
        if getattr(metadata, _COMMITTED, False):
            return False
        setattr(metadata, _COMMITTED, True)
        return True

    def reset(self, slots: torch.Tensor) -> None:
        """Zero the trackers of freshly prefilled blocks. Padding rows hold
        NULL_BLOCK_ID (0), a block no request owns, so writing it is harmless.
        `index_fill_`, not `tracker[index] = 0`: assigning a Python scalar
        copies it to the device with a blocking memcpy."""
        self.trackers.index_fill_(1, slots.long(), 0)

    def commit(
        self,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        query_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the previous step's accepted counts, record this step's query
        lengths and draw a new seed. Returns this step's `base`, `hist_len`
        and `ring_bank` per row, the op's inputs. The PyTorch reference of
        `_commit_step_kernel`, and its CPU path.

        `num_accepted[r]` is the number of tokens row r kept from its previous
        step (accepted drafts plus the target token); it is ignored for blocks
        with nothing to commit.
        """
        index = slots.long()
        base = self.base[index]
        hist_len = self.hist_len[index]
        ring_bank = self.ring_bank[index]
        committed = self.prev_query_len[index] > 0

        flush = committed & (hist_len >= FLUSH_MIN)
        base = torch.where(flush, (base + hist_len) % self.base_modulus, base)
        ring_bank = ring_bank ^ flush.to(ring_bank.dtype)
        hist_len = torch.where(
            committed,
            torch.where(flush, 0, hist_len) + num_accepted.to(hist_len.dtype),
            hist_len,
        )

        self.base[index] = base
        self.hist_len[index] = hist_len
        self.ring_bank[index] = ring_bank
        self.prev_query_len[index] = query_lens.to(torch.int32)
        torch.randint(0, 2**62, (1,), out=self.seed)
        return base, hist_len, ring_bank

    def commit_step(
        self,
        metadata: object,
        slots: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_accepted: torch.Tensor,
        tokens: int,
        padded: int,
        num_rows: int,
    ) -> GdpReplayStep:
        """Commit the previous step for a group's `slots` and plan this step's
        op inputs, `padded` rows of `tokens`-wide windows over `num_rows`
        scheduled tokens. Called by the group's first GDP layer
        (`claim`); the plan is kept on `metadata` for the other layers."""
        batch = slots.size(0)
        loc = query_start_loc[: batch + 1].contiguous()
        num_accepted = num_accepted.contiguous()
        device = slots.device
        # slots, valid_len, base, hist_len, ring_bank: rows of one allocation,
        # each starting 16-B aligned as gdp_conv_replay needs.
        row = -(-padded // 4) * 4
        out = list(torch.empty(5, row, dtype=torch.int32, device=device)[:, :padded])
        if device.type == "cuda":
            block = 128
            # The kernel reads `slots` with its stride: a block-table column.
            _commit_step_kernel[(triton.cdiv(padded, block),)](
                slots,
                slots.stride(0),
                loc,
                num_accepted,
                self.base,
                self.hist_len,
                self.ring_bank,
                self.prev_query_len,
                *out,
                self.rng,
                self.seed,
                batch,
                padded,
                self.base_modulus,
                FLUSH_MIN=FLUSH_MIN,
                NULL_BLOCK_ID=NULL_BLOCK_ID,
                BLOCK=block,
            )
        else:
            slots = slots.contiguous()
            valid = (loc[1:] - loc[:-1]).to(torch.int32)
            committed = self.commit(slots, num_accepted, valid)
            for tensor, live in zip(out, (slots, valid, *committed), strict=True):
                tensor.zero_()
                tensor[:batch] = live
            out[0][batch:] = NULL_BLOCK_ID
        dst = None
        if num_rows != batch * tokens:
            # `output_size` spares the step a device-to-host sync.
            repeats = (loc[1:] - loc[:-1]).long()
            requests = torch.repeat_interleave(
                torch.arange(batch, device=device), repeats, output_size=num_rows
            )
            offsets = torch.arange(num_rows, device=device) - torch.repeat_interleave(
                loc[:batch].long(), repeats, output_size=num_rows
            )
            dst = requests * tokens + offsets
        step = GdpReplayStep(
            slots=out[0],
            valid_len=out[1],
            base=out[2],
            hist_len=out[3],
            ring_bank=out[4],
            batch=batch,
            tokens=tokens,
            padded=padded,
            dst=dst,
        )
        setattr(metadata, _STEP, step)
        return step

    @staticmethod
    def step_of(metadata: object) -> GdpReplayStep:
        """The plan the group's first GDP layer computed this step."""
        step = getattr(metadata, _STEP, None)
        assert step is not None, "the group's first GDP layer plans each step"
        return step


def is_prefix_view(
    capacity: Sequence[torch.Tensor], views: Sequence[torch.Tensor]
) -> bool:
    """Whether each of `views` is a leading run of blocks of the matching
    `capacity` tensor: the same start, block shape and strides, and no more
    blocks."""
    return len(capacity) == len(views) and all(
        c.data_ptr() == v.data_ptr()
        and c.shape[1:] == v.shape[1:]
        and c.stride() == v.stride()
        and c.shape[0] >= v.shape[0]
        for c, v in zip(capacity, views)
    )


def validate_gdp_replay(vllm_config: VllmConfig) -> None:
    """Reject configurations FlashInfer's `gdp_conv_replay` cannot run."""
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    parallel_config = vllm_config.parallel_config
    config = model_config.hf_text_config
    if not current_platform.is_device_capability_family(100):
        raise ValueError("GDP replay runs on SM100-class GPUs only")
    geometry = (
        config.gdp_key_head_dim,
        config.gdp_head_dim,
        config.gdp_num_householder,
        config.gdp_conv_kernel,
    )
    if geometry != (128, 64, 3, 4):
        raise ValueError(
            "GDP replay supports key head dim 128, value head dim 64, 3 "
            "householder steps and conv width 4; got (key, value, "
            f"householder, conv) = {geometry}"
        )
    tp_size = parallel_config.tensor_parallel_size
    heads_per_rank, remainder = divmod(config.gdp_num_heads, tp_size)
    if remainder or heads_per_rank % 2:
        raise ValueError(
            "GDP replay needs an even number of GDP heads per TP rank; "
            f"{config.gdp_num_heads} heads at TP {tp_size}"
        )
    # Replay exists to roll back rejected drafts. Without spec decode it only
    # adds state, and a decode row could be routed to prefill.
    if not 1 <= vllm_config.num_speculative_tokens <= 7:
        raise ValueError(
            "GDP replay needs speculative decoding with 1 to 7 speculative "
            "tokens (at most 8 tokens per step); got "
            f"{vllm_config.num_speculative_tokens}"
        )
    if cache_config.enable_prefix_caching:
        raise ValueError(
            "GDP replay does not support prefix caching yet; "
            "use --no-enable-prefix-caching"
        )
    # The op takes BF16 activations and a BF16 conv ring.
    if model_config.dtype != torch.bfloat16 or cache_config.mamba_cache_dtype not in (
        "auto",
        "bfloat16",
    ):
        raise ValueError(
            "GDP replay needs a bfloat16 model and conv cache; got model dtype "
            f"{model_config.dtype} and --mamba-cache-dtype "
            f"{cache_config.mamba_cache_dtype}"
        )
    # The trackers commit each step from the previous step's accepted counts,
    # inside one forward over the whole batch.
    if parallel_config.pipeline_parallel_size > 1:
        raise ValueError("GDP replay currently requires pipeline_parallel_size=1")
    if parallel_config.use_ubatching:
        raise ValueError("GDP replay does not support microbatching")
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is not None
        and speculative_config.enable_adaptive_verification
    ):
        raise ValueError("GDP replay does not support adaptive verification")
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance:
        # A transferred block arrives without a prefill here, so nothing would
        # reset its replay trackers.
        raise ValueError(
            "GDP replay does not support KV connectors (P/D disaggregation, "
            "KV cache offload) yet"
        )
