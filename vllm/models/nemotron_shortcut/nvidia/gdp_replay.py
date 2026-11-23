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
collide across groups and one tracker set serves them all. Each group's
first GDP layer to run in a step commits the group's previous step before the
op runs. The commit sits in the forward, where the attention metadata carries
the previous step's accepted counts, so the bookkeeping stays inside a
captured CUDA graph.
"""

import torch

from vllm.config import VllmConfig
from vllm.platforms import current_platform

# The op's default flush threshold: a step with hist_len >= FLUSH_MIN folds
# the history into the checkpoint.
FLUSH_MIN = 9

# Set on a step's attention metadata once a layer has committed its group.
_COMMITTED = "_gdp_replay_committed"


class GdpReplayTrackers:
    """The replay cursors of every mamba block, sized when the cache binds."""

    def __init__(self) -> None:
        empty = torch.empty(0, dtype=torch.int32)
        self.base = self.hist_len = self.ring_bank = self.prev_query_len = empty
        self.seed = torch.zeros(1, dtype=torch.int64)
        """The stochastic-rounding seed of the checkpoint fold, drawn fresh
        every step (also under CUDA graph replay)."""
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

        def zeros() -> torch.Tensor:
            return torch.zeros(num_blocks, dtype=torch.int32, device=conv_state.device)

        self.base, self.hist_len = zeros(), zeros()
        self.ring_bank, self.prev_query_len = zeros(), zeros()
        self.seed = torch.zeros(1, dtype=torch.int64, device=conv_state.device)
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
        NULL_BLOCK_ID (0), a block no request owns, so writing it is harmless."""
        index = slots.long()
        for tracker in (self.base, self.hist_len, self.ring_bank, self.prev_query_len):
            tracker[index] = 0

    def commit(
        self,
        slots: torch.Tensor,
        num_accepted: torch.Tensor,
        query_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the previous step's accepted counts, record this step's query
        lengths and draw a new seed. Returns this step's `base`, `hist_len`
        and `ring_bank` per row, the op's inputs.

        `num_accepted[r]` is the number of tokens row r kept from its previous
        step (accepted drafts plus the target token); it is ignored for blocks
        with nothing to commit.
        """
        base, hist_len, ring_bank = self.read(slots)
        index = slots.long()
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

    def read(
        self, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This step's `base`, `hist_len` and `ring_bank` per row."""
        index = slots.long()
        return self.base[index], self.hist_len[index], self.ring_bank[index]


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
