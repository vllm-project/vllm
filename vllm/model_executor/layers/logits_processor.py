# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

from collections.abc import Callable
from functools import cache
from typing import Any

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.flashinfer import has_flashinfer
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

indexed_argmax_triton: Any
reduce_global_argmax_triton: Any

if HAS_TRITON:
    from vllm.model_executor.layers.argmax_triton import (
        indexed_argmax_triton as _indexed_argmax_triton,
    )
    from vllm.model_executor.layers.argmax_triton import (
        reduce_global_argmax_triton as _reduce_global_argmax_triton,
    )
    from vllm.model_executor.layers.presence_penalty_triton import (
        apply_presence_penalty_from_counts,
    )

    indexed_argmax_triton = _indexed_argmax_triton
    reduce_global_argmax_triton = _reduce_global_argmax_triton
    _apply_presence_penalty_from_counts = apply_presence_penalty_from_counts
else:
    indexed_argmax_triton = None
    reduce_global_argmax_triton = None
    _apply_presence_penalty_from_counts = None

logger = init_logger(__name__)

# Keep Gumbel noise bounded when the unrestricted path has a large vocabulary.
# The full local logits projection is unavoidable, but a second full-sized
# noise/scores matrix is not.
_FULL_SAMPLE_BLOCK_SIZE = 8192
_MAX_FLOAT32_TOKEN_ID = 1 << 24


def _stable_topk(
    values: torch.Tensor,
    k: int,
    token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k with value-descending, token-id-ascending tie handling."""
    if k <= 0 or k > values.shape[-1]:
        raise ValueError(f"k must be in [1, {values.shape[-1]}], got {k}")
    rank_values = torch.where(
        torch.isnan(values), torch.full_like(values, -float("inf")), values
    )
    if token_ids is None:
        order = torch.argsort(rank_values, dim=-1, descending=True, stable=True)
    else:
        if token_ids.shape != values.shape:
            raise ValueError("token_ids must have the same shape as values")
        id_order = torch.argsort(token_ids, dim=-1, stable=True)
        id_sorted_values = rank_values.gather(-1, id_order)
        value_order = torch.argsort(
            id_sorted_values, dim=-1, descending=True, stable=True
        )
        order = id_order.gather(-1, value_order)
    order = order[..., :k]
    return rank_values.gather(-1, order), order


@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """FlashInfer's radix top-k, or None for torch.topk.

    The top-k spans the vocabulary, where the radix kernel is about twice
    torch.topk.
    """
    if not current_platform.is_cuda():
        return None
    if not has_flashinfer():
        logger.info_once(
            "flashinfer is unavailable; vocab-parallel top-k uses torch.topk, "
            "at roughly half the speed."
        )
        return None
    from flashinfer import top_k

    return top_k


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        # ``torch.topk`` does not define the order of equal values.  The
        # stable fallback keeps local boundary candidates deterministic so a
        # later TP merge cannot depend on kernel launch details.
        return _stable_topk(scores, k)
    return impl(scores, k, sorted=True, deterministic=True)


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()
        # Dtype of the lm_head projection. Defaults to the model dtype; an
        # fp32 head (via `--hf-overrides '{"head_dtype": "float32"}'`) is
        # required for RL training-inference consistency.
        model_config = get_current_vllm_config().model_config
        self.head_dtype = model_config.head_dtype if model_config is not None else None
        # Callers that must keep one execution path for a CUDA graph (for
        # example a draft model) can disable the compact head for the whole
        # call.  The V2 runner uses the row mask below for mixed batches.
        self.hybrid_lm_head_enabled = True
        # The V2 runner may install a one-call row mask for mixed prefill and
        # decode batches.  ``True`` rows may use the compact approximate head;
        # ``False`` rows stay on the exact full-vocabulary path.
        self.hybrid_lm_head_row_mask: torch.Tensor | None = None

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(
                hidden_states, lm_head, embedding_bias, skip_gather
            )
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project hidden states through the lm_head, honoring head_dtype."""
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )

        if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError(
                "A head_dtype different from the model dtype is only "
                "supported for an unquantized lm_head."
            )
        if (
            self.head_dtype == torch.float32
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
        ):
            # Accumulate the projection directly into fp32. This avoids
            # materializing an fp32 copy of the lm_head weight on every step,
            # unlike casting both operands. `torch.mm(out_dtype=...)` only
            # supports fp32 output for fp16/bf16 inputs, and is only
            # implemented for CUDA and ROCm (the latter via the non-Lt GEMM
            # path); other platforms fall back to the cast path below.
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.mm(flat, lm_head.weight.t(), out_dtype=self.head_dtype)
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(self.head_dtype)
            return logits.reshape(*hidden_states.shape[:-1], -1)
        return F.linear(
            hidden_states.to(self.head_dtype),
            lm_head.weight.to(self.head_dtype),
            embedding_bias.to(self.head_dtype) if embedding_bias is not None else None,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if skip_gather:
            return logits

        # Gather logits for TP
        if lm_head.tp_size > 1:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_mask = self._get_hybrid_lm_head_row_mask(hidden_states)
        if row_mask is not None and not bool(row_mask.all()):
            result = torch.empty(
                hidden_states.shape[0],
                dtype=torch.int64,
                device=hidden_states.device,
            )
            enabled_rows = torch.nonzero(row_mask, as_tuple=True)[0]
            if enabled_rows.numel() > 0:
                result[enabled_rows] = self._get_top_tokens_single(
                    lm_head,
                    hidden_states[enabled_rows].contiguous(),
                    embedding_bias=embedding_bias,
                    _hybrid_enabled=True,
                )
            disabled_rows = torch.nonzero(~row_mask, as_tuple=True)[0]
            if disabled_rows.numel() > 0:
                result[disabled_rows] = self._get_top_tokens_single(
                    lm_head,
                    hidden_states[disabled_rows].contiguous(),
                    embedding_bias=embedding_bias,
                    _hybrid_enabled=False,
                )
            return result
        return self._get_top_tokens_single(
            lm_head,
            hidden_states,
            embedding_bias=embedding_bias,
        )

    def _get_top_tokens_single(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        *,
        _hybrid_enabled: bool | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size). The optional hybrid state uses an approximate
        NVFP4 candidate search and is only selected for explicitly compatible
        model configurations.
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = lm_head.tp_size

        hybrid_state = getattr(lm_head, "_hybrid_nvfp4_lm_head_state", None)
        shard_indices = lm_head.shard_indices
        num_pad = shard_indices.num_org_vocab_padding
        has_added_vocab = (
            getattr(lm_head, "num_added_embeddings", 0) > 0
            or getattr(shard_indices, "num_added_elements", 0) > 0
        )
        local_vocab_size = getattr(shard_indices, "num_elements_padded", None)
        if local_vocab_size is None:
            local_vocab_size = getattr(
                lm_head,
                "num_embeddings_per_partition",
                self.vocab_size,
            )
        assert local_vocab_size is not None
        active_vocab_size = local_vocab_size - num_pad
        if has_added_vocab:
            # Base and added vocabularies have separate padded regions.  Mask
            # both regions and map the surviving local positions explicitly;
            # treating the whole shard as ``local_id + org_vocab_start`` would
            # return incorrect ids for LoRA-added embeddings.
            logits = self._apply_head(lm_head, hidden_states, embedding_bias)
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale

            local_ids = torch.arange(logits.shape[-1], device=logits.device)
            org_local_start = (
                shard_indices.org_vocab_start_index
                - shard_indices.padded_org_vocab_start_index
            )
            org_local_end = org_local_start + shard_indices.num_org_elements
            valid_mask = (local_ids >= org_local_start) & (local_ids < org_local_end)
            num_org_padded = shard_indices.num_org_elements_padded
            if (
                shard_indices.added_vocab_end_index
                > shard_indices.added_vocab_start_index
            ):
                added_local_start = num_org_padded + (
                    shard_indices.added_vocab_start_index
                    - shard_indices.padded_added_vocab_start_index
                )
                added_local_end = added_local_start + shard_indices.num_added_elements
                valid_mask |= (local_ids >= added_local_start) & (
                    local_ids < added_local_end
                )
            global_ids = torch.where(
                local_ids < num_org_padded,
                shard_indices.padded_org_vocab_start_index + local_ids,
                shard_indices.padded_added_vocab_start_index
                + local_ids
                - num_org_padded,
            )
            finite_logits = torch.where(
                valid_mask & (logits == logits),
                logits,
                torch.full_like(logits, -float("inf")),
            )
            local_max_vals = finite_logits.max(dim=-1).values
            tie_break_ids = torch.where(
                valid_mask & (finite_logits == local_max_vals.unsqueeze(-1)),
                global_ids,
                torch.full_like(global_ids, torch.iinfo(torch.int64).max),
            )
            global_indices = tie_break_ids.min(dim=-1).values
            return self.reduce_local_argmax(
                local_max_vals,
                global_indices,
                tp_size=tp_size,
            )
        if (
            hybrid_state is not None
            and self.soft_cap is None
            and self.scale > 0.0
            and not envs.VLLM_BATCH_INVARIANT
            and not has_added_vocab
            and self._is_contiguous_org_shard(lm_head)
            and self.hybrid_lm_head_enabled
            and _hybrid_enabled is not False
            and not envs.VLLM_COMPUTE_NANS_IN_LOGITS
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
            and hybrid_state.can_use(
                hidden_states,
                bf16_weight=lm_head.weight,
                active_vocab_size=active_vocab_size,
                top_k=1,
            )
        ):
            coarse_logits = hybrid_state.coarse_logits(
                hidden_states,
                embedding_bias,
            )
            if num_pad > 0:
                coarse_logits[..., -num_pad:] = -float("inf")
            candidate_indices = self._select_hybrid_candidates(
                hybrid_state,
                coarse_logits,
                top_k=1,
            )
            exact_logits = hybrid_state.refine_logits(
                hidden_states,
                lm_head.weight,
                candidate_indices,
                embedding_bias,
            )
            if (
                indexed_argmax_triton is not None
                and exact_logits.is_cuda
                and 0 < exact_logits.shape[-1] <= 1024
            ):
                local_max_vals, global_indices = indexed_argmax_triton(
                    exact_logits,
                    candidate_indices,
                    index_offset=shard_indices.org_vocab_start_index,
                )
            else:
                # Match the Triton path's NaN handling and lower-token-id
                # tie break when the indexed kernel is unavailable.
                finite_logits = torch.where(
                    exact_logits == exact_logits,
                    exact_logits,
                    torch.full_like(exact_logits, -float("inf")),
                )
                local_max_vals = finite_logits.max(dim=-1).values
                candidate_indices_i64 = candidate_indices.to(torch.int64)
                tie_break_ids = torch.where(
                    finite_logits == local_max_vals.unsqueeze(-1),
                    candidate_indices_i64,
                    torch.full_like(
                        candidate_indices_i64, torch.iinfo(torch.int64).max
                    ),
                )
                local_max_indices = tie_break_ids.min(dim=-1).values
                global_indices = local_max_indices + shard_indices.org_vocab_start_index
            if self.scale != 1.0:
                local_max_vals = local_max_vals * self.scale
            return self.reduce_local_argmax(
                local_max_vals,
                global_indices,
                tp_size=tp_size,
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        local_max_vals, local_max_indices = logits.max(dim=-1)
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start
        return self.reduce_local_argmax(
            local_max_vals,
            global_indices,
            tp_size=tp_size,
        )

    def _get_hybrid_lm_head_row_mask(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a valid one-call compact-path row mask, if installed."""
        mask = self.hybrid_lm_head_row_mask
        if mask is None or mask.ndim != 1 or mask.shape[0] != hidden_states.shape[0]:
            return None
        if mask.device != hidden_states.device:
            mask = mask.to(hidden_states.device, non_blocking=True)
        return mask.to(dtype=torch.bool)

    def reduce_local_argmax(
        self,
        local_max_vals: torch.Tensor,
        global_indices: torch.Tensor,
        *,
        tp_size: int,
    ) -> torch.Tensor:
        if tp_size == 1:
            return global_indices.to(torch.int64)

        # Token ids are normally packed with scores for one compact
        # all-gather.  FP32 cannot represent every integer above 2**24, so
        # use separate typed gathers for unusually large vocabularies rather
        # than silently changing the argmax tie-break key.
        if self.vocab_size >= _MAX_FLOAT32_TOKEN_ID:
            gathered_values = tensor_model_parallel_all_gather(
                local_max_vals.float(), dim=-1
            )
            gathered_ids = tensor_model_parallel_all_gather(
                global_indices.to(torch.int64), dim=-1
            )
            return self._reduce_global_argmax_values_ids(
                gathered_values, gathered_ids, tp_size
            )

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        return self._reduce_global_argmax_pairs(gathered, tp_size)

    @staticmethod
    def _reduce_global_argmax_pairs(
        gathered: torch.Tensor,
        tp_size: int,
    ) -> torch.Tensor:
        """Reduce ``(score, token_id)`` pairs with NaN-safe tie handling."""
        if reduce_global_argmax_triton is not None and gathered.is_cuda:
            return reduce_global_argmax_triton(gathered, tp_size=tp_size).to(
                torch.int64
            )
        gathered = gathered.view(gathered.shape[0], tp_size, 2)
        return LogitsProcessor._reduce_global_argmax_values_ids(
            gathered[:, :, 0], gathered[:, :, 1], tp_size
        )

    @staticmethod
    def _reduce_global_argmax_values_ids(
        gathered_values: torch.Tensor,
        gathered_ids: torch.Tensor,
        tp_size: int,
    ) -> torch.Tensor:
        """Reduce separately typed ``(value, token_id)`` gathers."""
        gathered_ids = gathered_ids.to(torch.int64)
        values = torch.where(
            gathered_values == gathered_values,
            gathered_values,
            torch.full_like(gathered_values, -float("inf")),
        )
        if values.ndim == 1:
            values = values.view(-1, tp_size)
            gathered_ids = gathered_ids.view(-1, tp_size)
        max_values = values.max(dim=-1, keepdim=True).values
        tie_break_ids = torch.where(
            values == max_values,
            gathered_ids,
            torch.full_like(gathered_ids, torch.iinfo(torch.int64).max),
        )
        return tie_break_ids.min(dim=-1).values.to(torch.int64)

    def sample_full_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        temperature: float,
        embedding_bias: torch.Tensor | None = None,
        expanded_idx_mapping: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        temperature_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample an unrestricted distribution without gathering full logits.

        Gumbel-max is evaluated independently on each TP-local shard. Only
        one winning ``(score, global_token_id)`` pair per row is gathered and
        the sampled id is broadcast, preserving one-token-per-request
        semantics while avoiding the ``[batch, vocab]`` TP all-gather.
        """
        if hidden_states.ndim != 2:
            raise ValueError("sample_full_tokens requires rank-2 hidden states")
        if temperature <= 0.0:
            raise ValueError(
                "sample_full_tokens requires positive temperature; "
                f"got {temperature}"
            )
        if not self._is_contiguous_org_shard(lm_head):
            raise ValueError(
                "full-distribution sampling requires a contiguous original-vocab "
                "shard"
            )

        shard_indices = lm_head.shard_indices
        local_vocab_size = getattr(
            shard_indices,
            "num_elements_padded",
            getattr(lm_head, "num_embeddings_per_partition", self.vocab_size),
        )
        active_vocab_size = local_vocab_size - shard_indices.num_org_vocab_padding
        logits = self._apply_head(lm_head, hidden_states, embedding_bias).to(
            torch.float32
        )
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits.mul_(self.scale)
        logits.div_(temperature)
        self._mask_invalid_shard_logits(logits, active_vocab_size)

        if (
            (expanded_idx_mapping is None) != (seeds is None)
            or (expanded_idx_mapping is None) != (pos is None)
            or (expanded_idx_mapping is None) != (temperature_state is None)
        ):
            raise ValueError(
                "expanded_idx_mapping, seeds, pos, and temperature_state must be "
                "provided together"
            )

        if (
            expanded_idx_mapping is not None
            and seeds is not None
            and pos is not None
            and logits.is_cuda
            and HAS_TRITON
        ):
            if temperature_state is None:
                raise ValueError(
                    "temperature_state is required with keyed Gumbel metadata"
                )
            # Use the same keyed per-request Gumbel stream as the standard V2
            # sampler. The local token offset makes TP shards use the global
            # token id as the random key, so a TP=1 and TP>1 run agree.
            gumbel_result = gumbel_sample(
                logits,
                expanded_idx_mapping,
                temperature_state,
                seeds,
                pos,
                apply_temperature=False,
                use_fp64=False,
                token_key_offset=shard_indices.org_vocab_start_index,
                return_scores=True,
            )
            assert isinstance(gumbel_result, tuple)
            local_indices, local_scores = gumbel_result
        else:
            # Direct callers and CPU tests do not carry V2 sampling state. Keep
            # the bounded fallback for those users; runner fast paths always
            # pass keyed state on CUDA.
            for start in range(0, logits.shape[-1], _FULL_SAMPLE_BLOCK_SIZE):
                end = min(start + _FULL_SAMPLE_BLOCK_SIZE, logits.shape[-1])
                block = logits[..., start:end]
                noise = torch.empty_like(block)
                noise.exponential_()
                block.sub_(noise.log_())
                block.masked_fill_(torch.isnan(block), -float("inf"))
            local_scores, local_indices = logits.max(dim=-1)
        global_indices = local_indices.to(torch.int64) + (
            shard_indices.org_vocab_start_index
        )
        tp_size = lm_head.tp_size
        if tp_size == 1:
            return global_indices

        if self.vocab_size >= _MAX_FLOAT32_TOKEN_ID:
            gathered_scores = tensor_model_parallel_gather(
                local_scores, dst=0, dim=-1
            )
            gathered_ids = tensor_model_parallel_gather(
                global_indices, dst=0, dim=-1
            )
        else:
            local_pair = torch.stack(
                [local_scores, global_indices.to(torch.float32)], dim=-1
            )
            gathered_scores = tensor_model_parallel_gather(
                local_pair, dst=0, dim=-1
            )
            gathered_ids = None
        tp_group = get_tp_group()
        if tp_group.rank_in_group == 0:
            assert gathered_scores is not None
            if self.vocab_size >= _MAX_FLOAT32_TOKEN_ID:
                assert gathered_ids is not None
                sampled = self._reduce_global_argmax_values_ids(
                    gathered_scores, gathered_ids, tp_size
                )
            else:
                sampled = self._reduce_global_argmax_pairs(gathered_scores, tp_size)
        else:
            sampled = torch.empty(
                (hidden_states.shape[0],),
                dtype=torch.int64,
                device=hidden_states.device,
            )
        tp_group.broadcast(sampled, src=0)
        return sampled

    def get_top_k_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocab-parallel top-k without all-gathering full logits.

        The `get_top_tokens` reduction widened from one token to k, returning
        the values as well as the global ids. Communication is
        O(batch * 2k * tp_size) rather than O(batch * vocab_size).

        Scale and soft cap are applied to the k selected values rather than
        the whole vocabulary; both are monotonic, so the selection is the same
        and only k entries are touched.
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        values, ids = _topk(logits, k)
        # Convert shard-local indices to global vocab indices.
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index

        if lm_head.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, k)
            ids = ids.gather(-1, selected)

        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values

    @staticmethod
    def _is_contiguous_org_shard(lm_head: VocabParallelEmbedding) -> bool:
        """Whether local logits map to one contiguous original-vocab shard."""
        shard_indices = lm_head.shard_indices
        # Older shard-index implementations expose only one of these fields.
        # Treat either real or padded added-vocab entries as incompatible with
        # the contiguous original-vocabulary fast path.
        return (
            getattr(shard_indices, "num_added_elements_padded", 0) == 0
            and getattr(shard_indices, "num_added_elements", 0) == 0
        )

    @staticmethod
    def _mask_invalid_shard_logits(
        logits: torch.Tensor,
        active_vocab_size: int,
    ) -> None:
        if active_vocab_size < logits.shape[-1]:
            logits[..., active_vocab_size:] = -float("inf")

    @staticmethod
    def _select_hybrid_candidates(
        hybrid_state: Any,
        coarse_logits: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Select a widened candidate set while accepting legacy test states."""
        try:
            return hybrid_state.select_candidates(coarse_logits, top_k=top_k)
        except TypeError as exc:
            # Older state objects did not expose a top_k keyword.  Keep this
            # compatibility fallback narrow so unrelated implementation errors
            # are not silently hidden.
            if "top_k" not in str(exc):
                raise
            return hybrid_state.select_candidates(coarse_logits)

    @staticmethod
    def _apply_presence_penalty_from_counts(
        logits: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_token_counts: torch.Tensor,
        presence_request_indices: torch.Tensor,
        *,
        shard_indices: Any,
        local_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Apply V2 presence penalties without constructing a dense mask on CUDA."""
        if logits.numel() == 0:
            return
        if logits.ndim != 2:
            raise ValueError("logits must be rank 2")
        if output_token_counts is None or output_token_counts.ndim != 2:
            raise ValueError("output_token_counts must be rank 2")
        if presence_request_indices.shape != (logits.shape[0],):
            raise ValueError("presence_request_indices must have one entry per row")
        if presence_penalties.shape != (logits.shape[0],):
            raise ValueError("presence_penalties must have one entry per row")
        if local_token_ids is not None and local_token_ids.shape != logits.shape:
            raise ValueError("local_token_ids must match logits")

        num_org_elements = getattr(
            shard_indices,
            "num_org_elements",
            logits.shape[-1] - shard_indices.num_org_vocab_padding,
        )
        num_org_elements_padded = getattr(
            shard_indices,
            "num_org_elements_padded",
            num_org_elements + shard_indices.num_org_vocab_padding,
        )
        added_vocab_start = getattr(
            shard_indices,
            "added_vocab_start_index",
            getattr(shard_indices, "org_vocab_end_index", num_org_elements),
        )
        num_added_elements = getattr(shard_indices, "num_added_elements", 0)

        if (
            _apply_presence_penalty_from_counts is not None
            and logits.is_cuda
            and logits.is_contiguous()
            and output_token_counts.is_cuda
            and presence_request_indices.is_cuda
            and presence_penalties.is_cuda
            and (local_token_ids is None or local_token_ids.is_cuda)
        ):
            _apply_presence_penalty_from_counts(
                logits,
                output_token_counts,
                presence_request_indices,
                presence_penalties,
                org_vocab_start=shard_indices.org_vocab_start_index,
                num_org_elements=num_org_elements,
                num_org_elements_padded=num_org_elements_padded,
                added_vocab_start=added_vocab_start,
                num_added_elements=num_added_elements,
                local_token_ids=local_token_ids,
            )
            return

        # CPU and non-Triton fallback. This path is intentionally simple; the
        # CUDA implementation above is the memory-sensitive one.
        if not logits.is_contiguous():
            raise ValueError("logits must be contiguous")
        if local_token_ids is None:
            local_ids = torch.arange(
                logits.shape[-1], dtype=torch.int64, device=logits.device
            ).expand_as(logits)
        else:
            local_ids = local_token_ids.to(torch.int64)
        is_org = local_ids < num_org_elements
        added_offsets = local_ids - num_org_elements_padded
        is_added = (added_offsets >= 0) & (added_offsets < num_added_elements)
        global_ids = torch.where(
            is_org,
            local_ids + shard_indices.org_vocab_start_index,
            added_offsets + added_vocab_start,
        )
        valid = (is_org | is_added) & (global_ids >= 0)
        valid &= global_ids < output_token_counts.shape[-1]
        safe_ids = global_ids.clamp(0, output_token_counts.shape[-1] - 1)
        request_indices = presence_request_indices.to(torch.int64).unsqueeze(1)
        counts = output_token_counts[request_indices, safe_ids]
        penalty = presence_penalties.to(logits.dtype).unsqueeze(1)
        logits.sub_(torch.where(valid & (counts > 0), penalty, 0.0))

    @staticmethod
    def _apply_presence_penalty_from_token_ids(
        logits: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_token_ids: torch.Tensor,
        *,
        shard_indices: Any,
        local_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Apply presence penalties from a compact per-row token-id table.

        This is used by the legacy runner, which does not maintain V2's dense
        output-count table.  The full local-shard path uses ``scatter_add_``
        over unique token ids, while the refined matrix only compares against
        its at-most-1024 candidate columns.  Neither path allocates a
        ``[batch, vocab]`` penalty mask.
        """
        if logits.numel() == 0:
            return
        if logits.ndim != 2 or not logits.is_contiguous():
            raise ValueError("logits must be a contiguous rank-2 tensor")
        if output_token_ids.ndim != 2:
            raise ValueError("output_token_ids must be rank 2")
        if output_token_ids.shape[0] != logits.shape[0]:
            raise ValueError("output_token_ids must have one row per logits row")
        if presence_penalties.shape != (logits.shape[0],):
            raise ValueError("presence_penalties must have one entry per logits row")
        if local_token_ids is not None and local_token_ids.shape != logits.shape:
            raise ValueError("local_token_ids must match logits")
        if output_token_ids.shape[1] == 0:
            return

        output_ids = output_token_ids.to(torch.int64)
        org_vocab_start = shard_indices.org_vocab_start_index
        if local_token_ids is None:
            num_pad = shard_indices.num_org_vocab_padding
            active_vocab_size = logits.shape[-1] - num_pad
            local_ids = output_ids - org_vocab_start
            valid = (local_ids >= 0) & (local_ids < active_vocab_size)
            safe_ids = local_ids.clamp(0, max(active_vocab_size - 1, 0))
            delta = -presence_penalties.to(logits.dtype).unsqueeze(1) * valid.to(
                logits.dtype
            )
            # The caller supplies unique output ids, so scatter-add applies
            # each presence penalty exactly once per row.
            logits.scatter_add_(1, safe_ids, delta)
            return

        candidate_global_ids = local_token_ids.to(torch.int64) + org_vocab_start
        valid_output = output_ids >= 0
        sentinel = torch.iinfo(torch.int64).max
        sorted_output_ids = torch.where(
            valid_output,
            output_ids,
            torch.full_like(output_ids, sentinel),
        ).sort(dim=-1).values
        positions = torch.searchsorted(sorted_output_ids, candidate_global_ids)
        safe_positions = positions.clamp(max=sorted_output_ids.shape[-1] - 1)
        matched = sorted_output_ids.gather(1, safe_positions) == candidate_global_ids
        logits.sub_(
            matched.to(logits.dtype)
            * presence_penalties.to(logits.dtype).unsqueeze(1)
        )

    def _get_local_topk(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        *,
        top_k: int,
        temperature: float,
        presence_penalties: torch.Tensor | None = None,
        output_token_ids: torch.Tensor | None = None,
        output_token_counts: torch.Tensor | None = None,
        presence_request_indices: torch.Tensor | None = None,
        embedding_bias: torch.Tensor | None = None,
        _hybrid_enabled: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compute local top-k values and ids, optionally via hybrid refine."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if temperature <= 0.0:
            raise ValueError(
                "_get_local_topk requires positive temperature; "
                f"got {temperature}"
            )
        if presence_penalties is not None:
            if (
                output_token_counts is None
                and output_token_ids is None
            ) or (
                output_token_counts is not None
                and presence_request_indices is None
                and output_token_ids is None
            ):
                raise ValueError(
                    "presence penalties require either output token ids or "
                    "output token counts with request indices"
                )
            if (
                output_token_counts is not None
                and presence_request_indices is not None
            ):
                assert presence_request_indices is not None
            if output_token_ids is not None:
                if output_token_ids.ndim != 2:
                    raise ValueError("output_token_ids must be rank 2")
                if output_token_ids.shape[0] != hidden_states.shape[0]:
                    raise ValueError(
                        "output_token_ids must have one row per hidden-state row"
                    )

        shard_indices = lm_head.shard_indices
        if not self._is_contiguous_org_shard(lm_head):
            raise ValueError(
                "compact top-k sampling requires a contiguous original-vocab shard"
            )
        local_vocab_size = getattr(
            shard_indices,
            "num_elements_padded",
            getattr(lm_head, "num_embeddings_per_partition", self.vocab_size),
        )
        num_pad = shard_indices.num_org_vocab_padding
        active_vocab_size = local_vocab_size - num_pad
        local_top_k = min(top_k, active_vocab_size)
        if local_top_k <= 0:
            raise ValueError("the local vocabulary shard is empty")

        row_mask = (
            None
            if _hybrid_enabled is not None
            else self._get_hybrid_lm_head_row_mask(hidden_states)
        )
        if row_mask is not None and not bool(row_mask.all()):
            # Split decode rows from prompt-tail rows. The latter use the exact
            # full head while preserving one compact result tensor for the
            # subsequent TP gather. Presence metadata is row-aligned for
            # penalties; the dense count table itself remains shared.
            local_values = torch.empty(
                (hidden_states.shape[0], local_top_k),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            local_indices = torch.empty(
                (hidden_states.shape[0], local_top_k),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            enabled_rows = torch.nonzero(row_mask, as_tuple=True)[0]
            disabled_rows = torch.nonzero(~row_mask, as_tuple=True)[0]
            if enabled_rows.numel() > 0:
                enabled_values, enabled_indices, _ = self._get_local_topk(
                    lm_head,
                    hidden_states[enabled_rows].contiguous(),
                    top_k=top_k,
                    temperature=temperature,
                    presence_penalties=(
                        presence_penalties[enabled_rows]
                        if presence_penalties is not None
                        else None
                    ),
                    output_token_ids=(
                        output_token_ids[enabled_rows]
                        if output_token_ids is not None
                        else None
                    ),
                    output_token_counts=output_token_counts,
                    presence_request_indices=(
                        presence_request_indices[enabled_rows]
                        if presence_request_indices is not None
                        else None
                    ),
                    embedding_bias=embedding_bias,
                    _hybrid_enabled=True,
                )
                local_values[enabled_rows] = enabled_values
                local_indices[enabled_rows] = enabled_indices
            if disabled_rows.numel() > 0:
                disabled_values, disabled_indices, _ = self._get_local_topk(
                    lm_head,
                    hidden_states[disabled_rows].contiguous(),
                    top_k=top_k,
                    temperature=temperature,
                    presence_penalties=(
                        presence_penalties[disabled_rows]
                        if presence_penalties is not None
                        else None
                    ),
                    output_token_ids=(
                        output_token_ids[disabled_rows]
                        if output_token_ids is not None
                        else None
                    ),
                    output_token_counts=output_token_counts,
                    presence_request_indices=(
                        presence_request_indices[disabled_rows]
                        if presence_request_indices is not None
                        else None
                    ),
                    embedding_bias=embedding_bias,
                    _hybrid_enabled=False,
                )
                local_values[disabled_rows] = disabled_values
                local_indices[disabled_rows] = disabled_indices
            return local_values, local_indices, shard_indices.org_vocab_start_index

        # Added-vocabulary shards have a split physical layout and are rejected
        # above; the runner performs the same gate before calling this method.
        hybrid_state = getattr(lm_head, "_hybrid_nvfp4_lm_head_state", None)
        hybrid_eligible = (
            hybrid_state is not None
            and self._is_contiguous_org_shard(lm_head)
            and not envs.VLLM_BATCH_INVARIANT
            and getattr(self, "hybrid_lm_head_enabled", True)
            and self.soft_cap is None
            and self.scale > 0.0
            and (self.head_dtype is None or self.head_dtype == hidden_states.dtype)
            and _hybrid_enabled is not False
            and not envs.VLLM_COMPUTE_NANS_IN_LOGITS
            and hybrid_state.can_use(
                hidden_states,
                bf16_weight=lm_head.weight,
                active_vocab_size=active_vocab_size,
                top_k=local_top_k,
            )
        )

        if hybrid_eligible:
            coarse_logits = hybrid_state.coarse_logits(
                hidden_states,
                embedding_bias,
            )
            if self.scale != 1.0:
                coarse_logits = coarse_logits.to(torch.float32) * self.scale
            self._mask_invalid_shard_logits(coarse_logits, active_vocab_size)
            if presence_penalties is not None:
                if (
                    output_token_counts is not None
                    and presence_request_indices is not None
                ):
                    self._apply_presence_penalty_from_counts(
                        coarse_logits,
                        presence_penalties,
                        output_token_counts,
                        presence_request_indices,
                        shard_indices=shard_indices,
                    )
                else:
                    assert output_token_ids is not None
                    self._apply_presence_penalty_from_token_ids(
                        coarse_logits,
                        presence_penalties,
                        output_token_ids,
                        shard_indices=shard_indices,
                    )
            candidate_indices = self._select_hybrid_candidates(
                hybrid_state,
                coarse_logits,
                local_top_k,
            )
            exact_logits = hybrid_state.refine_logits(
                hidden_states,
                lm_head.weight,
                candidate_indices,
                embedding_bias,
            ).to(torch.float32)
            if self.scale != 1.0:
                exact_logits.mul_(self.scale)
            if presence_penalties is not None:
                if (
                    output_token_counts is not None
                    and presence_request_indices is not None
                ):
                    self._apply_presence_penalty_from_counts(
                        exact_logits,
                        presence_penalties,
                        output_token_counts,
                        presence_request_indices,
                        shard_indices=shard_indices,
                        local_token_ids=candidate_indices,
                    )
                else:
                    assert output_token_ids is not None
                    self._apply_presence_penalty_from_token_ids(
                        exact_logits,
                        presence_penalties,
                        output_token_ids,
                        shard_indices=shard_indices,
                        local_token_ids=candidate_indices,
                    )
            if temperature != 1.0:
                exact_logits.div_(temperature)
            local_vals, positions = _stable_topk(
                exact_logits,
                local_top_k,
                candidate_indices,
            )
            return (
                local_vals,
                candidate_indices.gather(1, positions),
                shard_indices.org_vocab_start_index,
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias).to(
            torch.float32
        )
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        if presence_penalties is not None:
            if (
                output_token_counts is not None
                and presence_request_indices is not None
            ):
                self._apply_presence_penalty_from_counts(
                    logits,
                    presence_penalties,
                    output_token_counts,
                    presence_request_indices,
                    shard_indices=shard_indices,
                )
            else:
                assert output_token_ids is not None
                self._apply_presence_penalty_from_token_ids(
                    logits,
                    presence_penalties,
                    output_token_ids,
                    shard_indices=shard_indices,
                )
        if temperature != 1.0:
            logits.div_(temperature)
        self._mask_invalid_shard_logits(logits, active_vocab_size)
        local_vals, local_indices = _topk(logits, local_top_k)
        return local_vals, local_indices, shard_indices.org_vocab_start_index

    @staticmethod
    def _select_compact_topk_pairs(
        gathered_pairs: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return LogitsProcessor._select_compact_topk_values_ids(
            gathered_pairs[..., 0], gathered_pairs[..., 1].to(torch.int64), top_k, top_p
        )

    @staticmethod
    def _select_compact_topk_values_ids(
        candidate_values: torch.Tensor,
        candidate_ids: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select compact candidates without converting integer ids to FP32."""
        if candidate_values.shape != candidate_ids.shape:
            raise ValueError("candidate values and ids must have matching shapes")
        top_k = min(top_k, candidate_values.shape[-1])
        top_values, positions = _stable_topk(
            candidate_values,
            top_k,
            candidate_ids,
        )
        top_ids = candidate_ids.gather(-1, positions)
        if top_p < 1.0:
            probs = top_values.softmax(dim=-1, dtype=torch.float32)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            remove_mask = cumulative_probs - probs > top_p
            top_values = top_values.masked_fill(remove_mask, -float("inf"))
        return top_values, top_ids

    @staticmethod
    def _pack_topk_pairs(
        local_values: torch.Tensor,
        local_indices: torch.Tensor,
        vocab_start: int,
    ) -> torch.Tensor:
        """Pack local values and global token ids for TP communication."""
        global_indices = local_indices.to(torch.int64) + vocab_start
        return torch.stack(
            [local_values, global_indices.to(torch.float32)], dim=-1
        ).flatten(start_dim=-2)

    def _sample_compact_topk(
        self,
        top_values: torch.Tensor,
        top_ids: torch.Tensor,
        *,
        expanded_idx_mapping: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        temperature_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            (expanded_idx_mapping is None) != (seeds is None)
            or (expanded_idx_mapping is None) != (pos is None)
            or (expanded_idx_mapping is None) != (temperature_state is None)
        ):
            raise ValueError(
                "expanded_idx_mapping, seeds, pos, and temperature_state must be "
                "provided together"
            )
        if (
            expanded_idx_mapping is not None
            and seeds is not None
            and pos is not None
            and top_values.is_cuda
            and HAS_TRITON
        ):
            if temperature_state is None:
                raise ValueError(
                    "temperature_state is required with keyed Gumbel metadata"
                )
            sampled_positions = gumbel_sample(
                top_values,
                expanded_idx_mapping,
                temperature_state,
                seeds,
                pos,
                apply_temperature=False,
                use_fp64=False,
                token_keys=top_ids,
            )
            assert isinstance(sampled_positions, torch.Tensor)
            return top_ids.gather(-1, sampled_positions.unsqueeze(-1)).view(-1)

        probs = top_values.softmax(dim=-1, dtype=torch.float32)
        noise = torch.empty_like(probs)
        noise.exponential_()
        sampled_positions = probs.div(noise).argmax(dim=-1, keepdim=True)
        return top_ids.gather(-1, sampled_positions).view(-1)

    def get_topk_candidates(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        presence_penalties: torch.Tensor | None = None,
        output_token_ids: torch.Tensor | None = None,
        output_token_counts: torch.Tensor | None = None,
        presence_request_indices: torch.Tensor | None = None,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compact global top-k/top-p candidates.

        Only ``presence_penalty`` is supported by the hybrid path. The runner
        uses the same method for native fallback, while rejecting sampling
        features that require the full-vocabulary logits tensor.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        local_values, local_indices, vocab_start = self._get_local_topk(
            lm_head,
            hidden_states,
            top_k=top_k,
            temperature=temperature,
            presence_penalties=presence_penalties,
            output_token_ids=output_token_ids,
            output_token_counts=output_token_counts,
            presence_request_indices=presence_request_indices,
            embedding_bias=embedding_bias,
        )
        tp_size = lm_head.tp_size
        if self.vocab_size >= _MAX_FLOAT32_TOKEN_ID:
            local_ids = local_indices.to(torch.int64) + vocab_start
            if tp_size > 1:
                gathered_values = tensor_model_parallel_all_gather(
                    local_values, dim=-1
                )
                gathered_ids = tensor_model_parallel_all_gather(local_ids, dim=-1)
            else:
                gathered_values = local_values
                gathered_ids = local_ids
            return self._select_compact_topk_values_ids(
                gathered_values, gathered_ids, top_k, top_p
            )

        local_pairs = self._pack_topk_pairs(local_values, local_indices, vocab_start)
        if tp_size > 1:
            gathered_pairs = tensor_model_parallel_all_gather(local_pairs, dim=-1)
            gathered_pairs = gathered_pairs.view(
                hidden_states.shape[0], tp_size * local_values.shape[-1], 2
            )
        else:
            gathered_pairs = local_pairs.view(
                hidden_states.shape[0], local_values.shape[-1], 2
            )
        return self._select_compact_topk_pairs(gathered_pairs, top_k, top_p)

    def sample_topk_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        presence_penalties: torch.Tensor | None = None,
        output_token_ids: torch.Tensor | None = None,
        output_token_counts: torch.Tensor | None = None,
        presence_request_indices: torch.Tensor | None = None,
        embedding_bias: torch.Tensor | None = None,
        expanded_idx_mapping: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        temperature_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from a compact uniform top-k candidate set."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if temperature <= 0.0:
            raise ValueError(
                "sample_topk_tokens requires positive temperature; "
                f"got {temperature}"
            )
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        local_values, local_indices, vocab_start = self._get_local_topk(
            lm_head,
            hidden_states,
            top_k=top_k,
            temperature=temperature,
            presence_penalties=presence_penalties,
            output_token_ids=output_token_ids,
            output_token_counts=output_token_counts,
            presence_request_indices=presence_request_indices,
            embedding_bias=embedding_bias,
        )
        tp_size = lm_head.tp_size
        if self.vocab_size >= _MAX_FLOAT32_TOKEN_ID:
            local_ids = local_indices.to(torch.int64) + vocab_start
            if tp_size == 1:
                top_values, top_ids = self._select_compact_topk_values_ids(
                    local_values, local_ids, top_k, top_p
                )
                return self._sample_compact_topk(
                    top_values,
                    top_ids,
                    expanded_idx_mapping=expanded_idx_mapping,
                    seeds=seeds,
                    pos=pos,
                    temperature_state=temperature_state,
                )
            gathered_values = tensor_model_parallel_gather(
                local_values, dst=0, dim=-1
            )
            gathered_ids = tensor_model_parallel_gather(local_ids, dst=0, dim=-1)
        else:
            local_pairs = self._pack_topk_pairs(
                local_values, local_indices, vocab_start
            )
            gathered_values = tensor_model_parallel_gather(local_pairs, dst=0, dim=-1)

        if tp_size == 1:
            if self.vocab_size >= _MAX_FLOAT32_TOKEN_ID:
                top_values, top_ids = self._select_compact_topk_values_ids(
                    local_values, local_ids, top_k, top_p
                )
            else:
                gathered_pairs = local_pairs.view(
                    hidden_states.shape[0], local_values.shape[-1], 2
                )
                top_values, top_ids = self._select_compact_topk_pairs(
                    gathered_pairs, top_k, top_p
                )
            return self._sample_compact_topk(
                top_values,
                top_ids,
                expanded_idx_mapping=expanded_idx_mapping,
                seeds=seeds,
                pos=pos,
                temperature_state=temperature_state,
            )

        # Sampling must happen once and be broadcast.  Independent RNG draws
        # on TP ranks can otherwise produce different tokens even though the
        # candidate pairs are identical.
        tp_group = get_tp_group()
        if tp_group.rank_in_group == 0:
            assert gathered_values is not None
            if self.vocab_size >= _MAX_FLOAT32_TOKEN_ID:
                assert gathered_ids is not None
                gathered_values = gathered_values.view(
                    hidden_states.shape[0], tp_size * local_values.shape[-1]
                )
                gathered_ids = gathered_ids.view(
                    hidden_states.shape[0], tp_size * local_values.shape[-1]
                )
                top_values, top_ids = self._select_compact_topk_values_ids(
                    gathered_values, gathered_ids, top_k, top_p
                )
            else:
                gathered_pairs = gathered_values.view(
                    hidden_states.shape[0], tp_size * local_values.shape[-1], 2
                )
                top_values, top_ids = self._select_compact_topk_pairs(
                    gathered_pairs, top_k, top_p
                )
            sampled = self._sample_compact_topk(
                top_values,
                top_ids,
                expanded_idx_mapping=expanded_idx_mapping,
                seeds=seeds,
                pos=pos,
                temperature_state=temperature_state,
            )
        else:
            sampled = torch.empty(
                (hidden_states.shape[0],),
                dtype=torch.int64,
                device=hidden_states.device,
            )
        tp_group.broadcast(sampled, src=0)
        return sampled

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
