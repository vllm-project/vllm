# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

logger = init_logger(__name__)


@dataclass(frozen=True)
class LocalLogitsMetadata:
    """Vocabulary layout for logits produced without a TP gather."""

    vocab_size: int
    vocab_start_index: int
    local_vocab_size: int


_local_logits_context: ContextVar[
    list[tuple[torch.Tensor, LocalLogitsMetadata]] | None
] = ContextVar("local_logits_context", default=None)


def compute_local_logits(
    compute_logits: Callable[[torch.Tensor], torch.Tensor | None],
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor | None, LocalLogitsMetadata | None]:
    """Compute local logits when supported, otherwise return full logits."""
    captures: list[tuple[torch.Tensor, LocalLogitsMetadata]] = []
    token = _local_logits_context.set(captures)
    try:
        logits = compute_logits(hidden_states)
    finally:
        _local_logits_context.reset(token)

    if len(captures) == 1 and isinstance(logits, torch.Tensor):
        captured_logits, metadata = captures[0]
        if logits is captured_logits and logits.shape[-1] == metadata.local_vocab_size:
            return logits, metadata

    if captures:
        logits = compute_logits(hidden_states)
    return logits, None


def _get_local_logits_metadata(
    logits_processor: "LogitsProcessor", lm_head: VocabParallelEmbedding
) -> LocalLogitsMetadata | None:
    """Return shard metadata when the active context can skip the TP gather."""
    context = _local_logits_context.get()
    if (
        context is None
        or hasattr(logits_processor, "base_layer")
        or getattr(lm_head, "tp_size", 1) <= 1
    ):
        return None

    shard_indices = getattr(lm_head, "shard_indices", None)
    required_indices = (
        "org_vocab_start_index",
        "num_elements_padded",
        "num_org_vocab_padding",
        "num_added_elements_padded",
    )
    if shard_indices is None or any(
        not hasattr(shard_indices, name) for name in required_indices
    ):
        return None
    if shard_indices.num_added_elements_padded > 0:
        return None

    return LocalLogitsMetadata(
        vocab_size=logits_processor.org_vocab_size,
        vocab_start_index=shard_indices.org_vocab_start_index,
        local_vocab_size=shard_indices.num_elements_padded,
    )


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
        return torch.topk(scores, k, dim=-1)
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

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        local_metadata = None
        if self.logits_as_input:
            logits = hidden_states
        else:
            if not skip_gather:
                local_metadata = _get_local_logits_metadata(self, lm_head)
                skip_gather = local_metadata is not None
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

            if local_metadata is not None:
                num_vocab_padding = lm_head.shard_indices.num_org_vocab_padding
                if num_vocab_padding > 0:
                    logits[..., -num_vocab_padding:] = -float("inf")
                context = _local_logits_context.get()
                assert context is not None
                context.append((logits, local_metadata))
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

        # A quant config that excludes lm_head hands out UnquantizedLinearMethod
        # rather than UnquantizedEmbeddingMethod, so accept both: either way the
        # weight is plain and `lm_head.weight` can be cast directly.
        if not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
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
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        tp_size = lm_head.tp_size

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

        # Convert shard-local indices to global vocab indices.
        vocab_start = lm_head.shard_indices.org_vocab_start_index
        global_indices = local_max_indices + vocab_start

        if tp_size == 1:
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        # [batch, 2] -> [batch, 2 * tp_size]
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

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

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
