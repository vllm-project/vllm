# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

logger = init_logger(__name__)


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
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
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
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)

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

    def get_prompt_logprobs(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        target_token_ids: torch.Tensor,
        num_logprobs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prompt log probabilities from TP-local LM-head shards.

        Args:
            lm_head: Unquantized vocabulary-parallel LM head.
            hidden_states: Prompt hidden states with shape ``[M, H]`` in BF16.
            target_token_ids: Global prompt target token IDs with shape ``[M]``.
            num_logprobs: Number of global top tokens to return, in ``[0, 32]``.

        Returns:
            Target and global top-K prompt log probabilities.
        """
        # The current register selection network and compact local state are
        # specialized for at most 32 candidates per row.
        if not 0 <= num_logprobs <= 32:
            raise ValueError("num_logprobs must be in [0, 32]")

        from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
            lm_head_logprobs,
            merge_tp_prompt_logprobs,
            prompt_target_logits,
        )

        hidden_states = hidden_states.contiguous()
        target_token_ids = target_token_ids.contiguous()
        shard_indices = lm_head.shard_indices
        vocab_start = shard_indices.org_vocab_start_index
        vocab_end = shard_indices.org_vocab_end_index
        target_is_local = (target_token_ids >= vocab_start) & (
            target_token_ids < vocab_end
        )
        local_target_ids = torch.where(
            target_is_local,
            target_token_ids - vocab_start,
            -1,
        )

        target_logits = prompt_target_logits(
            hidden_states,
            lm_head.weight,
            local_target_ids,
        )
        if lm_head.tp_size > 1 and hidden_states.shape[0] > 0:
            target_logits = tensor_model_parallel_all_reduce(target_logits)

        local_output = lm_head_logprobs(
            hidden_states,
            lm_head.weight,
            local_target_ids,
            target_logits,
            num_logprobs,
            valid_vocab_size=shard_indices.num_org_elements,
            global_vocab_start=vocab_start,
        )
        if hidden_states.shape[0] == 0:
            return merge_tp_prompt_logprobs(
                local_output.topk_values.unsqueeze(0),
                local_output.topk_ids.unsqueeze(0),
                local_output.lse.unsqueeze(0),
                local_output.rank_count.unsqueeze(0),
                target_token_ids,
                target_logits,
                num_logprobs,
            )

        # INT32 IDs and ranks are bit-cast into FP32 words so all compact state
        # can use one homogeneous all-gather without losing integer precision.
        compact = torch.cat(
            (
                local_output.topk_values,
                local_output.topk_ids.view(torch.float32),
                local_output.lse[:, None],
                local_output.rank_count.view(torch.float32)[:, None],
            ),
            dim=-1,
        )
        if lm_head.tp_size > 1:
            compact = tensor_model_parallel_all_gather(compact, dim=0)

        compact_width = 2 * num_logprobs + 2
        tp_compact = compact.view(
            lm_head.tp_size, hidden_states.shape[0], compact_width
        )
        tp_topk_values = tp_compact[..., :num_logprobs]
        tp_topk_ids = tp_compact[..., num_logprobs : 2 * num_logprobs].view(torch.int32)
        tp_local_lse = tp_compact[..., -2]
        tp_rank_count = tp_compact[..., -1:].view(torch.int32).squeeze(-1)
        return merge_tp_prompt_logprobs(
            tp_topk_values,
            tp_topk_ids,
            tp_local_lse,
            tp_rank_count,
            target_token_ids,
            target_logits,
            num_logprobs,
        )

    def validate_prompt_logprobs(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_dtype: torch.dtype,
    ) -> None:
        """Validate static requirements of the compact prompt-logprobs path."""
        # This path performs the LM-head projection itself. Precomputed logits
        # have no hidden states or local weight shard for the fused kernels.
        if self.logits_as_input:
            raise ValueError("prompt logprobs require an LM-head projection")

        # The fused projection currently emits the raw BF16 dot product. Match
        # forward() semantics by rejecting post-projection transforms and FP32
        # head overrides until the kernels implement them directly.
        if self.scale != 1.0 or self.soft_cap is not None:
            raise ValueError("prompt logprobs require unmodified LM-head logits")
        if self.head_dtype not in (None, torch.bfloat16):
            raise ValueError("prompt logprobs do not support an FP32 LM head")

        # The CuTe mainloop consumes a dense BF16 weight shard and uses BF16
        # Tensor Cores with FP32 accumulation. Quantized weight layouts and
        # other input dtypes require separate kernel implementations.
        if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
            raise TypeError("prompt logprobs require an unquantized LM head")
        if hidden_dtype != torch.bfloat16 or lm_head.weight.dtype != torch.bfloat16:
            raise TypeError("prompt logprobs require BF16 hidden states and LM head")

        # Although the target-only kernel can add bias, the tiled LM-head
        # kernel does not yet add it to every vocabulary logit. Allowing bias
        # would therefore produce inconsistent top-K, LSE, and rank results.
        if getattr(lm_head, "bias", None) is not None:
            raise ValueError("prompt logprobs do not support an LM-head bias")

        # Added vocabulary rows use a separate padded layout inside each shard.
        # The current global-ID mapping assumes one contiguous original-vocab
        # range per rank, so added rows must use the existing logits path.
        if lm_head.num_added_embeddings != 0:
            raise ValueError("prompt logprobs do not support added vocabulary entries")

        # Both components must describe the same global token-ID space; otherwise
        # padding removal and shard-local ID conversion would disagree.
        if self.org_vocab_size != lm_head.org_vocab_size:
            raise ValueError("logits processor and LM-head vocabulary sizes must match")

    def warmup_prompt_logprobs(self, lm_head: VocabParallelEmbedding) -> None:
        """Compile the compact prompt-logprobs CuTe specializations."""
        try:
            from vllm.model_executor.kernels.linear.cute_dsl.lm_head_logprobs import (
                validate_lm_head_logprobs_environment,
            )
        except ImportError as exc:
            raise RuntimeError(
                "VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS is enabled, but CuTe DSL "
                "dependencies could not be imported"
            ) from exc

        try:
            validate_lm_head_logprobs_environment(lm_head.weight)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS is enabled, but {exc}"
            ) from exc

        # M is symbolic in the CuTe kernel. A single row therefore compiles the
        # same shape specialization used by every runtime prompt chunk.
        hidden_states = torch.zeros(
            (1, lm_head.weight.shape[1]),
            dtype=torch.bfloat16,
            device=lm_head.weight.device,
        )
        target_token_ids = torch.zeros(
            1,
            dtype=torch.int64,
            device=lm_head.weight.device,
        )
        try:
            with torch.inference_mode():
                # Positive K values share the top-32 CuTe specialization.
                # Triton merge kernels remain lazily compiled per requested K.
                for num_logprobs in (0, 32):
                    self.get_prompt_logprobs(
                        lm_head,
                        hidden_states,
                        target_token_ids,
                        num_logprobs,
                    )
                torch.accelerator.synchronize(lm_head.weight.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "VLLM_USE_V2_COMPACT_PROMPT_LOGPROBS failed during startup "
                "kernel compilation"
            ) from exc

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
