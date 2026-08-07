# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _aiter_fused_ar_rmsnorm,
    _unpack,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)


class ROCmLatentTailTier(IntEnum):
    """Which portable tail implementation the fused path runs, by token count.

    Mirrors CUDA's ``LatentTailTier`` minus its SM100-only tail-fusion tier,
    which relies on tcgen05 kernels with no ROCm equivalent. Both tiers here
    share the same replicated up-projection weight, so the choice is per batch
    and needs no weight relayout.
    """

    # ``_overlap_allreduce_tail``. The portable default, up to
    # VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD tokens: reduce the latent,
    # up-project the full hidden dim from the replicated weight, and add the
    # separately reduced shared output, hiding that shared all-reduce behind the
    # up-projection GEMM on the aux stream.
    ALLREDUCE_OVERLAP = 0

    # ``_shard_up_proj_tail``. Prefill-sized: each rank up-projects only its
    # hidden shard and accumulates into the shared partial, so the shared
    # all-reduce also stitches the routed shards. Same two all-reduces as the
    # overlap tier at 1/tp of the up-projection FLOPs; it gives up that tier's
    # aux-stream overlap, since the reduce now has to follow the accumulate.
    COLUMN_PARALLEL = 1


class ROCmLatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Mirrors CUDA's ``LatentMoERunner`` for the two portable tail tiers. The
    fused path (tp>1, un-reduced combine output, shared expert, no SP, unit
    routed scale) dispatches over ``ROCmLatentTailTier`` by token count; see
    that enum for what each tier does and when it applies.

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Overlap the shared-expert all-reduce with the tier-1 up-projection.
        self._shared_ar_events = (torch.cuda.Event(), torch.cuda.Event())
        self._logged_column_parallel = False
        self._logged_overlap_fallback = False

    def _use_fused_path(self) -> bool:
        # The fused path merges the latent and shared reductions into one
        # all-reduce, so it needs actual TP parallelism, a shared expert (to
        # concat), an un-reduced combine output, and no sequence parallelism.
        # It also assumes a unit routed scale: the tiers do not apply
        # routed_scaling_factor, so non-unit scales fall back to the base path.
        return (
            self.moe_config.tp_size > 1
            and self._shared_experts is not None
            and not self._fused_output_is_reduced
            and not self.moe_config.is_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )

    def _column_parallel_shardable(self) -> bool:
        transform = self.routed_output_transform
        up_proj = getattr(transform, "up_proj", None)
        return (
            up_proj is not None
            and up_proj.weight.shape[0] % self.moe_config.tp_size == 0
        )

    def _select_tail_tier(
        self,
        fused_output: torch.Tensor,
    ) -> ROCmLatentTailTier:
        num_tokens = fused_output.shape[0]
        # tier 1
        if (
            num_tokens <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            and not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
        ):
            return ROCmLatentTailTier.ALLREDUCE_OVERLAP
        # tier 2, when the up-projection rows divide evenly across ranks;
        # otherwise the overlap tier is correct at any size.
        if self._column_parallel_shardable():
            return ROCmLatentTailTier.COLUMN_PARALLEL
        if not self._logged_overlap_fallback:
            self._logged_overlap_fallback = True
            logger.warning_once(
                "K3 latent-MoE tail is not shardable under this config, "
                "falling back to the replicated up-projection.",
                scope="global",
            )
        return ROCmLatentTailTier.ALLREDUCE_OVERLAP

    def _allreduce_norm_latent_out(
        self,
        fused_output: torch.Tensor,
        norm: RMSNorm,
    ) -> torch.Tensor:
        """All-reduce the latent routed output and RMSNorm it.

        On ROCm the pair collapses into a single aiter fused kernel when the
        input is eligible; otherwise it falls back to a plain all-reduce
        followed by the RMSNorm. The zero residual makes the fused kernel
        compute ``rmsnorm(all_reduce(x))`` with nothing to add.
        """
        if self.moe_config.tp_size == 1:
            return norm(fused_output)

        if (
            _aiter_fused_ar_rmsnorm is not None
            and fused_output.is_cuda
            and fused_output.dim() == 2
            and fused_output.is_contiguous()
            and fused_output.dtype in (torch.bfloat16, torch.float16)
        ):
            normed, _ = _aiter_fused_ar_rmsnorm(
                input_=fused_output,
                residual=self._get_zero_residual(fused_output),
                weight=norm.weight.to(fused_output.dtype),
                epsilon=norm.variance_epsilon,
            )
            return normed

        return norm(tensor_model_parallel_all_reduce(fused_output))

    def _overlap_allreduce_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 1: reduce the latent, up-project the full hidden dim from the
        replicated weight, and add the separately reduced shared output.

        Small enough batches hide that shared all-reduce behind the up-projection
        GEMM on the aux stream.
        """
        transform = self.routed_output_transform
        assert transform is not None
        if transform.norm is not None:
            fused_latent = self._allreduce_norm_latent_out(fused_output, transform.norm)
        else:
            fused_latent = tensor_model_parallel_all_reduce(fused_output)

        # Overlap the shared-expert all-reduce with the up-projection GEMM while
        # the batch is small enough for it to pay off.
        result, shared_output = maybe_execute_in_parallel(
            lambda: torch.mm(fused_latent, transform.up_proj.weight.t()),
            lambda: tensor_model_parallel_all_reduce(shared_output),
            self._shared_ar_events[0],
            self._shared_ar_events[1],
            aux_stream(),
        )
        result.add_(shared_output)

        # Output is already fully reduced; this only strips padding.
        return self._maybe_reduce_final_output(
            result, trunc_size, output_is_reduced=True
        )

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """
        Tier 2: column-parallel up-projection folded into the final reduce.
        """
        if not self._logged_column_parallel:
            self._logged_column_parallel = True
            logger.info_once(
                "Kimi-K3 latent-MoE tail: up-projecting only this rank's "
                "hidden shard into the shared output.",
                scope="global",
            )

        transform = self.routed_output_transform
        assert transform is not None

        if transform.norm is not None:
            latent = self._allreduce_norm_latent_out(fused_output, transform.norm)
        else:
            latent = tensor_model_parallel_all_reduce(fused_output)

        weight = transform.up_proj.weight
        shard_size = weight.shape[0] // self.moe_config.tp_size
        shard_start = get_tensor_model_parallel_rank() * shard_size

        # column-parallel
        up_proj_shard = weight.narrow(0, shard_start, shard_size)
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_fused_path():
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = _unpack(result)
        assert shared_output is not None

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        tier = self._select_tail_tier(fused_output)
        if tier is ROCmLatentTailTier.ALLREDUCE_OVERLAP:
            latent_tail = self._overlap_allreduce_tail
        else:
            latent_tail = self._shard_up_proj_tail

        result = latent_tail(fused_output, shared_output, og_hidden_dim_post_xform)

        return self._maybe_add_zero_expert_output(result)
