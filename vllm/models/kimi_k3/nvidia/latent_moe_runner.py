# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner, _unpack
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)


class LatentTailTier(IntEnum):
    """Which tail implementation the fused path runs, by token count.

    The tiers share the same replicated up-projection weight, so the choice is
    per batch and needs no weight relayout: tiers 0 and 2 read it as a
    column-parallel row-shard, tier 1 reads it whole.
    """

    # ``_small_batch_tail``. Decode-sized (<= the op's max_num_tokens): one
    # CuTeDSL collective fuses the latent reduce, RMSNorm and the shared
    # reduce-scatter, then a sharded up-projection multicasts through a Lamport
    # copy. Requires SM100, TP 8/16, BF16
    TAIL_FUSION = 0

    # ``_overlap_allreduce_tail``. The portable default, up to
    # VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD tokens: reduce the latent,
    # up-project the full hidden dim from the replicated weight, and add the
    # separately reduced shared output, hiding that shared all-reduce behind the
    # up-projection GEMM on the aux stream.
    ALLREDUCE_OVERLAP = 1

    # ``_shard_up_proj_tail``. Prefill-sized: each rank up-projects only its
    # hidden shard and accumulates into the shared partial, so the shared
    # all-reduce also stitches the routed shards. Same two all-reduces as tier 1
    # at 1/tp of the up-projection FLOPs; it gives up tier 1's aux-stream
    # overlap, since the reduce now has to follow the accumulate.
    COLUMN_PARALLEL = 2


class LatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    The fused path (tp>1, un-reduced combine output, shared expert, no SP)
    dispatches over ``LatentTailTier`` by token count; see that enum for what
    each tier does and when it applies.

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # The tail-fusion kernels are tcgen05-based, so they require an
        # SM100 NVIDIA device; the runner falls back to the default latent
        # MoE path everywhere else.
        self.enable_k3_latent_moe_tail_fusion = (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        )
        # Overlap the shared-expert all-reduce with the tier-1 up-projection.
        self._shared_ar_events = (torch.cuda.Event(), torch.cuda.Event())
        use_fused_path = self._use_fused_path()
        if (
            self.enable_k3_latent_moe_tail_fusion
            and use_fused_path
            and self.moe_config.tp_size not in (8, 16)
        ):
            logger.warning_once(
                "K3 latent-MoE tail fusion currently supports TP=8 and TP=16, "
                "but TP=%d is configured. Falling back to the default path.",
                self.moe_config.tp_size,
            )
            self.enable_k3_latent_moe_tail_fusion = False

        if self.enable_k3_latent_moe_tail_fusion and use_fused_path:
            vllm_config = get_current_vllm_config()
            if vllm_config.parallel_config.use_ubatching:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support DBO or ubatching."
                )
            if vllm_config.model_config.enable_sleep_mode:
                raise ValueError(
                    "K3 latent-MoE tail fusion does not support sleep mode."
                )
            transform = self.routed_output_transform
            assert transform is not None
            norm = transform.norm
            assert norm is not None
            from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import (
                KimiK3LatentMoETailOp,
            )

            op = KimiK3LatentMoETailOp.initialize(
                hidden_size=transform.up_proj.weight.shape[0],
                latent_size=norm.weight.shape[0],
                dtype=norm.weight.dtype,
                device=norm.weight.device,
                rms_eps=norm.variance_epsilon,
            )
            self._k3_latent_moe_tail_op = op

    def _get_zero_residual(
        self,
        hidden_states: torch.Tensor,
        max_token_num: int,
    ) -> torch.Tensor:
        """Read-only zero ``residual_in`` for the fused AR+RMSNorm kernel.

        flashinfer requires a residual buffer even when there is no residual to
        add.
        """
        buf = getattr(self, "_zero_residual", None)
        if buf is None:
            buf = torch.zeros(
                max_token_num * hidden_states.shape[-1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            self._zero_residual = buf

        assert buf.dtype == hidden_states.dtype
        assert buf.device == hidden_states.device
        assert hidden_states.numel() <= buf.numel()

        return buf[: hidden_states.numel()].view_as(hidden_states)

    def _use_fused_path(self) -> bool:
        # The fused path merges the latent and shared reductions into one
        # all-reduce, so it needs actual TP parallelism, a shared expert (to
        # concat), an un-reduced combine output, and no sequence parallelism.
        return (
            self.moe_config.tp_size > 1
            and self._shared_experts is not None
            and not self._fused_output_is_reduced
            and not self.moe_config.is_sequence_parallel
        )

    def _select_tail_tier(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
    ) -> LatentTailTier:
        num_tokens = fused_output.shape[0]
        # tier 0
        if self.enable_k3_latent_moe_tail_fusion and (
            0 < num_tokens <= self._k3_latent_moe_tail_op.contract.max_num_tokens
        ):
            return LatentTailTier.TAIL_FUSION

        transform = self.routed_output_transform
        assert transform is not None
        # tier 1
        if (
            num_tokens <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            and not envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
        ):
            return LatentTailTier.ALLREDUCE_OVERLAP
        # tier 2
        return LatentTailTier.COLUMN_PARALLEL

    def _small_batch_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 0: the CuTeDSL operator fuses the whole tail."""
        transform = self.routed_output_transform
        assert transform is not None
        norm = transform.norm
        assert norm is not None

        result = self._k3_latent_moe_tail_op(
            fused_output,
            shared_output,
            norm.weight,
            transform.up_proj.weight,
        )
        # The operator already reduced; this only strips padding.
        return self._maybe_reduce_final_output(
            result, trunc_size, output_is_reduced=True
        )

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
        assert shared_output.size(0) <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
        if transform.norm is not None:
            fused_latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
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
        transform = self.routed_output_transform
        assert transform is not None

        if transform.norm is not None:
            latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
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
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
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

        tier = self._select_tail_tier(fused_output, shared_output)
        if tier is LatentTailTier.TAIL_FUSION:
            latent_tail = self._small_batch_tail
        elif tier is LatentTailTier.ALLREDUCE_OVERLAP:
            latent_tail = self._overlap_allreduce_tail
        else:
            latent_tail = self._shard_up_proj_tail

        result = latent_tail(fused_output, shared_output, og_hidden_dim_post_xform)

        return self._maybe_add_zero_expert_output(result)

    def allreduce_norm_latent_out(
        self,
        hidden_states: torch.Tensor,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce + add residual + (standard) RMSNorm, fused via flashinfer."""
        from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
            _AR_RESIDUAL_RMS_NORM,
            _can_use_flashinfer,
            flashinfer_trtllm_fused_allreduce_norm,
        )

        if self.moe_config.tp_size == 1:
            return norm(hidden_states)

        if flashinfer_trtllm_fused_allreduce_norm is not None:
            ok, max_token_num = _can_use_flashinfer(
                hidden_states, self.moe_config.tp_size
            )
            if ok:
                norm_out = torch.empty_like(hidden_states)
                # With norm_out provided, the kernel writes the new residual
                # (all_reduce(hidden_states) + residual) into the hidden_states
                # buffer and the normalized result into norm_out.
                flashinfer_trtllm_fused_allreduce_norm(
                    allreduce_in=hidden_states,
                    residual=self._get_zero_residual(hidden_states, max_token_num),
                    rms_gamma=norm.weight,
                    rms_eps=norm.variance_epsilon,
                    world_size=self.moe_config.tp_size,
                    weight_bias=0.0,
                    launch_with_pdl=True,
                    fp32_acc=True,
                    max_token_num=max_token_num,
                    pattern_code=_AR_RESIDUAL_RMS_NORM,
                    norm_out=norm_out,
                )
                return norm_out

        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return norm(reduced)
